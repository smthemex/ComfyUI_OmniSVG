# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
OmniSVG on the ComfyUI text-encoder core inference flow.

This replaces the old HF-transformers wrapper (decoder.py SketchDecoder ->
Qwen2_5_VLForConditionalGeneration + accelerate + qwen_vl_utils) with ComfyUI's
own text-encoder core, so the plugin survives transformers / diffusers API
changes and inherits comfy's VRAM management:

  comfy.text_encoders.llama.Llama2_       Qwen2.5 LLM body, state-dict keys are
                                          1:1 with the HF Qwen2.5-VL checkpoint
  comfy.text_encoders.llama.BaseGenerate  core AR decode loop: KV cache,
                                          prefill/decode, temperature / top_k /
                                          top_p / min_p / repetition_penalty,
                                          seed, stop tokens, progress bar
  comfy.text_encoders.qwen_vl             Qwen2-VL vision tower + image
                                          preprocessing + 3D MRoPE position ids

No ComfyUI source file is touched; everything below subclasses / imports from
the public `comfy` package.

Model = Qwen2.5-VL-3B ("OmniSVG1.1_4B", 4B params incl. the extended vocab)
with `tie_word_embeddings=True`, vocab extended to 197000
(151936 Qwen tokens + SVG coordinate / command / colour tokens).

Verified against D:/Downloads/OmniSVG1.1_4B/pytorch_model.bin:
  824 checkpoint keys (after stripping the "transformer." wrapper prefix)
  <-> 824 module keys of Llama2_ + Qwen2VLVisionTransformer:
  0 missing, 0 unexpected, 0 shape mismatch.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple

import torch

import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.utils
from comfy.text_encoders import llama as comfy_llama
from comfy.text_encoders import qwen_vl

# SVG special tokens (config.yaml -> model.*)
SVG_BOS_TOKEN = 196998
SVG_EOS_TOKEN = 196999
SVG_PAD_TOKEN = 151643
SVG_VOCAB_SIZE = 197000      # 151936 Qwen tokens + SVG coord / command / colour tokens

# Qwen2.5-VL multimodal placeholder
IMAGE_PAD_TOKEN = 151655


@dataclass
class OmniSVG4BConfig:
    """Qwen2.5-VL-3B backbone + 197000 SVG vocab (OmniSVG1.1_4B/config.json)."""

    vocab_size: int = 197000
    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    max_position_embeddings: int = 128000
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = True
    rope_dims = [16, 24, 24]          # Qwen2.5-VL 3D MRoPE section layout
    q_norm = None
    k_norm = None
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False             # tied embeddings -> BaseGenerate.logits uses embed_tokens
    stop_tokens = [SVG_EOS_TOKEN]


@dataclass
class OmniSVG8BConfig:
    """Qwen2.5-VL-7B backbone + 197000 SVG vocab (OmniSVG1.1_8B)."""

    vocab_size: int = 197000
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    max_position_embeddings: int = 128000
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    transformer_type: str = "llama"
    head_dim = 128
    rms_norm_add = False
    mlp_activation = "silu"
    qkv_bias = True
    rope_dims = [16, 24, 24]
    q_norm = None
    k_norm = None
    rope_scale = None
    final_norm: bool = True
    lm_head: bool = False
    stop_tokens = [SVG_EOS_TOKEN]


MODEL_CONFIGS = {"4B": OmniSVG4BConfig, "8B": OmniSVG8BConfig}


class OmniSVGTE(comfy_llama.BaseLlama, comfy_llama.BaseGenerate, torch.nn.Module):
    """
    OmniSVG generator on the ComfyUI TE core.

    Skeleton mirrors comfy.text_encoders.llama.Qwen25_7BVLI (the official
    "Qwen2.5-VL + BaseGenerate" reference used by Qwen-Image), only the config
    and the vision tower widths change.
    """

    def __init__(self, config_dict=None, dtype=None, device=None, operations=None):
        super().__init__()
        cfg = dict(config_dict or {})
        model_size = cfg.pop("model_size", "4B")
        config = MODEL_CONFIGS[model_size](**cfg)
        self.config = config
        self.num_layers = config.num_hidden_layers

        operations = operations or comfy.ops.manual_cast
        self.model = comfy_llama.Llama2_(config, device=device, dtype=dtype, ops=operations)
        # Qwen2.5-VL vision tower: hidden 1280 -> LLM hidden, depth 32,
        # 4 full-attention blocks (7/15/23/31), patch 14, temporal 2, merge 2.
        self.visual = qwen_vl.Qwen2VLVisionTransformer(
            hidden_size=1280, output_hidden_size=config.hidden_size,
            device=device, dtype=dtype, ops=operations)
        self.dtype = dtype

    # ------------------------------------------------------------------ #
    # vision encoding
    # ------------------------------------------------------------------ #
    def encode_image(self, image: torch.Tensor, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        image: (1, H, W, 3) float32 in [0, 1].
        Returns (image_embeds (n_tokens, hidden), grid_thw (1, 3)).

        `process_qwen2vl_images` performs the Qwen2.5-VL smart_resize +
        mean/std normalisation + patchify; the tower itself runs in float32
        (same numeric path comfy uses for Qwen-Image's Qwen2.5-VL encoder).
        """
        patches, grid = qwen_vl.process_qwen2vl_images(image)
        return self.visual(patches.to(device, dtype=torch.float32), grid), grid


class OmniSVGWrapper:
    """What the loader node hands to the sampler node."""

    def __init__(self, te: OmniSVGTE, patcher: comfy.model_patcher.ModelPatcher,
                 model_size: str, missing_keys: List[str], unexpected_keys: List[str]):
        self.te = te
        self.patcher = patcher
        self.model_size = model_size
        self.missing_keys = missing_keys
        self.unexpected_keys = unexpected_keys

    @property
    def device(self):
        return mm.text_encoder_device()

    def execution_dtype(self) -> torch.dtype:
        device = self.device
        if mm.should_use_bf16(device):
            return torch.bfloat16
        return torch.float32


def _strip_prefix(sd: dict) -> dict:
    """The official checkpoint is saved from the old SketchDecoder wrapper and
    carries a "transformer." prefix on every key."""
    first_key = next(iter(sd))
    if first_key.startswith("transformer."):
        return {k[len("transformer."):]: v for k, v in sd.items() if k.startswith("transformer.")}
    return sd


def detect_model_size(sd: dict) -> str:
    """
    Pick the config from the checkpoint itself instead of trusting a filename.
    embed_tokens is (vocab, hidden): 2048 -> 4B (Qwen2.5-VL-3B),
    3584 -> 8B (Qwen2.5-VL-7B).  Cheap: the tensors are still mmap'd here.
    """
    w = sd.get("model.embed_tokens.weight")
    if w is None:
        return "4B"
    hidden = int(w.shape[1])
    if hidden == 3584:
        return "8B"
    if hidden != 2048:
        raise ValueError(f"unsupported OmniSVG hidden size {hidden} "
                         f"(expected 2048 for 4B or 3584 for 8B)")
    return "4B"


def load_omnisvg_model(checkpoint_path: str, model_size: str = "auto",
                       dtype: torch.dtype = torch.bfloat16) -> OmniSVGWrapper:
    """
    Load the OmniSVG checkpoint into OmniSVGTE and register it with comfy's
    model management (VRAM placement / offload / weight casting).

    Key mapping vs the original checkpoint:
      - "transformer." wrapper prefix stripped;
      - "model.*" / "visual.*" map 1:1 onto comfy Llama2_ / VisionTransformer;
      - "lm_head.weight" dropped (tie_word_embeddings -> embed_tokens reused).
    """
    print(f"[OmniSVG] loading checkpoint {checkpoint_path}")
    sd = comfy.utils.load_torch_file(checkpoint_path)
    sd = _strip_prefix(sd)
    sd.pop("lm_head.weight", None)  # tied to embed_tokens

    # fail fast when someone picks an unrelated file from models/text_encoders.
    # OmniLottie / plain Qwen2.5-VL encoders share the same key layout, so the
    # vocabulary size is the reliable discriminator (OmniSVG1.1 = 197000).
    if "model.embed_tokens.weight" not in sd or "visual.patch_embed.proj.weight" not in sd:
        raise ValueError(
            f"{os.path.basename(checkpoint_path)} is not an OmniSVG checkpoint "
            f"(no model.embed_tokens / visual.patch_embed keys after stripping the "
            f"'transformer.' prefix)")
    emb = sd["model.embed_tokens.weight"]
    if int(emb.shape[0]) != SVG_VOCAB_SIZE:
        raise ValueError(
            f"{os.path.basename(checkpoint_path)} is not an OmniSVG1.1 checkpoint: "
            f"vocab {int(emb.shape[0])} != {SVG_VOCAB_SIZE}. "
            f"(other Qwen2.5-VL based encoders such as OmniLottie share the key layout)")

    if model_size not in MODEL_CONFIGS:
        model_size = detect_model_size(sd)
        print(f"[OmniSVG] detected {model_size} from checkpoint "
              f"(embed_tokens hidden {int(emb.shape[1])})")

    te = OmniSVGTE({"model_size": model_size}, dtype=dtype, device="cpu",
                   operations=comfy.ops.manual_cast)
    # strict=True is the self-check: keys were verified 1:1 against the real file
    te.load_state_dict(sd, strict=True, assign=True)
    # assign=True keeps the checkpoint tensors' requires_grad; the core decode
    # loop writes through `out=` and would abort with an autograd error outside
    # ComfyUI's inference_mode (e.g. when driven by the HTTP API or a script).
    te.requires_grad_(False)
    te.eval()
    del sd

    load_device = mm.text_encoder_device()
    offload_device = mm.text_encoder_offload_device()
    patcher = comfy.model_patcher.ModelPatcher(te, load_device=load_device,
                                               offload_device=offload_device)
    print(f"[OmniSVG] {model_size} loaded on comfy TE core "
          f"(load_device={load_device}, offload={offload_device})")
    return OmniSVGWrapper(te, patcher, model_size, [], [])


# ---------------------------------------------------------------------- #
# prompt construction
# ---------------------------------------------------------------------- #
# Verbatim from inference.py (note the trailing space after "generator.")
OMNISVG_SYSTEM_PROMPT = """You are an expert SVG code generator. 
Generate precise, valid SVG path commands that accurately represent the described scene or object.
Focus on capturing key shapes, spatial relationships, and visual composition."""


def render_chat(system_prompt: str, user_content: str) -> str:
    """Qwen2.5-VL chat template with add_generation_prompt=True."""
    return (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def build_prompt_text(prompt: str) -> str:
    """
    Byte-exact replica of the original inference.prepare_inputs() text branch
    (the indentation inside the f-string is part of the trained prompt).
    """
    instruction = f"""Generate an SVG illustration for: {prompt}
        Requirements:
        - Create complete SVG path commands
        - Include proper coordinates and colors
        - Maintain visual clarity and composition"""
    return render_chat(OMNISVG_SYSTEM_PROMPT, instruction)


def build_prompt_image(n_pad_tokens: int) -> str:
    """Original image branch: text instruction, then the vision block."""
    media = "<|vision_start|>" + ("<|image_pad|>" * n_pad_tokens) + "<|vision_end|>"
    instruction = "Generate SVG code that accurately represents this image:" + media
    return render_chat(OMNISVG_SYSTEM_PROMPT, instruction)


def find_span(input_ids: torch.Tensor, pad_token: int) -> Tuple[int, int]:
    ids = input_ids[0]
    idx = (ids == pad_token).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        raise ValueError(f"No {pad_token} placeholder tokens found in prompt")
    return int(idx[0]), int(idx[-1]) + 1
