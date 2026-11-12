# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
OmniSVG nodes on the ComfyUI text-encoder core inference flow.

  OmniSVG_Loader  : loads the OmniSVG checkpoint into a comfy-TE-core model
                    (Llama2_ + Qwen2-VL vision tower) and hands out a
                    VRAM-managed ModelPatcher wrapper.
  OmniSVG_Sampler : builds the Qwen2.5-VL chat prompt, encodes the reference
                    image with the core vision tower, decodes SVG tokens with
                    the core BaseGenerate loop, then detokenises to SVG and
                    rasterises it with pillow-lunasvg.

Outputs are plain ComfyUI data: an IMAGE tensor (the rendered SVG) plus the
saved .svg/.png paths and the raw SVG source. There is no custom web extension
any more — the built-in image preview shows the result.

The old wrapper path (decoder.py / inference.py / OmniSVG/deepsvg) is not
imported anywhere, which is what makes the plugin independent of the installed
transformers / diffusers versions and of the Cairo toolchain.
"""

import io as pyio
import os
import time

import numpy as np
import torch
from PIL import Image

import folder_paths
import comfy.model_management as mm
from comfy.text_encoders import qwen_vl
from comfy_api.latest import ComfyExtension, io

try:  # native IO.SVG payload (list of BytesIO); what ComfyUI's "Save SVG" node eats
    from comfy_api.latest._util import SVG as SVGData
except ImportError:  # pragma: no cover - fallback for a moved module
    class SVGData:  # minimal duck type: SaveSVGNode only reads `.data`
        def __init__(self, data):
            self.data = data

from .omnisvg_svg import (
    HAS_LUNASVG,
    SvgCodec,
    default_config_path,
    geometry_ok,
    pil_to_tensor,
    render_svg,
    svg_mean_brightness,
)
from .omnisvg_te import (
    IMAGE_PAD_TOKEN,
    SVG_EOS_TOKEN,
    OmniSVGWrapper,
    build_prompt_image,
    build_prompt_text,
    find_span,
    load_omnisvg_model,
)

MAX_SEED = np.iinfo(np.int32).max
NODE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_DIR = os.path.join(NODE_DIR, "OmniSVG", "Qwen2.5-VL-3B-Instruct")

IMAGE_RES = 448          # 16x16 merged patches -> 256 vision tokens

# generation presets from OmniSVG/config.yaml (same numbers the original used)
TASK_PRESETS = {
    "text-to-svg-icon": {"temperature": 0.5, "top_p": 0.88, "top_k": 50, "rep": 1.05},
    "text-to-svg-illustration": {"temperature": 0.6, "top_p": 0.90, "top_k": 60, "rep": 1.03},
    "image-to-svg": {"temperature": 0.3, "top_p": 0.90, "top_k": 50, "rep": 1.05},
}

_tokenizer = None
_model_cache = {}
_codec_cache = {}


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    return _tokenizer


def get_codec(model_size: str) -> SvgCodec:
    codec = _codec_cache.get(model_size)
    if codec is None:
        codec = SvgCodec(default_config_path(), model_size)
        _codec_cache[model_size] = codec
    return codec


def resolve_ckpt(ckpt: str) -> str:
    if ckpt == "none":
        raise ValueError("Select an OmniSVG checkpoint (drop it into ComfyUI/models/text_encoders)")
    full = folder_paths.get_full_path("text_encoders", ckpt)
    if full is None:
        raise ValueError(f"OmniSVG checkpoint not found in models/text_encoders: {ckpt}")
    return full


def get_model(ckpt: str, model_size: str, dtype: str) -> OmniSVGWrapper:
    full = resolve_ckpt(ckpt)
    mtime = os.path.getmtime(full)
    if dtype == "auto":
        dt = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    else:
        dt = torch.bfloat16 if dtype == "bf16" else torch.float16
    key = (full, mtime, model_size, str(dt))
    wrapper = _model_cache.get(key)
    if wrapper is None:
        _model_cache.clear()
        wrapper = load_omnisvg_model(full, model_size=model_size, dtype=dt)
        _model_cache[key] = wrapper
    return wrapper


# ---------------------------------------------------------------------- #
# image preprocessing
# ---------------------------------------------------------------------- #
def image_batch_to_pil(image: torch.Tensor, mask: torch.Tensor = None, size: int = IMAGE_RES):
    """
    ComfyUI IMAGE (B,H,W,C) + optional MASK (B,H,W) -> list of RGB PIL images.
    The mask marks the region to keep; everything else is flattened onto white
    (the original plugin dropped the *unmasked* region instead, which is the
    opposite of what a "convert this object" workflow expects).
    """
    if mask is not None and mask.ndim == 2:
        mask = mask.unsqueeze(0)
    out = []
    for i in range(image.shape[0]):
        arr = (image[i].detach().cpu().float().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr[:, :, :3], "RGB")
        if mask is not None:
            m = mask[min(i, mask.shape[0] - 1)].detach().cpu().float().numpy()
            m = np.clip(m, 0.0, 1.0)
            m_img = Image.fromarray((m * 255).astype(np.uint8), "L").resize(pil.size, Image.BILINEAR)
            alpha = np.asarray(m_img, dtype=np.float32) / 255.0
            rgb = np.asarray(pil, dtype=np.float32)
            rgb = rgb * alpha[:, :, None] + 255.0 * (1.0 - alpha[:, :, None])
            pil = Image.fromarray(rgb.astype(np.uint8), "RGB")
        out.append(pil.resize((size, size), Image.LANCZOS))
    return out


ICON_KEYWORDS = ('icon', 'logo', 'symbol', 'badge', 'button', 'emoji', 'glyph', 'simple',
                 'arrow', 'triangle', 'circle', 'square', 'heart', 'star', 'checkmark')
ILLUSTRATION_KEYWORDS = (
    'illustration', 'scene', 'person', 'people', 'character', 'man', 'woman', 'boy', 'girl',
    'avatar', 'portrait', 'face', 'head', 'body', 'cat', 'dog', 'bird', 'animal', 'pet', 'fox',
    'rabbit', 'sitting', 'standing', 'walking', 'running', 'sleeping', 'holding', 'playing',
    'house', 'building', 'tree', 'garden', 'landscape', 'mountain', 'forest', 'city', 'ocean',
    'beach', 'sunset', 'sunrise', 'sky')


def detect_text_subtype(prompt: str) -> str:
    """Original heuristic: icons are short and keyword driven, the rest is art."""
    text = prompt.lower()
    if any(k in text for k in ICON_KEYWORDS):
        return "icon"
    if sum(1 for k in ILLUSTRATION_KEYWORDS if k in text) >= 1 or len(prompt) > 50:
        return "illustration"
    return "icon"


def is_valid_candidate(metrics: dict, img, subtype: str):
    """
    Original acceptance gate: the result must not be a blank page.

    With a raster backend the exact original test is used (mean brightness of
    the render). Without one we fall back to the renderer-free geometry check,
    which cannot distinguish a giant white rectangle from real art.
    """
    if metrics is None:
        return False, "no_geometry"
    if "<svg" not in metrics["svg"] or len(metrics["svg"]) < 20:
        return False, "too_short"
    if img is None:
        return geometry_ok(metrics)
    threshold = 250 if subtype == "illustration" else 252
    if svg_mean_brightness(img) > threshold:
        return False, "empty_image"
    return True, "ok"


# ---------------------------------------------------------------------- #
# inference
# ---------------------------------------------------------------------- #
def encode_prompt(wrapper: OmniSVGWrapper, prompt: str, pil_image=None):
    """Returns (embeds, position_ids) ready for BaseGenerate.generate()."""
    te = wrapper.te
    device = wrapper.device
    tok = get_tokenizer()

    span_embeds = None
    grid = None
    if pil_image is not None:
        arr = torch.from_numpy(np.asarray(pil_image, dtype=np.float32) / 255.0).unsqueeze(0)
        span_embeds, grid = te.encode_image(arr, device)
        text = build_prompt_image(int(span_embeds.size(0)))
    else:
        text = build_prompt_text(prompt)

    input_ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    embeds = te.model.embed_tokens(input_ids).to(wrapper.execution_dtype())

    position_ids = None
    if span_embeds is not None:
        start, end = find_span(input_ids, IMAGE_PAD_TOKEN)
        if end - start != span_embeds.size(0):
            raise ValueError(f"vision span {end - start} != vision embeds {span_embeds.size(0)}")
        embeds[0, start:end] = span_embeds.to(embeds.dtype)
        position_ids = qwen_vl.qwen2vl_mrope_position_ids(
            [{"type": "image", "index": start, "size": end - start, "extra": grid}],
            embeds.shape[1], device)
    print(f"[OmniSVG] prompt tokens: {input_ids.shape[1]}"
          + (f" (incl. {span_embeds.size(0)} vision tokens, grid {grid.tolist()})" if grid is not None else ""))
    return embeds, position_ids


def generate_svgs(wrapper: OmniSVGWrapper, prompt: str, pil_images, max_length: int,
                  temperature: float, top_p: float, top_k: int, rep_penalty: float,
                  num_candidates: int, seed: int, render_size: int, target_subtype: str):
    """
    Generates up to `num_candidates` accepted SVGs; mirrors the original
    generate_candidates(): draw `num_candidates + extra` samples and keep the
    first ones that pass the blank-page gate.
    """
    codec = get_codec(wrapper.model_size)
    te = wrapper.te
    do_sample = temperature > 0.0
    extra = 4  # EXTRA_CANDIDATES_BUFFER from config.yaml
    total = num_candidates + extra

    results = []
    for img_idx, pil in enumerate(pil_images):
        embeds, position_ids = encode_prompt(wrapper, prompt, pil)
        accepted = 0
        t0 = time.time()
        for i in range(total):
            sample_seed = seed + i * 1009
            # comfy's TransformerBlock does `output = x` then
            # `torch.add(residual, x, out=output)`, so layer 0 writes its result
            # straight back into the tensor it was handed. For the prefill that
            # tensor is our `embeds`, i.e. every sample after the first would be
            # conditioned on the *mutated* embeds (the first one is fine because
            # it is only read before being written). Hand each sample its own copy.
            sample_embeds = embeds.clone()
            # nested inside ComfyUI's own inference_mode; keeps the out= writes in
            # comfy's decode loop valid when the node is driven from a script too
            with torch.inference_mode():
                ids = te.generate(
                    embeds=sample_embeds,
                    position_ids=position_ids,
                    do_sample=do_sample,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=0.0,
                    repetition_penalty=rep_penalty,
                    seed=sample_seed,
                    stop_tokens=[SVG_EOS_TOKEN],
                    execution_dtype=wrapper.execution_dtype(),
                )
            metrics = codec.detokenise(ids)
            if metrics is None:
                print(f"[OmniSVG] sample {i}: {len(ids)} tokens -> no SVG paths")
                continue
            rendered = render_svg(metrics["svg"], render_size)
            ok, reason = is_valid_candidate(metrics, rendered, target_subtype)
            if not ok:
                print(f"[OmniSVG] sample {i}: {len(ids)} tokens -> rejected ({reason})")
                continue
            accepted += 1
            print(f"[OmniSVG] sample {i}: {len(ids)} tokens -> accepted "
                  f"({len(metrics['svg'])} bytes, {metrics['subpaths']} subpaths, seed {sample_seed})")
            results.append({"svg": metrics["svg"], "img": rendered, "ids": ids,
                            "seed": sample_seed, "image_index": img_idx})
            if accepted >= num_candidates:
                break
        print(f"[OmniSVG] image {img_idx + 1}/{len(pil_images)}: "
              f"{accepted} candidate(s) in {time.time() - t0:.1f}s")
    return results


# ---------------------------------------------------------------------- #
# nodes
# ---------------------------------------------------------------------- #
class OmniSVG_Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        options = ["none"] + folder_paths.get_filename_list("text_encoders")
        # preselect an OmniSVG checkpoint so a fresh node is ready to run
        default = next((n for n in options if "omnisvg" in n.lower()), "none")
        return io.Schema(
            node_id="OmniSVG_Loader",
            display_name="OmniSVG_Loader",
            category="OmniSVG",
            inputs=[
                io.Combo.Input("ckpt", options=options, default=default,
                               tooltip="OmniSVG1.1 checkpoint from ComfyUI/models/text_encoders"),
                io.Combo.Input("model_size", options=["auto", "4B", "8B"], default="auto",
                               tooltip="auto reads the hidden size from the checkpoint "
                                       "(2048 = 4B / Qwen2.5-VL-3B, 3584 = 8B / Qwen2.5-VL-7B)"),
                io.Combo.Input("dtype", options=["auto", "bf16", "fp16"], default="auto"),
            ],
            outputs=[io.Custom("OMNISVG_TE").Output(display_name="model")],
        )

    @classmethod
    def execute(cls, ckpt, model_size, dtype) -> io.NodeOutput:
        wrapper = get_model(ckpt, model_size, dtype)
        print(f"[OmniSVG] model ready ({wrapper.model_size}, {wrapper.execution_dtype()})")
        return io.NodeOutput(wrapper)


class OmniSVG_Sampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OmniSVG_Sampler",
            display_name="OmniSVG_Sampler",
            category="OmniSVG",
            inputs=[
                io.Custom("OMNISVG_TE").Input("model"),
                io.String.Input("prompt", multiline=True,
                                default="A yellow t-shirt with a heart design represents love and positivity."),
                io.Combo.Input("subtype", options=["auto", "illustration", "icon"], default="auto",
                               tooltip="Text task preset (ignored for image-to-svg)"),
                io.Image.Input("image", optional=True,
                               tooltip="Connect an image for image-to-svg; leave empty for text-to-svg"),
                io.Mask.Input("mask", optional=True,
                              tooltip="Optional: keep only the masked region, the rest goes white"),
                io.Int.Input("max_length", default=512, min=64, max=2048, step=32,
                             display_mode=io.NumberDisplay.number,
                             tooltip="Max new tokens. 384-768 is plenty for one icon/illustration; "
                                     "the model often misses EOS and then keeps inventing garbage, "
                                     "so a large value only buys you a longer (and noisier) tail"),
                io.Boolean.Input("auto_params", default=True,
                                 tooltip="Use the per-task presets from config.yaml "
                                         "(temp/top_p/top_k/rep); disable to use the widgets below"),
                io.Float.Input("temperature", default=0.5, min=0.0, max=1.5, step=0.01,
                               display_mode=io.NumberDisplay.number),
                io.Float.Input("top_p", default=0.9, min=0.0, max=1.0, step=0.01,
                               display_mode=io.NumberDisplay.number),
                io.Int.Input("top_k", default=50, min=0, max=1000, step=1,
                             display_mode=io.NumberDisplay.number),
                io.Float.Input("rep_penalty", default=1.05, min=0.5, max=3.0, step=0.01,
                               display_mode=io.NumberDisplay.number),
                io.Int.Input("num_candidates", default=1, min=1, max=8, step=1,
                             display_mode=io.NumberDisplay.number),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED, control_after_generate="randomize"),
                io.Int.Input("render_size", default=1024, min=128, max=4096, step=64,
                             display_mode=io.NumberDisplay.number),
            ],
            outputs=[
                # native ComfyUI SVG type -> feed the built-in "Save SVG" node
                # (it also previews the file in the UI via image/svg+xml)
                io.SVG.Output(display_name="svg"),
                # rasterised preview of the same document (needs pillow-lunasvg)
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, model, prompt, subtype, max_length, auto_params, temperature, top_p,
                top_k, rep_penalty, num_candidates, seed, render_size,
                image=None, mask=None) -> io.NodeOutput:
        wrapper: OmniSVGWrapper = model
        mm.load_models_gpu([wrapper.patcher])

        is_image_task = isinstance(image, torch.Tensor)
        if is_image_task:
            task_key = "image-to-svg"
        else:
            task_key = f"text-to-svg-{subtype if subtype != 'auto' else detect_text_subtype(prompt)}"
        if auto_params:
            preset = TASK_PRESETS[task_key]
            temperature, top_p = preset["temperature"], preset["top_p"]
            top_k, rep_penalty = preset["top_k"], preset["rep"]
        target_subtype = "image" if is_image_task else task_key.rsplit("-", 1)[-1]
        print(f"[OmniSVG] task={task_key} temp={temperature} top_p={top_p} "
              f"top_k={top_k} rep={rep_penalty} max_length={max_length}")

        pil_images = image_batch_to_pil(image, mask) if is_image_task else [None]
        results = generate_svgs(wrapper, prompt, pil_images, max_length,
                                temperature, top_p, top_k, rep_penalty,
                                num_candidates, seed, render_size, target_subtype)

        if not results:
            print("[OmniSVG] no valid SVG produced")
            return io.NodeOutput(SVGData([]), torch.ones((1, 1, 1, 3)))

        svg_batch = SVGData([pyio.BytesIO(r["svg"].encode("utf-8")) for r in results])

        if not HAS_LUNASVG:
            print("[OmniSVG] pillow-lunasvg is not installed -> no IMAGE output "
                  "(the `svg` output is unaffected; connect the built-in 'Save SVG' node)")
            return io.NodeOutput(svg_batch, torch.ones((1, 1, 1, 3)))

        tensors = [pil_to_tensor(r["img"]) for r in results if r["img"] is not None]
        if not tensors:
            return io.NodeOutput(svg_batch, torch.ones((1, 1, 1, 3)))
        out_images = torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]

        mm.soft_empty_cache()
        return io.NodeOutput(svg_batch, out_images)



