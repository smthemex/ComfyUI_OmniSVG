import os
from dataclasses import dataclass

import torch
import json
import cv2
import numpy as np
from PIL import Image
from safetensors.torch import load_file as load_sft
import gc

import torch
import torch.nn as nn
import logging
import os
from typing import Dict, List, Optional, Union
from accelerate.big_modeling import dispatch_model
import logging
from accelerate.utils import (
    find_tied_parameters,
    get_balanced_memory,
    infer_auto_device_map,
    retie_parameters, 
)
from accelerate.utils.offload import  offload_weight, save_offload_index
from accelerate.utils.modeling import (
    check_tied_parameters_in_config,
    check_tied_parameters_on_same_device,
    set_module_tensor_to_device,
    load_offloaded_weights
)
from accelerate.utils.constants import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
import shutil
import tempfile
logger = logging.getLogger(__name__)

def load_checkpoint_and_dispatch_(
    model: nn.Module,
    sd: Union[str, os.PathLike],
    device_map: Optional[Union[str, Dict[str, Union[int, str, torch.device]]]] = None,
    max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
    no_split_module_classes: Optional[List[str]] = None,
    offload_folder: Optional[Union[str, os.PathLike]] = None,
    offload_buffers: bool = False,
    dtype: Optional[Union[str, torch.dtype]] = None,
    offload_state_dict: Optional[bool] = None,
    skip_keys: Optional[Union[str, List[str]]] = None,
    preload_module_classes: Optional[List[str]] = None,
    force_hooks: bool = False,
    strict: bool = False,
):
    """
    Loads a (potentially sharded) checkpoint inside a model, potentially sending weights to a given device as they are
    loaded and adds the various hooks that will make this model run properly (even if split across devices).

    Args:
        model (`torch.nn.Module`): The model in which we want to load a checkpoint.
        sd (`dict`):
        # checkpoint (`str` or `os.PathLike`):
        #     The folder checkpoint to load. It can be:
        #     - a path to a file containing a whole model state dict
        #     - a path to a `.json` file containing the index to a sharded checkpoint
        #     - a path to a folder containing a unique `.index.json` file and the shards of a checkpoint.
        device_map (`Dict[str, Union[int, str, torch.device]]`, *optional*):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.

            To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For more
            information about each option see [here](../concept_guides/big_model_inference#designing-a-device-map).
            Defaults to None, which means [`dispatch_model`] will not be called.
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available for each GPU
            and the available CPU RAM if unset.
        no_split_module_classes (`List[str]`, *optional*):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        offload_folder (`str` or `os.PathLike`, *optional*):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            In the layers that are offloaded on the CPU or the hard drive, whether or not to offload the buffers as
            well as the parameters.
        dtype (`str` or `torch.dtype`, *optional*):
            If provided, the weights will be converted to that type when loaded.
        offload_state_dict (`bool`, *optional*):
            If `True`, will temporarily offload the CPU state dict on the hard drive to avoid getting out of CPU RAM if
            the weight of the CPU state dict + the biggest shard does not fit. Will default to `True` if the device map
            picked contains `"disk"` values.
        skip_keys (`str` or `List[str]`, *optional*):
            A list of keys to ignore when moving inputs or outputs between devices.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
        force_hooks (`bool`, *optional*, defaults to `False`):
            Whether or not to force device hooks to be attached to the model even if all layers are dispatched to a
            single device.
        strict (`bool`, *optional*, defaults to `False`):
            Whether to strictly enforce that the keys in the checkpoint state_dict match the keys of the model's
            state_dict.

    Example:

    ```python
    >>> from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    >>> from huggingface_hub import hf_hub_download
    >>> from transformers import AutoConfig, AutoModelForCausalLM

    >>> # Download the Weights
    >>> checkpoint = "EleutherAI/gpt-j-6B"
    >>> weights_location = hf_hub_download(checkpoint, "pytorch_model.bin")

    >>> # Create a model and initialize it with empty weights
    >>> config = AutoConfig.from_pretrained(checkpoint)
    >>> with init_empty_weights():
    ...     model = AutoModelForCausalLM.from_config(config)

    >>> # Load the checkpoint and dispatch it to the right devices
    >>> model = load_checkpoint_and_dispatch(
    ...     model, weights_location, device_map="auto", no_split_module_classes=["GPTJBlock"]
    ... )
    ```
    """
    if isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
        raise ValueError(
            "If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or "
            "'sequential'."
        )
    if isinstance(device_map, str):
        if device_map != "sequential":
            max_memory = get_balanced_memory(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                dtype=dtype,
                low_zero=(device_map == "balanced_low_0"),
            )
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=no_split_module_classes,
            dtype=dtype,
            offload_buffers=offload_buffers,
        )
    if offload_state_dict is None and device_map is not None and "disk" in device_map.values():
        offload_state_dict = True
    load_checkpoint_in_model_(
        model,
        sd,
        device_map=device_map,
        offload_folder=offload_folder,
        dtype=dtype,
        offload_state_dict=offload_state_dict,
        offload_buffers=offload_buffers,
        strict=strict,
    )
    if device_map is None:
        return model
    return dispatch_model(
        model,
        device_map=device_map,
        offload_dir=offload_folder,
        offload_buffers=offload_buffers,
        skip_keys=skip_keys,
        preload_module_classes=preload_module_classes,
        force_hooks=force_hooks,
    )



def load_checkpoint_in_model_(
    model: nn.Module,
    sd,
    device_map: Optional[Dict[str, Union[int, str, torch.device]]] = None,
    offload_folder: Optional[Union[str, os.PathLike]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    offload_state_dict: bool = False,
    offload_buffers: bool = False,
    keep_in_fp32_modules: List[str] = None,
    offload_8bit_bnb: bool = False,
    strict: bool = False,
):
    """
    Loads a (potentially sharded) checkpoint inside a model, potentially sending weights to a given device as they are
    loaded.

    <Tip warning={true}>

    Once loaded across devices, you still need to call [`dispatch_model`] on your model to make it able to run. To
    group the checkpoint loading and dispatch in one single call, use [`load_checkpoint_and_dispatch`].

    </Tip>

    Args:
        model (`torch.nn.Module`):
            The model in which we want to load a checkpoint.
        sd,
        # checkpoint (`str` or `os.PathLike`):
        #     The folder checkpoint to load. It can be:
        #     - a path to a file containing a whole model state dict
        #     - a path to a `.json` file containing the index to a sharded checkpoint
        #     - a path to a folder containing a unique `.index.json` file and the shards of a checkpoint.
        #     - a path to a folder containing a unique pytorch_model.bin or a model.safetensors file.
        device_map (`Dict[str, Union[int, str, torch.device]]`, *optional*):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.
        offload_folder (`str` or `os.PathLike`, *optional*):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        dtype (`str` or `torch.dtype`, *optional*):
            If provided, the weights will be converted to that type when loaded.
        offload_state_dict (`bool`, *optional*, defaults to `False`):
            If `True`, will temporarily offload the CPU state dict on the hard drive to avoid getting out of CPU RAM if
            the weight of the CPU state dict + the biggest shard does not fit.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to include the buffers in the weights offloaded to disk.
        keep_in_fp32_modules(`List[str]`, *optional*):
            A list of the modules that we keep in `torch.float32` dtype.
        offload_8bit_bnb (`bool`, *optional*):
            Whether or not to enable offload of 8-bit modules on cpu/disk.
        strict (`bool`, *optional*, defaults to `False`):
            Whether to strictly enforce that the keys in the checkpoint state_dict match the keys of the model's
            state_dict.

    """
    if offload_8bit_bnb:
        from accelerate.utils.bnb import quantize_and_offload_8bit

    tied_params = find_tied_parameters(model)

    if check_tied_parameters_in_config(model) and len(tied_params) == 0:
        logger.warn(
            "The model weights are not tied. Please use the `tie_weights` method before using the `infer_auto_device` function."
        )
    if device_map is not None:
        check_tied_parameters_on_same_device(tied_params, device_map)

    if offload_folder is None and device_map is not None and "disk" in device_map.values():
        raise ValueError(
            "At least one of the model submodule will be offloaded to disk, please pass along an `offload_folder`."
        )
    elif offload_folder is not None and device_map is not None and "disk" in device_map.values():
        os.makedirs(offload_folder, exist_ok=True)

    if isinstance(dtype, str):
        # We accept "torch.float16" or just "float16"
        dtype = dtype.replace("torch.", "")
        dtype = getattr(torch, dtype)

    #checkpoint_files = None
    #index_filename = None
    # if os.path.isfile(checkpoint):
    #     if str(checkpoint).endswith(".json"):
    #         index_filename = checkpoint
    #     else:
    #         checkpoint_files = [checkpoint]
    # elif os.path.isdir(checkpoint):
    #     # check if the whole state dict is present
    #     potential_state_bin = [f for f in os.listdir(checkpoint) if f == WEIGHTS_NAME]
    #     potential_state_safetensor = [f for f in os.listdir(checkpoint) if f == SAFE_WEIGHTS_NAME]
    #     if len(potential_state_bin) == 1:
    #         checkpoint_files = [os.path.join(checkpoint, potential_state_bin[0])]
    #     elif len(potential_state_safetensor) == 1:
    #         checkpoint_files = [os.path.join(checkpoint, potential_state_safetensor[0])]
    #     else:
    #         # otherwise check for sharded checkpoints
    #         potential_index = [f for f in os.listdir(checkpoint) if f.endswith(".index.json")]
    #         if len(potential_index) == 0:
    #             raise ValueError(
    #                 f"{checkpoint} is not a folder containing a `.index.json` file or a {WEIGHTS_NAME} or a {SAFE_WEIGHTS_NAME} file"
    #             )
    #         elif len(potential_index) == 1:
    #             index_filename = os.path.join(checkpoint, potential_index[0])
    #         else:
    #             raise ValueError(
    #                 f"{checkpoint} containing more than one `.index.json` file, delete the irrelevant ones."
    #             )
    # else:
    #     raise ValueError(
    #         "`checkpoint` should be the path to a file containing a whole state dict, or the index of a sharded "
    #         f"checkpoint, or a folder containing a sharded checkpoint or the whole state dict, but got {checkpoint}."
    #     )

    # if index_filename is not None:
    #     checkpoint_folder = os.path.split(index_filename)[0]
    #     with open(index_filename) as f:
    #         index = json.loads(f.read())

    #     if "weight_map" in index:
    #         index = index["weight_map"]
    #     checkpoint_files = sorted(list(set(index.values())))
    #     checkpoint_files = [os.path.join(checkpoint_folder, f) for f in checkpoint_files]

    # Logic for missing/unexepected keys goes here.

    offload_index = {}
    if offload_state_dict:
        state_dict_folder = tempfile.mkdtemp()
        state_dict_index = {}

    unexpected_keys = set()
    model_keys = set(model.state_dict().keys())
    buffer_names = [name for name, _ in model.named_buffers()]

    for loaded_checkpoint in [sd]:
        #loaded_checkpoint = load_state_dict(checkpoint_file, device_map=device_map)
        if device_map is None:
            model.load_state_dict(loaded_checkpoint, strict=strict)
            unexpected_keys.update(set(loaded_checkpoint.keys()) - model_keys)
        else:
            for param_name, param in loaded_checkpoint.items():
                # skip SCB parameter (for 8-bit serialization)
                if "SCB" in param_name:
                    continue

                if param_name not in model_keys:
                    unexpected_keys.add(param_name)
                    if not strict:
                        continue  # Skip loading this parameter.

                module_name = param_name

                while len(module_name) > 0 and module_name not in device_map:
                    module_name = ".".join(module_name.split(".")[:-1])
                if module_name == "" and "" not in device_map:
                    # TODO: group all errors and raise at the end.
                    raise ValueError(f"{param_name} doesn't have any device set.")
                param_device = device_map[module_name]
                new_dtype = dtype
                if dtype is not None and torch.is_floating_point(param):
                    if keep_in_fp32_modules is not None and dtype == torch.float16:
                        proceed = False
                        for key in keep_in_fp32_modules:
                            if ((key in param_name) and (key + "." in param_name)) or key == param_name:
                                proceed = True
                                break
                        if proceed:
                            new_dtype = torch.float32

                if "weight" in param_name and param_name.replace("weight", "SCB") in loaded_checkpoint.keys():
                    if param.dtype == torch.int8:
                        fp16_statistics = loaded_checkpoint[param_name.replace("weight", "SCB")]
                else:
                    fp16_statistics = None

                if param_device == "disk":
                    if offload_buffers or param_name not in buffer_names:
                        if new_dtype is None:
                            new_dtype = param.dtype
                        if offload_8bit_bnb:
                            quantize_and_offload_8bit(
                                model, param, param_name, new_dtype, offload_folder, offload_index, fp16_statistics
                            )
                            continue
                        else:
                            set_module_tensor_to_device(model, param_name, "meta", dtype=new_dtype)
                        offload_weight(param, param_name, offload_folder, index=offload_index)
                elif param_device == "cpu" and offload_state_dict:
                    if new_dtype is None:
                        new_dtype = param.dtype
                    if offload_8bit_bnb:
                        quantize_and_offload_8bit(
                            model, param, param_name, new_dtype, state_dict_folder, state_dict_index, fp16_statistics
                        )
                    else:
                        set_module_tensor_to_device(model, param_name, "meta", dtype=new_dtype)
                        offload_weight(param, param_name, state_dict_folder, index=state_dict_index)
                else:
                    set_module_tensor_to_device(
                        model,
                        param_name,
                        param_device,
                        value=param,
                        dtype=new_dtype,
                        fp16_statistics=fp16_statistics,
                    )

        # Force Python to clean up.
        del loaded_checkpoint
        gc.collect()

    if not strict and len(unexpected_keys) > 0:
        logger.warning(
            f"Some weights of the model checkpoint were not used when"
            f" initializing {model.__class__.__name__}: {unexpected_keys}. This may or may not be an issue - make sure that the checkpoint does not have unnecessary parameters, or that the model definition correctly corresponds to the checkpoint."
        )

    save_offload_index(offload_index, offload_folder)

    # Load back offloaded state dict on CPU
    if offload_state_dict:
        load_offloaded_weights(model, state_dict_index, state_dict_folder)
        shutil.rmtree(state_dict_folder)

    retie_parameters(model, tied_params)

   