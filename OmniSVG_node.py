# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
import numpy as np
from omegaconf import OmegaConf
from pathlib import PureWindowsPath
import yaml
from .node_utils import gc_cleanup,tensor2pil_list,load_images,pil2narry,tensor2pil_RGBA_list
from .OmniSVG.inference import load_models,process_text_file,process_image_folder
import folder_paths


########
MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

# add checkpoints dir
OmniSVG_weigths_path = os.path.join(folder_paths.models_dir, "OmniSVG")
if not os.path.exists(OmniSVG_weigths_path):
    os.makedirs(OmniSVG_weigths_path)
folder_paths.add_model_folder_path("OmniSVG", OmniSVG_weigths_path)

######



class OmniSVGLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen_repo":("STRING", {"multiline": False,"default":"Qwen/Qwen2.5-VL-3B-Instruct"}),
                "transfromer":(["none"] + folder_paths.get_filename_list("diffusion_models"),),
                "use_low_vram":("BOOLEAN", {"default": False},),
            },
        }

    RETURN_TYPES = ("MODEL_OmniSVG",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loader_main"
    CATEGORY = "OmniSVG"

    def loader_main(self,qwen_repo,transfromer,use_low_vram,):


        if qwen_repo:
            if qwen_repo.count('/') == 1:
                extra_repo=qwen_repo
            else:
                extra_repo=PureWindowsPath(qwen_repo).as_posix()
        else:
            extra_repo="Qwen/Qwen2.5-VL-3B-Instruct"

        # load model
        print("***********Load model ***********")

        if transfromer == "none" :

            raise Exception("Please select a transformer model")

        else:
            weight_path=folder_paths.get_full_path("diffusion_models", transfromer)

        tokenizer, processor, sketch_decoder, svg_tokenizer = load_models(weight_path,current_node_path,extra_repo)

        with open(os.path.join(current_node_path,'OmniSVG/config.yaml'), 'r') as f:
            config = yaml.safe_load(f)


        print("***********Load model done ***********")

        gc_cleanup()

        return ({"tokenizer":tokenizer,"processor":processor,"sketch_decoder":sketch_decoder,"svg_tokenizer":svg_tokenizer,"config":config},)



class OmniSVGSampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_OmniSVG",),
                "prompt":("STRING", {"multiline": True,"default":"A yellow t-shirt with a heart design represents love and positivity."}),
                "height": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 16, "display": "number"}),
                "width": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 16, "display": "number"}),
                },
            "optional": {
                "iamge": ("IMAGE",),
                "mask": ("MASK",),

            }}

    RETURN_TYPES = ("STRING","IMAGE", )
    RETURN_NAMES = ("string","image",)
    FUNCTION = "sampler_main"
    CATEGORY = "OmniSVG"

    def sampler_main(self,model,prompt,height,width,**kwargs):
        infer_images=kwargs.get("iamge",None)
        mask_info=kwargs.get("mask",None)

        if isinstance(infer_images,torch.Tensor):
            if isinstance(mask_info,torch.Tensor):
                images_list=tensor2pil_RGBA_list(infer_images, mask_info, width, height)
            else:
                images_list=tensor2pil_list(infer_images, width, height)
            svgpath,images=process_image_folder(model.get("sketch_decoder"),model.get("svg_tokenizer"),model.get("processor"),model.get("config"),images_list, folder_paths.get_output_directory())
            gc.collect()
            torch.cuda.empty_cache()
            return (svgpath,load_images(images), )

        else:
            svgpath,images=process_text_file(model.get("sketch_decoder"),model.get("svg_tokenizer"),model.get("processor"),model.get("config"),prompt, folder_paths.get_output_directory())
            gc.collect()
            torch.cuda.empty_cache()
            return (svgpath,pil2narry(images), )
       



NODE_CLASS_MAPPINGS = {

    "OmniSVGLoader": OmniSVGLoader,
    "OmniSVGSampler": OmniSVGSampler,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniSVGLoader": "OmniSVGLoader",
    "OmniSVGSampler": "OmniSVGSampler",

}
