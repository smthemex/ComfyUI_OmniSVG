# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
import numpy as np
from omegaconf import OmegaConf
from pathlib import PureWindowsPath
import yaml
from .node_utils import gc_cleanup,tensor2pil_list,load_images,set_seed,tensor2pil_RGBA_list
from .OmniSVG.inference import load_models,process_text_to_svg,process_image_to_svg,format_save_files,format_save_pngfiles
import folder_paths
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
import nodes
import comfy.model_management as mm

########
MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")


class OmniSVG_Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OmniSVG_Loader",
            display_name="OmniSVG_Loader",
            category="OmniSVG",
            inputs=[
                io.String.Input("qwen_repo",default="Qwen/Qwen2.5-VL-3B-Instruct"),
                io.Combo.Input("qwen_dit",options= ["none"] + folder_paths.get_filename_list("clip") ),
                io.Combo.Input("transfromer",options= ["none"] + folder_paths.get_filename_list("diffusion_models") ),
                io.Boolean.Input("use_accelerate", default=False),
                io.Combo.Input("attn",options= ["none","sdpa","flash_attention_2"] ),
            ],
            outputs=[
                io.Custom("OmniSVG_Loader").Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, qwen_repo,qwen_dit,transfromer,use_accelerate,attn) -> io.NodeOutput:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if attn=="none":
            attn=None
        if qwen_repo:
            extra_repo=qwen_repo if qwen_repo.count('/') == 1 else PureWindowsPath(qwen_repo).as_posix()
        else:
            extra_repo=None

        dit_path=folder_paths.get_full_path("clip", qwen_dit) if qwen_dit!="none" else None

        model_size = "4B" if "4b" in transfromer.lower() else "8B"
        
        with open(os.path.join(current_node_path,'OmniSVG/config.yaml'), 'r') as f:
            config = yaml.safe_load(f)

        # load model
        print("***********Load model ***********")
        assert transfromer!="none","Please select a transformer model"
        weight_path=folder_paths.get_full_path("diffusion_models", transfromer)

        tokenizer, processor, sketch_decoder, svg_tokenizer = load_models(config,model_size,weight_path,dit_path,current_node_path,attn,use_accelerate,torch_dtype,extra_repo)

        print("***********Load model done ***********")

        gc_cleanup()
        model={"tokenizer":tokenizer,"processor":processor,"sketch_decoder":sketch_decoder,"svg_tokenizer":svg_tokenizer,"config":config,"torch_dtype":torch_dtype,"use_accelerate":use_accelerate}
        return io.NodeOutput(model)

class OmniSVG_Sampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OmniSVG_Sampler",
            display_name="OmniSVG_Sampler",
            category="OmniSVG",
            inputs=[
                io.Custom("OmniSVG_Loader").Input("model"),
                io.String.Input("prompt", multiline=True,default="A yellow t-shirt with a heart design represents love and positivity."),
                io.Int.Input("max_length", default=384, min=256, max=2048,step=64,display_mode=io.NumberDisplay.number),
                io.Float.Input("top_p", default=0.9, min=0, max=1,step=0.01,display_mode=io.NumberDisplay.number),
                io.Int.Input("top_k", default=50, min=1, max=1000,step=1,display_mode=io.NumberDisplay.number),
                io.Float.Input("temperature", default=0.5, min=0, max=1,step=0.01,display_mode=io.NumberDisplay.number),
                io.Float.Input("rep_penalty", default=1.05, min=0, max=10,step=0.01,display_mode=io.NumberDisplay.number),
                io.Int.Input("num_candidates", default=1, min=1, max=4,step=1,display_mode=io.NumberDisplay.number),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED,display_mode=io.NumberDisplay.number),
                io.Combo.Input("subtype",options= ["auto","illustration","icon"]  ),          
                io.Boolean.Input("save_all", default=False),
                io.Image.Input("image",optional=True),
                io.Mask.Input("mask",optional=True),
            ], 
            outputs=[
                io.Image.Output(display_name="image"),
                io.String.Output(display_name="svg_path"),
                ],
        )
    @classmethod
    def execute(cls,model,prompt,max_length,top_p,top_k,temperature,rep_penalty,num_candidates,seed,subtype,save_all,image=None,mask=None)-> io.NodeOutput:
        set_seed(seed)
        org_args={"verbose":True,}
        args=OmegaConf.create(org_args)
        args.output=folder_paths.get_output_directory()
        args.save_all_candidates=save_all
        args.save_png=True
        args.top_p=top_p
        args.top_k=top_k
        args.temperature=temperature
        args.repetition_penalty=rep_penalty
        args.max_length=max_length
        args.replace_background=False
        args.num_candidates=num_candidates
        args.use_accelerate=model["use_accelerate"]
        if isinstance(image,torch.Tensor):
            print("*********** Image to Svg ***********")
            if isinstance(mask,torch.Tensor):             
                images_list=tensor2pil_RGBA_list(image, mask, 448, 448)
            else:    
                images_list=tensor2pil_list(image, 448, 448)  
           
            for i,img in enumerate(images_list):   
                img.save(os.path.join(args.output,f"{i}.png"))   
            saved_files,saved_pngs=process_image_to_svg(model.get("sketch_decoder"),model.get("svg_tokenizer"),model.get("processor"),images_list,device,model.get("torch_dtype"),args,model.get("config") )   
        else:
            print("*********** Text to Svg ***********")
            saved_files,saved_pngs=process_text_to_svg(model.get("sketch_decoder"),model.get("svg_tokenizer"),model.get("processor"),prompt,device,model.get("torch_dtype"),args,subtype,model.get("config"))
        images=format_save_pngfiles(saved_pngs)
        gc.collect()
        torch.cuda.empty_cache()
        return io.NodeOutput(load_images(images),format_save_files(saved_files))
       

from aiohttp import web
from server import PromptServer
@PromptServer.instance.routes.get("/OmniSVG_SM_Extension")
async def get_hello(request):
    return web.json_response("OmniSVG_SM_Extension")

class OmniSVG_SM_Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            OmniSVG_Loader,
            OmniSVG_Sampler,
        ]
async def comfy_entrypoint() -> OmniSVG_SM_Extension:  # ComfyUI calls this to load your extension and its nodes.
    return OmniSVG_SM_Extension()
