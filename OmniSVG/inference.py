import torch
import os
from PIL import Image
import cairosvg
import io
import gc
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from datetime import datetime
import yaml
from .decoder import SketchDecoder
from .tokenizer import SVGTokenizer
import numpy as np
import time
import glob
import tempfile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Constants from config
SYSTEM_PROMPT = """You are an expert SVG code generator. 
Generate precise, valid SVG path commands that accurately represent the described scene or object.
Focus on capturing key shapes, spatial relationships, and visual composition."""

SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']

# CONFIG_PATH = './config.yaml'
# with open(CONFIG_PATH, 'r') as f:
#     config = yaml.safe_load(f)

# AVAILABLE_MODEL_SIZES = list(config.get('models', {}).keys())
# DEFAULT_MODEL_SIZE = config.get('default_model_size', '8B')

# Image processing settings from config
#image_config = config.get('image', {})
#TARGET_IMAGE_SIZE = image_config.get('target_size', 448)
#RENDER_SIZE = image_config.get('render_size', 512)
#BACKGROUND_THRESHOLD = image_config.get('background_threshold', 240)
#EMPTY_THRESHOLD_ILLUSTRATION = image_config.get('empty_threshold_illustration', 250)
# EMPTY_THRESHOLD_ICON = image_config.get('empty_threshold_icon', 252)
# EDGE_SAMPLE_RATIO = image_config.get('edge_sample_ratio', 0.1)
#COLOR_SIMILARITY_THRESHOLD = image_config.get('color_similarity_threshold', 30)
# MIN_EDGE_SAMPLES = image_config.get('min_edge_samples', 10)

# # Color settings from config
# colors_config = config.get('colors', {})
# BLACK_COLOR_TOKEN = colors_config.get('black_color_token', 
#                                        colors_config.get('color_token_start', 40010) + 2)
# Model settings from config
# model_config = config.get('model', {})
# BOS_TOKEN_ID = model_config.get('bos_token_id', 196998)
# EOS_TOKEN_ID = model_config.get('eos_token_id', 196999)
# PAD_TOKEN_ID = model_config.get('pad_token_id', 151643)
# MAX_LENGTH = model_config.get('max_length', 1024)
# MIN_MAX_LENGTH = 256
# MAX_MAX_LENGTH = 2048

# Task configurations with defaults from config


# Generation parameters from config
# gen_config = config.get('generation', {})
# DEFAULT_NUM_CANDIDATES = gen_config.get('default_num_candidates', 4)
# MAX_NUM_CANDIDATES = gen_config.get('max_num_candidates', 8)
# EXTRA_CANDIDATES_BUFFER = gen_config.get('extra_candidates_buffer', 4)

# Validation settings from config
# validation_config = config.get('validation', {})
# MIN_SVG_LENGTH = validation_config.get('min_svg_length', 20)

# def get_model_input_device():
#     """
#     Get the device where model inputs should be placed.
#     This handles multi-GPU scenarios where the model is distributed across devices.
#     """
#     global sketch_decoder
    
#     if sketch_decoder is None:
#         return default_device
    
#     try:
#         # Get the transformer model
#         model = sketch_decoder.transformer
        
#         # Try to get device from the embedding layer (this is where input_ids will be processed)
#         if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
#             embed_device = next(model.model.embed_tokens.parameters()).device
#             return embed_device
#         elif hasattr(model, 'embed_tokens'):
#             embed_device = next(model.embed_tokens.parameters()).device
#             return embed_device
        
#         # Alternative: try to get from the first parameter
#         first_param = next(model.parameters())
#         return first_param.device
        
#     except (StopIteration, AttributeError) as e:
#         print(f"Warning: Could not determine model device, using default: {default_device}")
#         return default_device


def get_config_value(config,model_size, *keys):
    """Get config value with model-specific override support."""
    model_cfg = config.get('models', {}).get(model_size, {})
    value = model_cfg
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            value = None
            break
    
    if value is None:
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
    
    return value


def load_models(config,model_size,model_path,dit_path,node_path,attn,use_accelerate,torch_dtype,qwen_repo):

    print("Loading models...")
    local_repo=os.path.join(node_path,"OmniSVG/Qwen2.5-VL-7B-Instruct") if model_size=="8B" else os.path.join(node_path,"OmniSVG/Qwen2.5-VL-3B-Instruct") 
    # Load Qwen tokenizer and processor
    print("\n[1/3] Loading tokenizer and processor...")
    tokenizer = AutoTokenizer.from_pretrained(
        local_repo, 
        padding_side="left",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        local_repo, 
        padding_side="left",
        trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"
    print("Tokenizer and processor loaded successfully!")

    # Initialize sketch decoder with model_size
    print("\n[2/3] Initializing SketchDecoder...")
    sketch_decoder = SketchDecoder(
        config_path=os.path.join(node_path,'OmniSVG/config.yaml'),
        model_path=qwen_repo,
        clip_path=dit_path,
        repo=local_repo,
        attn=attn,
        model_size=model_size,
        use_accelerate=use_accelerate,
        pix_len=2048,
        text_len=config.get('text', {}).get('max_length', 200),
        torch_dtype=torch_dtype
    )
    # Load OmniSVG weights
    print(f"\n[3/3] Loading OmniSVG is done...")
   
    has_language_keys = any("language_model" in k for k in sketch_decoder.state_dict().keys()) # new transformer vesrion load language model dict

    sketch_dict=torch.load(model_path,map_location='cpu')
    
    if has_language_keys:
        print("New transformer version,try to load language model dict")
        new_dict = {}
        for k, v in sketch_dict.items():
            if k.startswith("transformer.visual."):
                new_k = k.replace("transformer.visual.", "transformer.model.visual.")
            elif k.startswith("transformer.model.layers.") or k.startswith("transformer.model.embed_tokens.") or k.startswith("transformer.model.norm.") :
                new_k = k.replace("transformer.model.", "transformer.model.language_model.")
            else:
                new_k = k
            new_dict[new_k] = v
        mis,uns=sketch_decoder.load_state_dict(new_dict,assign=True)
        print(f"Mis: {mis}")
        print(f"Uns: {uns}")
        del sketch_dict,new_dict
    else:
        sketch_decoder.load_state_dict(sketch_dict)
    sketch_decoder.eval()
    gc.collect()
    
    # Initialize SVG tokenizer
    svg_tokenizer = SVGTokenizer(os.path.join(node_path,'OmniSVG/config.yaml'),model_size=model_size)
    print("All models loaded successfully!")
    return tokenizer, processor, sketch_decoder, svg_tokenizer

def detect_text_subtype(text_prompt):
    """Auto-detect text prompt subtype"""
    text_lower = text_prompt.lower()
    
    icon_keywords = ['icon', 'logo', 'symbol', 'badge', 'button', 'emoji', 'glyph', 'simple', 
                     'arrow', 'triangle', 'circle', 'square', 'heart', 'star', 'checkmark']
    if any(kw in text_lower for kw in icon_keywords):
        return "icon"
    
    illustration_keywords = [
        'illustration', 'scene', 'person', 'people', 'character', 'man', 'woman', 'boy', 'girl',
        'avatar', 'portrait', 'face', 'head', 'body',
        'cat', 'dog', 'bird', 'animal', 'pet', 'fox', 'rabbit',
        'sitting', 'standing', 'walking', 'running', 'sleeping', 'holding', 'playing',
        'house', 'building', 'tree', 'garden', 'landscape', 'mountain', 'forest', 'city',
        'ocean', 'beach', 'sunset', 'sunrise', 'sky'
    ]
    
    match_count = sum(1 for kw in illustration_keywords if kw in text_lower)
    if match_count >= 1 or len(text_prompt) > 50:
        return "illustration"
    
    return "icon"

def detect_and_replace_background(image, threshold=None, edge_sample_ratio=None):
    """Detect if image has non-white background and optionally replace it."""
    if threshold is None:
        threshold = 240 #BACKGROUND_THRESHOLD
    if edge_sample_ratio is None:
        edge_sample_ratio = 0.1 #EDGE_SAMPLE_RATIO
    
    img_array = np.array(image)
    
    if image.mode == 'RGBA':
        bg = Image.new('RGBA', image.size, (255, 255, 255, 255))
        composite = Image.alpha_composite(bg, image)
        return composite.convert('RGB'), True
    
    h, w = img_array.shape[:2]
    edge_pixels = []
    
    sample_count = max(10, int(min(h, w) * edge_sample_ratio)) #MIN_EDGE_SAMPLES
    
    for i in range(0, w, max(1, w // sample_count)):
        edge_pixels.append(img_array[0, i])
        edge_pixels.append(img_array[h-1, i])
    
    for i in range(0, h, max(1, h // sample_count)):
        edge_pixels.append(img_array[i, 0])
        edge_pixels.append(img_array[i, w-1])
    
    edge_pixels = np.array(edge_pixels)
    
    if len(edge_pixels) > 0:
        mean_edge = edge_pixels.mean(axis=0)
        if np.all(mean_edge > threshold):
            return image, False
    
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        if img_array.shape[2] == 4:
            gray = np.mean(img_array[:, :, :3], axis=2)
        else:
            gray = np.mean(img_array, axis=2)
        
        edge_colors = []
        for i in range(w):
            edge_colors.append(tuple(img_array[0, i, :3]))
            edge_colors.append(tuple(img_array[h-1, i, :3]))
        for i in range(h):
            edge_colors.append(tuple(img_array[i, 0, :3]))
            edge_colors.append(tuple(img_array[i, w-1, :3]))
        
        from collections import Counter
        color_counts = Counter(edge_colors)
        bg_color = color_counts.most_common(1)[0][0]
        
        color_diff = np.sqrt(np.sum((img_array[:, :, :3].astype(float) - np.array(bg_color)) ** 2, axis=2))
        bg_mask = color_diff < 30 # COLOR_SIMILARITY_THRESHOLD
        
        result = img_array.copy()
        if result.shape[2] == 4:
            result[bg_mask] = [255, 255, 255, 255]
        else:
            result[bg_mask] = [255, 255, 255]
        
        return Image.fromarray(result).convert('RGB'), True
    
    return image, False

# def get_model_devices_info():
#     """Get information about which devices the model is using (for debugging)."""
#     global sketch_decoder
    
#     if sketch_decoder is None:
#         return "Model not loaded"
    
#     devices = set()
#     try:
#         model = sketch_decoder.transformer
#         for name, param in model.named_parameters():
#             devices.add(str(param.device))
#     except Exception as e:
#         return f"Error getting device info: {e}"
    
#     return f"Model distributed across: {sorted(devices)}"

def prepare_inputs(task_type,processor, content):
    """Prepare model inputs"""
    if task_type == "text-to-svg":
        prompt_text = str(content).strip()
        
        instruction = f"""Generate an SVG illustration for: {prompt_text}
        Requirements:
        - Create complete SVG path commands
        - Include proper coordinates and colors
        - Maintain visual clarity and composition"""
                
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": instruction}]}
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        #print("Text input:", text_input)
        inputs = processor(text=[text_input], padding=True, truncation=True, return_tensors="pt")
        
    else:  # image-to-svg
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "Generate SVG code that accurately represents this image:"},
                {"type": "image", "image": content},
            ]}
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text_input], images=image_inputs, padding=True, truncation=True, return_tensors="pt")

    return inputs

def preprocess_image_for_svg(image, replace_background=True, target_size=None):
    """Preprocess image for SVG generation."""
    if target_size is None:
        target_size =448 # TARGET_IMAGE_SIZE
    
    if isinstance(image, str):
        raw_img = Image.open(image)
    else:
        raw_img = image
    
    was_modified = False
    
    if raw_img.mode == 'RGBA':
        bg = Image.new('RGBA', raw_img.size, (255, 255, 255, 255))
        img_with_bg = Image.alpha_composite(bg, raw_img).convert('RGB')
        was_modified = True
    elif raw_img.mode == 'LA' or raw_img.mode == 'PA':
        raw_img = raw_img.convert('RGBA')
        bg = Image.new('RGBA', raw_img.size, (255, 255, 255, 255))
        img_with_bg = Image.alpha_composite(bg, raw_img).convert('RGB')
        was_modified = True
    elif raw_img.mode != 'RGB':
        img_with_bg = raw_img.convert('RGB')
    else:
        img_with_bg = raw_img
    
    if replace_background:
        img_with_bg, bg_replaced = detect_and_replace_background(img_with_bg)
        was_modified = was_modified or bg_replaced
    
    img_resized = img_with_bg.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    return img_resized, was_modified


def render_svg_to_image(svg_str, size=None):
    """Render SVG to high-quality PIL Image"""
    if size is None:
        size = 1024 #RENDER_SIZE
    
    try:
        png_data = cairosvg.svg2png(
            bytestring=svg_str.encode('utf-8'),
            output_width=size,
            output_height=size
        )
        image_rgba = Image.open(io.BytesIO(png_data)).convert("RGBA")
        bg = Image.new("RGB", image_rgba.size, (255, 255, 255))
        bg.paste(image_rgba, mask=image_rgba.split()[3])
        return bg
    except Exception as e:
        print(f"Render error: {e}")
        return None


def is_valid_candidate(svg_str, img, subtype="illustration"):
    """Check candidate validity"""
    if not svg_str or len(svg_str) < 20 : # MIN_SVG_LENGTH:
        return False, "too_short"
    
    if '<svg' not in svg_str:
        return False, "no_svg_tag"
    
    if img is None:
        return False, "render_failed"
    
    img_array = np.array(img)
    mean_val = img_array.mean()
    
    threshold = 250 if subtype == "illustration" else 252 #  EMPTY_THRESHOLD_ILLUSTRATION #EMPTY_THRESHOLD_ICON
    
    if mean_val > threshold:
        return False, "empty_image"
    
    return True, "ok"


def generate_candidates(sketch_decoder,svg_tokenizer,inputs, task_type, subtype, temperature, top_p, top_k, repetition_penalty, 
                       max_length, num_samples,config,torch_dtype=torch.bfloat16):
    """Generate candidate SVGs with full parameter control"""
    
    # Model settings from config
    model_config = config.get('model', {})
    BOS_TOKEN_ID = model_config.get('bos_token_id', 196998)
    EOS_TOKEN_ID = model_config.get('eos_token_id', 196999)
    PAD_TOKEN_ID = model_config.get('pad_token_id', 151643)
    MAX_LENGTH = model_config.get('max_length', 1024)
    MIN_MAX_LENGTH = 256
    MAX_MAX_LENGTH = 2048

    gen_config = config.get('generation', {})
    DEFAULT_NUM_CANDIDATES = gen_config.get('default_num_candidates', 4)
    MAX_NUM_CANDIDATES = gen_config.get('max_num_candidates', 8)
    EXTRA_CANDIDATES_BUFFER = gen_config.get('extra_candidates_buffer', 4)

    colors_config = config.get('colors', {})
    BLACK_COLOR_TOKEN = colors_config.get('black_color_token', 
                                       colors_config.get('color_token_start', 40010) + 2)

    # Get the correct device from the model's embedding layer
    #input_device = get_model_input_device()
    input_device = next(sketch_decoder.transformer.parameters()).device

    input_ids = inputs['input_ids'].to(input_device)
    attention_mask = inputs['attention_mask'].to(input_device)
    
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    
    if 'pixel_values' in inputs:
        model_inputs["pixel_values"] = inputs['pixel_values'].to(input_device, dtype=torch_dtype)
    
    if 'image_grid_thw' in inputs:
        model_inputs["image_grid_thw"] = inputs['image_grid_thw'].to(input_device)
    
    all_candidates = []
    
    gen_cfg = {
        'do_sample': True,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': int(top_k),
        'repetition_penalty': repetition_penalty,
        'early_stopping': True,
        'no_repeat_ngram_size': 0,
        'eos_token_id': EOS_TOKEN_ID,
        'pad_token_id': PAD_TOKEN_ID,
        'bos_token_id': BOS_TOKEN_ID,
    }
    
    actual_samples = num_samples + EXTRA_CANDIDATES_BUFFER
    
    try:
        with torch.no_grad():
            results = sketch_decoder.transformer.generate(
                **model_inputs,
                max_new_tokens=max_length,
                num_return_sequences=actual_samples,
                use_cache=True,
                **gen_cfg
            )

            input_len = input_ids.shape[1]
            generated_ids_batch = results[:, input_len:]

        for i in range(min(actual_samples, generated_ids_batch.shape[0])):
            try:
                current_ids = generated_ids_batch[i:i+1]
                
                # Move to CPU for post-processing to avoid device issues
                current_ids_cpu = current_ids.cpu()
                
                fake_wrapper = torch.cat([
                    torch.full((1, 1), BOS_TOKEN_ID, device='cpu'),
                    current_ids_cpu,
                    torch.full((1, 1), EOS_TOKEN_ID, device='cpu')
                ], dim=1)

                generated_xy = svg_tokenizer.process_generated_tokens(fake_wrapper)
                if len(generated_xy) == 0:
                    continue

                svg_tensors, color_tensors = svg_tokenizer.raster_svg(generated_xy)
                if not svg_tensors or not svg_tensors[0]:
                    continue

                num_paths = len(svg_tensors[0])
                while len(color_tensors) < num_paths:
                    color_tensors.append(BLACK_COLOR_TOKEN)
                
                svg = svg_tokenizer.apply_colors_to_svg(svg_tensors[0], color_tensors)
                svg_str = svg.to_str()
                
                if 'width=' not in svg_str:
                    svg_str = svg_str.replace('<svg', f'<svg width="{448}" height="{448}"', 1) #TARGET_IMAGE_SIZE
                
                png_image = render_svg_to_image(svg_str, size=1024) #RENDER_SIZE
                #print(f"  Rendered image {png_image},{svg_str},{num_paths},{all_candidates}")
                is_valid, reason = is_valid_candidate(svg_str, png_image, subtype)
                if is_valid:
                    all_candidates.append({
                        'svg': svg_str,
                        'img': png_image,
                        'path_count': num_paths,
                        'index': len(all_candidates) + 1
                    })
                    
                    print(f"  Found valid candidate {len(all_candidates)} with {num_paths} paths")
                    if len(all_candidates) >= num_samples:
                        break
                
                    print(f"  Candidate {i} invalid: {reason}")
                        
            except Exception as e:
                print(f"  Candidate {i} error: {e}")
                continue

    except Exception as e:
        print(f"Generation Error: {e}")
        import traceback
        traceback.print_exc()
    
    return all_candidates

def save_results(candidates, output_dir, base_name, save_png=False, save_all=False):
    """Save generated SVG(s) and optionally PNG(s)"""
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    saved_pngs = []
    
    if not candidates:
        return saved_files,saved_pngs
    
    if save_all:
        for i, cand in enumerate(candidates):
            svg_path = os.path.join(output_dir, f"{base_name}_candidate_{i+1}.svg")
            with open(svg_path, 'w', encoding='utf-8') as f:
                f.write(cand['svg'])
            saved_files.append(svg_path)
            
            if save_png and cand['img'] is not None:
                # png_path = os.path.join(output_dir, f"{base_name}_candidate_{i+1}.png")
                # cand['img'].save(png_path)
                saved_pngs.append(cand['img'])
    else:
        # Save only the best (first valid) candidate
        best = candidates[0]
        svg_path = os.path.join(output_dir, f"{base_name}.svg")
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(best['svg'])
        saved_files.append(svg_path)
        
        if save_png and best['img'] is not None:
            # png_path = os.path.join(output_dir, f"{base_name}.png")
            # best['img'].save(png_path)
            saved_pngs.append(best['img'])
    
    return saved_files,saved_pngs

def format_save_pngfiles(save_pngs):
    """
    将save_pngs列表转换为PIL图像列表
    
    Args:
        save_pngs: 列表，可能为空[]、包含None元素或包含图像路径字符串
        
    Returns:
        list: PIL图像对象列表，如果全部为空则返回包含一张512*512白色图片的列表
    """
    # 处理空列表的情况
    if not save_pngs:
        return [Image.new('RGB', (512, 512), 'white')]
    
    # 收集有效的图像
    pil_images = []
    
    for item in save_pngs:
        # 跳过None元素
        if item is None:
            continue
            
        # 如果是字符串路径且文件存在，则加载图像
        if isinstance(item, Image.Image):
            pil_images.append(item)
         
        if isinstance(item, list):
            for i in item:
                if isinstance(i, Image.Image):
                    pil_images.append(i)
    # 如果没有有效图像，返回默认的白色图像
    if not pil_images:
        return [Image.new('RGB', (512, 512), 'white')]
    
    return pil_images


def format_save_files(save_files):
    """
    将save_files列表转换为多行字符串
    
    Args:
        save_files: 列表，可能为空[]、包含路径字符串或包含None元素
        
    Returns:
        str: 多行字符串，每行一个路径，空列表或[None]返回"none"
    """
    # 处理空列表的情况
    if not save_files:
        return "none"
    
    # 处理只包含None元素的列表
    if len(save_files) == 1 and save_files[0] is None:
        return "none"
    
    # 过滤掉None元素，保留有效路径
    valid_paths = [path for path in save_files if path is not None]

    
    # 如果过滤后没有有效路径，返回"none"
    if not valid_paths:
        return "none"
    
    valid_paths_=[]
    for path in valid_paths:
        if isinstance(path, str):
            valid_paths_.append(path)
        if isinstance(path, list):
            for i in path:
                if isinstance(i, str):
                    valid_paths_.append(i)

    if not valid_paths_:
        return "none"   
    
    # 将有效路径连接成多行字符串
    return "\n".join(valid_paths_)

def process_text_to_svg(sketch_decoder,svg_tokenizer,processor,prompts,input_device,torch_dtype,args,subtype,config):
    if not args.use_accelerate:
        sketch_decoder.to(input_device)
    """Process text-to-svg task"""
    prompts=[prompts]
    task_config = config.get('task_configs', {})
    TASK_CONFIGS = {
        "text-to-svg-icon": task_config.get('text_to_svg_icon', {
            "default_temperature": 0.5,
            "default_top_p": 0.88,
            "default_top_k": 50,
            "default_repetition_penalty": 1.05,
        }),
        "text-to-svg-illustration": task_config.get('text_to_svg_illustration', {
            "default_temperature": 0.6,
            "default_top_p": 0.90,
            "default_top_k": 60,
            "default_repetition_penalty": 1.03,
        }),
        "image-to-svg": task_config.get('image_to_svg', {
            "default_temperature": 0.3,
            "default_top_p": 0.90,
            "default_top_k": 50,
            "default_repetition_penalty": 1.05,
        })
    }
    
    # Process each prompt
    total_success = 0
    total_failed = 0
    saved_files = []
    saved_pngs = []
    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx+1}/{len(prompts)}] Processing: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        
        start_time = time.time()
        
        # Detect subtype
        if subtype=="auto":
            subtype = detect_text_subtype(prompt)
        task_key = f"text-to-svg-{subtype}"
        
        # Get default parameters based on task
        temperature = args.temperature if args.temperature is not None else TASK_CONFIGS[task_key].get("default_temperature", 0.5)
        top_p = args.top_p if args.top_p is not None else TASK_CONFIGS[task_key].get("default_top_p", 0.90)
        top_k = args.top_k if args.top_k is not None else TASK_CONFIGS[task_key].get("default_top_k", 50)
        rep_penalty = args.repetition_penalty if args.repetition_penalty is not None else TASK_CONFIGS[task_key].get("default_repetition_penalty", 1.05)
        
        if args.verbose:
            print(f"  Subtype: {subtype}")
            print(f"  Params: temp={temperature}, top_p={top_p}, top_k={top_k}, rep={rep_penalty}")
        
        # Prepare inputs
        inputs = prepare_inputs("text-to-svg", processor,prompt)
        
        # Generate candidates
        candidates = generate_candidates(sketch_decoder,svg_tokenizer,
            inputs, "text-to-svg", subtype,
            temperature, top_p, top_k, rep_penalty,
            args.max_length, args.num_candidates,config,
           torch_dtype=torch_dtype,
        )
        
        elapsed = time.time() - start_time
        
        if candidates:
            # Create safe filename from prompt
            safe_name = "".join(c if c.isalnum() or c in ' -_' else '_' for c in prompt[:50]).strip()
            safe_name = f"{idx+1:04d}_{safe_name}"
            
            saved,saved_png = save_results(candidates, args.output, safe_name, 
                               save_png=args.save_png, save_all=args.save_all_candidates)
            saved_files.append(saved)
            saved_pngs.append(saved_png)
            print(f"  ✓ Generated {len(candidates)} candidates in {elapsed:.2f}s")
            print(f"  Saved: {', '.join(os.path.basename(f) for f in saved)}")
            total_success += 1
        else:
            print(f"  ✗ Failed to generate valid SVG ({elapsed:.2f}s)")
            total_failed += 1
            saved_files.append(None)
            saved_pngs.append(None)
        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print(f"Text-to-SVG Complete!")
    print(f"  Success: {total_success}/{len(prompts)}")
    print(f"  Failed: {total_failed}/{len(prompts)}")
    print(f"  Output: {args.output}")
    print("="*60)
    if not args.use_accelerate:
        sketch_decoder.to('cpu')
        torch.cuda.empty_cache()
    return saved_files,saved_pngs


def process_image_to_svg(sketch_decoder,svg_tokenizer,processor,image_files,input_device,torch_dtype,args,config):
    if not args.use_accelerate:
        sketch_decoder.to(input_device)
    """Process image-to-svg task"""
    # input_path = args.input
    task_config = config.get('task_configs', {})
    TASK_CONFIGS = {
        "text-to-svg-icon": task_config.get('text_to_svg_icon', {
            "default_temperature": 0.5,
            "default_top_p": 0.88,
            "default_top_k": 50,
            "default_repetition_penalty": 1.05,
        }),
        "text-to-svg-illustration": task_config.get('text_to_svg_illustration', {
            "default_temperature": 0.6,
            "default_top_p": 0.90,
            "default_top_k": 60,
            "default_repetition_penalty": 1.03,
        }),
        "image-to-svg": task_config.get('image_to_svg', {
            "default_temperature": 0.3,
            "default_top_p": 0.90,
            "default_top_k": 50,
            "default_repetition_penalty": 1.05,
        })
    }
    
    # Get default parameters
    task_key = "image-to-svg"
    temperature = args.temperature if args.temperature is not None else TASK_CONFIGS[task_key].get("default_temperature", 0.3)
    top_p = args.top_p if args.top_p is not None else TASK_CONFIGS[task_key].get("default_top_p", 0.90)
    top_k = args.top_k if args.top_k is not None else TASK_CONFIGS[task_key].get("default_top_k", 50)
    rep_penalty = args.repetition_penalty if args.repetition_penalty is not None else TASK_CONFIGS[task_key].get("default_repetition_penalty", 1.05)
    
    if args.verbose:
        print(f"Params: temp={temperature}, top_p={top_p}, top_k={top_k}, rep={rep_penalty}")
    
    # Process each image
    total_success = 0
    total_failed = 0
    base_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_files = []
    saved_pngs = []
    for idx, image in enumerate(image_files):
        #img_name = os.path.basename(img_path)
        #print(f"\n[{idx+1}/{len(image_files)}] Processing: {img_name}")
        
        start_time = time.time()
        
        try:
            # Load and preprocess image
            #image = Image.open(img_path)
            img_processed, was_modified = preprocess_image_for_svg(
                image, 
                replace_background=args.replace_background,
                target_size=448 ,#TARGET_IMAGE_SIZE
            )
            
            # if args.verbose and was_modified:
            #     print("  Background processed/replaced")
            
            # Save to temp file for processing
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                img_processed.save(tmp_file.name, format='PNG', quality=100)
                tmp_path = tmp_file.name
            
            try:
                # Prepare inputs
                inputs = prepare_inputs("image-to-svg",processor, tmp_path)
                
                # Generate candidates
                candidates = generate_candidates(sketch_decoder,svg_tokenizer,
                    inputs, "image-to-svg", "image",
                    temperature, top_p, top_k, rep_penalty,
                    args.max_length, args.num_candidates,config,
                   torch_dtype=torch_dtype,
                )
                
                elapsed = time.time() - start_time
                
                if candidates:
                    # Use original filename (without extension) as base name
                    #base_name = os.path.splitext(img_name)[0]
                    
                    saved,saved_png = save_results(candidates, args.output, base_name, 
                                       save_png=args.save_png, save_all=args.save_all_candidates)
                    saved_files.append(saved)
                    saved_pngs.append(saved_png)
                    print(f"  ✓ Generated {len(candidates)} candidates in {elapsed:.2f}s")
                    print(f"  Saved: {', '.join(os.path.basename(f) for f in saved)}")
                    total_success += 1
                else:
                    print(f"  ✗ Failed to generate valid SVG ({elapsed:.2f}s)")
                    saved_files.append(None)
                    saved_pngs.append(None)
                    total_failed += 1
                    
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            print(f"  ✗ Error: {e}")
            total_failed += 1
        
        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print(f"Image-to-SVG Complete!")
    print(f"  Success: {total_success}/{len(image_files)}")
    print(f"  Failed: {total_failed}/{len(image_files)}")
    print(f"  Output: {args.output}")
    print("="*60)
    if not args.use_accelerate:
        sketch_decoder.to('cpu')
        torch.cuda.empty_cache()
    return saved_files,saved_pngs

# def process_and_resize_image(image_input, target_size=(200, 200)):
#     """Process and resize image to target size"""
#     if isinstance(image_input, str):
#         image = Image.open(image_input)
#     elif isinstance(image_input, Image.Image):
#         image = image_input
#     else:
#         image = Image.fromarray(image_input)

    
#     #image = image.resize(target_size, Image.Resampling.LANCZOS)
    
#     return image

# def process_text_to_svg(processor,text_description):
#     """Process text-to-svg task"""
    
#     messages = [{
#         "role": "system",
#         "content": SYSTEM_PROMPT
#     }, {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": f"Task: text-to-svg\nDescription: {text_description}\nGenerate SVG code based on the above description."}
#         ]
#     }]
    
#     text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     inputs = processor(
#         text=[text_input], 
#         truncation=True,
#         return_tensors="pt"
#     )
    
#     input_ids = inputs['input_ids'].to(device)
#     attention_mask = inputs['attention_mask'].to(device)
#     pixel_values = None
#     image_grid_thw = None
    
#     return input_ids, attention_mask, pixel_values, image_grid_thw

# def process_image_to_svg(processor,image_path):
#     """Process image-to-svg task"""
    
#     messages = [{
#         "role": "system",
#         "content": SYSTEM_PROMPT
#     }, {
#         "role": "user", 
#         "content": [
#             {"type": "text", "text": f"Task: image-to-svg\nGenerate SVG code that accurately represents the following image."},
#             {"type": "image", "image": image_path},
#         ]
#     }]
    
#     text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     image_inputs, _ = process_vision_info(messages)
    
#     inputs = processor(
#         text=[text_input], 
#         images=image_inputs,
#         truncation=True, 
#         return_tensors="pt"
#     )
    
#     input_ids = inputs['input_ids'].to(device)
#     attention_mask = inputs['attention_mask'].to(device)
#     pixel_values = inputs['pixel_values'].to(device) if 'pixel_values' in inputs else None
#     image_grid_thw = inputs['image_grid_thw'].to(device) if 'image_grid_thw' in inputs else None
    
#     return input_ids, attention_mask, pixel_values, image_grid_thw

# def generate_svg(sketch_decoder,svg_tokenizer,config,input_ids, attention_mask, pixel_values=None, image_grid_thw=None, task_type="image-to-svg"):
#     """Generate SVG"""
#     try:
#         # Clean memory before generation
#         gc.collect()
#         torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
#         print(f"Generating SVG for {task_type}...")
        
#         # Generation configuration
#         if task_type == "image-to-svg":
#             gen_config = dict(
#                 do_sample=True,
#                 temperature=0.1,
#                 top_p=0.001,
#                 top_k=1,
#                 repetition_penalty=1.05,
#                 early_stopping=True,
#             )
#         else:
#             gen_config = dict(
#                 do_sample=True,
#                 temperature=0.8,
#                 top_p=0.95,
#                 top_k=50,
#                 repetition_penalty=1.05,
#                 early_stopping=True,
#             )
        
#         # Synchronize CUDA
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()

#         # Generate SVG
#         model_config = config['model']
#         max_length = model_config['max_length']
#         output_ids = torch.ones(1, max_length+1).long().to(device) * model_config['eos_token_id']
        
#         with torch.no_grad():
#             results = sketch_decoder.transformer.generate(
#                 input_ids=input_ids, 
#                 attention_mask=attention_mask, 
#                 pixel_values=pixel_values, 
#                 image_grid_thw=image_grid_thw,
#                 max_new_tokens=max_length,
#                 num_return_sequences=1,
#                 bos_token_id=model_config['bos_token_id'],
#                 eos_token_id=model_config['eos_token_id'],
#                 pad_token_id=model_config['pad_token_id'],
#                 use_cache=True,
#                 **gen_config
#             )
            
#             results = results[:, :max_length]
            
#             output_ids[:, :results.shape[1]] = results
            
#             # Process generated tokens
#             generated_xy, generated_colors = svg_tokenizer.process_generated_tokens(output_ids)
#             print(f"Generated {len(generated_colors)} colors")

#         print('Rendering...')
#         # Convert to SVG tensors
#         svg_tensors = svg_tokenizer.raster_svg(generated_xy)
        
#         if not svg_tensors or not svg_tensors[0]:
#             return "Error: No valid SVG paths generated", None
            
#         print('Creating SVG...')
#         # Apply colors and create SVG
#         try:
#             svg = svg_tokenizer.apply_colors_to_svg(svg_tensors[0], generated_colors)
#             svg_str = svg.to_str()
            
#             # Convert to PNG for visualization
#             png_data = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'))
#             png_image = Image.open(io.BytesIO(png_data))
            
#             return svg_str, png_image
            
#         except Exception as e:
#             print(f"Error creating SVG: {e}")
#             return f"Error: {e}", None
                
#     except Exception as e:
#         print(f"Generation error: {e}")
#         import traceback
#         traceback.print_exc()
#         return f"Error: {e}", None

# def process_image_folder(sketch_decoder,svg_tokenizer,processor,config,image_files, output_dir):
#     """Process all images in a folder for image-to-svg task"""
#     # Create output directory if it doesn't exist
#     #os.makedirs(output_dir, exist_ok=True)
    
#     # Get all image files
#     # image_files = []
#     # for ext in SUPPORTED_FORMATS:
#     #     image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
#     #     image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    
#     print(f"Found {len(image_files)} images to process")

#     base_name = datetime.now().strftime('%Y%m%d_%H%M%S')

#     svg_path_list= []
#     png_image_list= []
#     for idx, image_path in enumerate(image_files):
#         #print(f"\nProcessing {idx+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
#         try:
#             # Get filename without extension
#             #base_name = Path(image_path).stem
            
#             # Process and resize image
#             processed_image = process_and_resize_image(image_path)
            
#             # Save processed image
#             processed_image_path = os.path.join(output_dir, f"{base_name}_raw{idx}.png")
#             processed_image.save(processed_image_path)
#             print(f"Saved raw image: {processed_image_path}")
            
#             # Generate SVG
#             input_ids, attention_mask, pixel_values, image_grid_thw = process_image_to_svg(processor,processed_image_path)
#             svg_code, png_image = generate_svg(sketch_decoder,svg_tokenizer,config,input_ids, attention_mask, pixel_values, image_grid_thw, "image-to-svg")
#             png_image_list.append(png_image)
#             if svg_code and not svg_code.startswith("Error"):
#                 # Save SVG file
#                 svg_path = os.path.join(output_dir, f"{base_name}_{idx}.svg")
#                 with open(svg_path, 'w') as f:
#                     f.write(svg_code)
#                 print(f"Saved SVG: {svg_path}")
#                 svg_path_list.append(svg_path)
#                 # Save PNG preview
#                 # if png_image:
#                 #     png_path = os.path.join(output_dir, f"{base_name}_{i}.png")
#                 #     png_image.save(png_path)
#                 #     print(f"Saved PNG preview: {png_path}")
#             else:
#                 print(f"Failed to generate SVG: {svg_code}")
                
#         except Exception as e:
#             print(f"Error processing {image_path}: {e}")
#             continue
#     return svg_path_list,png_image_list
    

# def process_text_file(sketch_decoder,svg_tokenizer,processor,config,text_description, output_dir):
#     """Process text file for text-to-svg task"""
#     # Create output directory if it doesn't exist
#     #os.makedirs(output_dir, exist_ok=True)
    
#     # Read text file
#     # with open(input_file, 'r', encoding='utf-8') as f:
#     #     lines = f.readlines()
    
#     #print(f"Found {len(lines)} text descriptions to process")
#     print(f"Processing {text_description}")
#     svg_path=""
#     png_image=None
#     try:
#         # Generate SVG
#         input_ids, attention_mask, pixel_values, image_grid_thw = process_text_to_svg(processor,text_description)
#         svg_code, png_image = generate_svg(sketch_decoder,svg_tokenizer,config,input_ids, attention_mask, pixel_values, image_grid_thw, "text-to-svg")
        
#         if svg_code and not svg_code.startswith("Error"):
#             # Create filename from text (sanitize for filesystem)
#             filename = text_description[:150].replace(' ', '_').replace('/', '_').replace('\\', '_')
#             filename = ''.join(c for c in filename if c.isalnum() or c in ['_', '-'])
            
#             # Save SVG file
#             svg_path = os.path.join(output_dir, f"{filename}.svg")
#             with open(svg_path, 'w') as f:
#                 f.write(svg_code)
#             print(f"Saved SVG: {svg_path}")
            
#             # Save PNG preview
#             # if png_image:
#             #     png_path = os.path.join(output_dir, f"{filename}.png")
#             #     png_image.save(png_path)
#             #     print(f"Saved PNG preview: {png_path}")

#             print("\nProcessing completed!")
#         else:
#             print(f"Failed to generate SVG: {svg_code}")
            
#     except Exception as e:
#         print(f"Error processing '{text_description}': {e}")
    
#     return svg_path,png_image
 




# def main():
#     # Set environment variable to avoid tokenizer parallelization warning
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
#     args = parse_args()
    
#     # Load models with specified weight path
#     load_models(args.weight_path)
    
#     if args.task_type == "image-to-svg":
#         if not os.path.isdir(args.input_dir):
#             print(f"Error: {args.input_dir} is not a directory")
#             return
#         process_image_folder(args.input_dir, args.output_dir)
#     else:  # text-to-svg
#         if not os.path.isfile(args.input_dir):
#             print(f"Error: {args.input_dir} is not a file")
#             return
#         process_text_file(args.input_dir, args.output_dir)
    
#     print("\nProcessing completed!")

# if __name__ == "__main__":
#     main()
