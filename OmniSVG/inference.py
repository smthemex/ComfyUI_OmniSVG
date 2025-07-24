import torch
import os
from PIL import Image
import cairosvg
import io
import gc
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from datetime import datetime

from .decoder import SketchDecoder
from .tokenizer import SVGTokenizer



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# tokenizer = None
# processor = None
# sketch_decoder = None
# svg_tokenizer = None

SYSTEM_PROMPT = "You are a multimodal SVG generation assistant capable of generating SVG code from both text descriptions and images."
SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']

# def parse_args():
#     parser = argparse.ArgumentParser(description='SVG Generator Service')
#     parser.add_argument('--input_dir', type=str, required=True,
#                        help='Input directory for images or text file path')
#     parser.add_argument('--output_dir', type=str, required=True,
#                        help='Output directory for generated SVGs')
#     parser.add_argument('--task_type', type=str, required=True,
#                        choices=['image-to-svg', 'text-to-svg'],
#                        help='Task type: image-to-svg or text-to-svg')
#     parser.add_argument('--weight_path', type=str, required=True,
#                        help='Path to model weights directory containing pytorch_model.bin')
#     parser.add_argument('--debug', action='store_true',
#                        help='Enable debug mode')
#     return parser.parse_args()

def load_models(sketch_weight_file,node_path,qwen_repo="Qwen/Qwen2.5-VL-3B-Instruct"):
    """Load models"""
    #global tokenizer, processor, sketch_decoder, svg_tokenizer
    
   
    print("Loading models...")

    tokenizer = AutoTokenizer.from_pretrained(qwen_repo, padding_side="left")
    processor = AutoProcessor.from_pretrained(qwen_repo, padding_side="left")

    sketch_decoder = SketchDecoder(qwen_repo)
    
    # sketch_weight_file = os.path.join(weight_path, "pytorch_model.bin")
    # if not os.path.exists(sketch_weight_file):
    #     raise FileNotFoundError(f"pytorch_model.bin not found in {weight_path}")
    
    print(f"Loading weights from: {sketch_weight_file}")
    sketch_dict=torch.load(sketch_weight_file,weights_only=False)
    sketch_decoder.load_state_dict(sketch_dict)
    sketch_decoder = sketch_decoder.to(device).eval()
    del sketch_dict
    gc.collect()
    
    # Initialize SVG tokenizer
    svg_tokenizer = SVGTokenizer(os.path.join(node_path,'OmniSVG/config.yaml'))
    print("Models loaded successfully!")
    return tokenizer, processor, sketch_decoder, svg_tokenizer

def process_and_resize_image(image_input, target_size=(200, 200)):
    """Process and resize image to target size"""
    if isinstance(image_input, str):
        image = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        image = Image.fromarray(image_input)

    
    #image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    return image

def process_text_to_svg(processor,text_description):
    """Process text-to-svg task"""
    
    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }, {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Task: text-to-svg\nDescription: {text_description}\nGenerate SVG code based on the above description."}
        ]
    }]
    
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_input], 
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    pixel_values = None
    image_grid_thw = None
    
    return input_ids, attention_mask, pixel_values, image_grid_thw

def process_image_to_svg(processor,image_path):
    """Process image-to-svg task"""
    
    messages = [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }, {
        "role": "user", 
        "content": [
            {"type": "text", "text": f"Task: image-to-svg\nGenerate SVG code that accurately represents the following image."},
            {"type": "image", "image": image_path},
        ]
    }]
    
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    
    inputs = processor(
        text=[text_input], 
        images=image_inputs,
        truncation=True, 
        return_tensors="pt"
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    pixel_values = inputs['pixel_values'].to(device) if 'pixel_values' in inputs else None
    image_grid_thw = inputs['image_grid_thw'].to(device) if 'image_grid_thw' in inputs else None
    
    return input_ids, attention_mask, pixel_values, image_grid_thw

def generate_svg(sketch_decoder,svg_tokenizer,config,input_ids, attention_mask, pixel_values=None, image_grid_thw=None, task_type="image-to-svg"):
    """Generate SVG"""
    try:
        # Clean memory before generation
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"Generating SVG for {task_type}...")
        
        # Generation configuration
        if task_type == "image-to-svg":
            gen_config = dict(
                do_sample=True,
                temperature=0.1,
                top_p=0.001,
                top_k=1,
                repetition_penalty=1.05,
                early_stopping=True,
            )
        else:
            gen_config = dict(
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.05,
                early_stopping=True,
            )
        
        # Synchronize CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Generate SVG
        model_config = config['model']
        max_length = model_config['max_length']
        output_ids = torch.ones(1, max_length+1).long().to(device) * model_config['eos_token_id']
        
        with torch.no_grad():
            results = sketch_decoder.transformer.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pixel_values=pixel_values, 
                image_grid_thw=image_grid_thw,
                max_new_tokens=max_length,
                num_return_sequences=1,
                bos_token_id=model_config['bos_token_id'],
                eos_token_id=model_config['eos_token_id'],
                pad_token_id=model_config['pad_token_id'],
                use_cache=True,
                **gen_config
            )
            
            results = results[:, :max_length]
            
            output_ids[:, :results.shape[1]] = results
            
            # Process generated tokens
            generated_xy, generated_colors = svg_tokenizer.process_generated_tokens(output_ids)
            print(f"Generated {len(generated_colors)} colors")

        print('Rendering...')
        # Convert to SVG tensors
        svg_tensors = svg_tokenizer.raster_svg(generated_xy)
        
        if not svg_tensors or not svg_tensors[0]:
            return "Error: No valid SVG paths generated", None
            
        print('Creating SVG...')
        # Apply colors and create SVG
        try:
            svg = svg_tokenizer.apply_colors_to_svg(svg_tensors[0], generated_colors)
            svg_str = svg.to_str()
            
            # Convert to PNG for visualization
            png_data = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'))
            png_image = Image.open(io.BytesIO(png_data))
            
            return svg_str, png_image
            
        except Exception as e:
            print(f"Error creating SVG: {e}")
            return f"Error: {e}", None
                
    except Exception as e:
        print(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}", None

def process_image_folder(sketch_decoder,svg_tokenizer,processor,config,image_files, output_dir):
    """Process all images in a folder for image-to-svg task"""
    # Create output directory if it doesn't exist
    #os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    # image_files = []
    # for ext in SUPPORTED_FORMATS:
    #     image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    #     image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    
    print(f"Found {len(image_files)} images to process")

    base_name = datetime.now().strftime('%Y%m%d_%H%M%S')

    svg_path_list= []
    png_image_list= []
    for idx, image_path in enumerate(image_files):
        #print(f"\nProcessing {idx+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            # Get filename without extension
            #base_name = Path(image_path).stem
            
            # Process and resize image
            processed_image = process_and_resize_image(image_path)
            
            # Save processed image
            processed_image_path = os.path.join(output_dir, f"{base_name}_raw{idx}.png")
            processed_image.save(processed_image_path)
            print(f"Saved raw image: {processed_image_path}")
            
            # Generate SVG
            input_ids, attention_mask, pixel_values, image_grid_thw = process_image_to_svg(processor,processed_image_path)
            svg_code, png_image = generate_svg(sketch_decoder,svg_tokenizer,config,input_ids, attention_mask, pixel_values, image_grid_thw, "image-to-svg")
            png_image_list.append(png_image)
            if svg_code and not svg_code.startswith("Error"):
                # Save SVG file
                svg_path = os.path.join(output_dir, f"{base_name}_{idx}.svg")
                with open(svg_path, 'w') as f:
                    f.write(svg_code)
                print(f"Saved SVG: {svg_path}")
                svg_path_list.append(svg_path)
                # Save PNG preview
                # if png_image:
                #     png_path = os.path.join(output_dir, f"{base_name}_{i}.png")
                #     png_image.save(png_path)
                #     print(f"Saved PNG preview: {png_path}")
            else:
                print(f"Failed to generate SVG: {svg_code}")
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    return svg_path_list,png_image_list
    

def process_text_file(sketch_decoder,svg_tokenizer,processor,config,text_description, output_dir):
    """Process text file for text-to-svg task"""
    # Create output directory if it doesn't exist
    #os.makedirs(output_dir, exist_ok=True)
    
    # Read text file
    # with open(input_file, 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    
    #print(f"Found {len(lines)} text descriptions to process")
    print(f"Processing {text_description}")
    svg_path=""
    png_image=None
    try:
        # Generate SVG
        input_ids, attention_mask, pixel_values, image_grid_thw = process_text_to_svg(processor,text_description)
        svg_code, png_image = generate_svg(sketch_decoder,svg_tokenizer,config,input_ids, attention_mask, pixel_values, image_grid_thw, "text-to-svg")
        
        if svg_code and not svg_code.startswith("Error"):
            # Create filename from text (sanitize for filesystem)
            filename = text_description[:150].replace(' ', '_').replace('/', '_').replace('\\', '_')
            filename = ''.join(c for c in filename if c.isalnum() or c in ['_', '-'])
            
            # Save SVG file
            svg_path = os.path.join(output_dir, f"{filename}.svg")
            with open(svg_path, 'w') as f:
                f.write(svg_code)
            print(f"Saved SVG: {svg_path}")
            
            # Save PNG preview
            # if png_image:
            #     png_path = os.path.join(output_dir, f"{filename}.png")
            #     png_image.save(png_path)
            #     print(f"Saved PNG preview: {png_path}")

            print("\nProcessing completed!")
        else:
            print(f"Failed to generate SVG: {svg_code}")
            
    except Exception as e:
        print(f"Error processing '{text_description}': {e}")
    
    return svg_path,png_image
 




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
