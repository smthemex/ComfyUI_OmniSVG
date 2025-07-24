import gradio as gr
import torch
import os
from PIL import Image
import cairosvg
import io
import tempfile
import argparse
import gc
import yaml
import glob


from decoder import SketchDecoder
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tokenizer import SVGTokenizer

with open('/PATH/TO/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = None
processor = None
sketch_decoder = None
svg_tokenizer = None

# System prompt
SYSTEM_PROMPT = "You are a multimodal SVG generation assistant capable of generating SVG code from both text descriptions and images."
SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SVG Generator Service')
    parser.add_argument('--listen', type=str, default='0.0.0.0', 
                       help='Listen address (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=7860, 
                       help='Port number (default: 7860)')
    parser.add_argument('--share', action='store_true', 
                       help='Enable gradio share link')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode')
    return parser.parse_args()

def load_models():
    """Load models"""
    global tokenizer, processor, sketch_decoder, svg_tokenizer
    
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", padding_side="left")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", padding_side="left")

        sketch_decoder = SketchDecoder()
        
        sketch_weight_path = "/PATH/TO/OmniSVG-3B"
        sketch_decoder.load_state_dict(torch.load(sketch_weight_path))
        sketch_decoder = sketch_decoder.to(device).eval()
        
        svg_tokenizer = SVGTokenizer('/PATH/TO/config.yaml')


def process_and_resize_image(image_input, target_size=(200, 200)):
    """Process and resize image to target size"""
    if isinstance(image_input, str):
        image = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        image = Image.fromarray(image_input)
    
    
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    return image

def get_example_images():
    """Get example images from the examples directory"""
    example_dir = "./examples"
    example_images = []
    
    if os.path.exists(example_dir):
        for ext in SUPPORTED_FORMATS:
            pattern = os.path.join(example_dir, f"*{ext}")
            example_images.extend(glob.glob(pattern))
        
        example_images.sort()
    
    return example_images

def process_text_to_svg(text_description):
    """Process text-to-svg task"""
    load_models()
    
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

def process_image_to_svg(image_path):
    """Process image-to-svg task"""
    load_models()
    
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

def generate_svg(input_ids, attention_mask, pixel_values=None, image_grid_thw=None, task_type="image-to-svg"):
    """Generate SVG"""
    try:
        # Clean memory before generation
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"Generating SVG for {task_type}...")
        
        # Generation configuration, just adjust for better results.
        if task_type == "image-to-svg":
            #Image-to-SVG configuration
            gen_config = dict(
                do_sample=True,
                temperature=0.1,
                top_p=0.001,
                top_k=1,
                num_beams=5,
                repetition_penalty=1.05,
            )
        else:
            #Text-to-SVG configuration
            gen_config = dict(
                do_sample=True,
                temperature=0.1,
                top_p=0.001,
                top_k=1,
                repetition_penalty=1.05,
                early_stopping=True,
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Generate SVG
        model_config = config['model']
        max_length = model_config['max_length']
        output_ids = torch.ones(1, max_length).long().to(device) * model_config['eos_token_id']
        
        with torch.no_grad():
            results = sketch_decoder.transformer.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                pixel_values=pixel_values, 
                image_grid_thw=image_grid_thw,
                max_new_tokens=max_length-1,
                num_return_sequences=1,
                bos_token_id=model_config['bos_token_id'],
                eos_token_id=model_config['eos_token_id'],
                pad_token_id=model_config['pad_token_id'],
                use_cache=True,
                **gen_config
            )
            results = results[:, :max_length-1]
            output_ids[:, :results.shape[1]] = results
        
            generated_xy, generated_colors = svg_tokenizer.process_generated_tokens(output_ids)

        svg_tensors = svg_tokenizer.raster_svg(generated_xy)
        if not svg_tensors or not svg_tensors[0]:
            return "Error: No valid SVG paths generated", None
            
        print('Creating SVG...')

        svg = svg_tokenizer.apply_colors_to_svg(svg_tensors[0], generated_colors)
        
        svg_str = svg.to_str()
        
        # Convert to PNG for visualization
        png_data = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'))
        png_image = Image.open(io.BytesIO(png_data))
        
        return svg_str, png_image
                
    except Exception as e:
        print(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}", None

def gradio_image_to_svg(image):
    """Gradio interface function - image-to-svg"""
    if image is None:
        return "Please upload an image", None
    processed_image = process_and_resize_image(image)
    
    # Save temporary image file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        processed_image.save(tmp_file.name, format='PNG')
        tmp_path = tmp_file.name
    
    try:
        input_ids, attention_mask, pixel_values, image_grid_thw = process_image_to_svg(tmp_path)
        svg_code, png_image = generate_svg(input_ids, attention_mask, pixel_values, image_grid_thw, "image-to-svg")
        return svg_code, png_image
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def gradio_text_to_svg(text_description):
    """Gradio interface function - text-to-svg"""
    if not text_description or text_description.strip() == "":
        return "Please enter a description", None
    
    input_ids, attention_mask, pixel_values, image_grid_thw = process_text_to_svg(text_description)
    svg_code, png_image = generate_svg(input_ids, attention_mask, pixel_values, image_grid_thw, "text-to-svg")
    return svg_code, png_image

def create_interface():
    # Example texts
    example_texts = [
        "A red heart shape with rounded corners.",
        "A yellow star with five points.",
        "Cloud icon with an upward arrow symbolizes uploading or cloud storage.",
        "A brown chocolate bar is depicted in four square segments with a shiny glossy finish.",
        "A colorful moving truck icon with a red and orange cargo container.",
        "A gray padlock icon symbolizes security and protection.",
        "A light blue T-shirt icon is outlined with a bold blue border.",
        "A person in a blue shirt and dark pants stands with one hand in a pocket gesturing outward.",
    ]
    example_images = get_example_images()
    
    with gr.Blocks(title="OmniSVG Demo Page", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# OmniSVG Demo Page")
        gr.Markdown("Generate SVG code from images or text descriptions")
        
        with gr.Tabs():
            # Image-to-SVG tab
            with gr.TabItem("Image-to-SVG"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Input Image", 
                            type="pil",
                            image_mode="RGBA"
                        )
                        if example_images:
                            gr.Examples(
                                examples=example_images,
                                inputs=[image_input],
                                label="Example Images (click to use)",
                                examples_per_page=10
                            )
                        image_generate_btn = gr.Button("Generate SVG", variant="primary")
                    
                    with gr.Column():
                        image_svg_output = gr.Textbox(
                            label="Generated SVG Code", 
                            lines=10,
                            max_lines=20,
                            show_copy_button=True
                        )
                        image_png_preview = gr.Image(label="SVG Preview", type="pil")
                
                image_generate_btn.click(
                    fn=gradio_image_to_svg,
                    inputs=[image_input],
                    outputs=[image_svg_output, image_png_preview],
                    queue=True
                )
            
            # Text-to-SVG tab
            with gr.TabItem("Text-to-SVG"):
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Description",
                            placeholder="Enter SVG description, e.g.: a red circle with a blue square inside",
                            lines=3
                        )
                        
                        # Add example texts
                        gr.Examples(
                            examples=[[text] for text in example_texts],
                            inputs=[text_input],
                            label="Example Descriptions (click to use)",
                            examples_per_page=10
                        )
                        
                        text_generate_btn = gr.Button("Generate SVG", variant="primary")
                    
                    with gr.Column():
                        text_svg_output = gr.Textbox(
                            label="Generated SVG Code", 
                            lines=10,
                            max_lines=20,
                            show_copy_button=True
                        )
                        text_png_preview = gr.Image(label="SVG Preview", type="pil")
                
                text_generate_btn.click(
                    fn=gradio_text_to_svg,
                    inputs=[text_input],
                    outputs=[text_svg_output, text_png_preview],
                    queue=True
                )
        
        # Add usage instructions
        gr.Markdown("""
        ## Usage Instructions
        - **Image-to-SVG**: Upload a PNG image and click "Generate SVG"
        - **Text-to-SVG**: Enter a text description or click an example, then click "Generate SVG"
        
        """)
    
    return demo

if __name__ == "__main__":
    # Set environment variable to avoid tokenizer parallelization warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    args = parse_args()
    
    # Load models before starting
    print("Loading models...")
    load_models()
    print("Models loaded successfully!")
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(
        server_name=args.listen,
        server_port=args.port,
        share=args.share,
        debug=args.debug
    )
