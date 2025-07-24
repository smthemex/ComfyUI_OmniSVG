import numpy as np
import torch
import yaml
from typing import List, Tuple, Dict, Optional, Union
from .deepsvg.difflib.tensor import SVGTensor
from .deepsvg.svglib.svg import SVG
from .deepsvg.svglib.geom import Bbox


class SVGTokenizer:
    """SVG tokenizer for converting between tokens and SVG representations"""
    
    def __init__(self, config_path: str = "/PATH/TO/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract configuration values
        self.tokens_config = self.config['tokens']
        self.coordinates_config = self.config['coordinates']
        self.colors_config = self.config['colors']
        self.svg_commands = self.config['svg_commands']
        
        self.pixel2xy = self._create_pixel2xy_mapping()
        
    def _create_pixel2xy_mapping(self) -> Dict[int, np.ndarray]:
        """Create mapping from pixel indices to xy coordinates"""
        bbox = self.coordinates_config['bbox']
        coord_pad = self.coordinates_config['coord_pad_offset']
        svg_end = self.tokens_config['svg_end']
        
        pixel2xy = {}
        x = np.linspace(0, bbox-1, bbox)
        y = np.linspace(0, bbox-1, bbox)
        xx, yy = np.meshgrid(x, y)
        xy_grid = (np.array((xx.ravel(), yy.ravel())).T).astype(int)
        
        for pixel, xy in enumerate(xy_grid):
            pixel2xy[pixel] = xy + coord_pad + svg_end
            
        return pixel2xy
    
    def token_to_color(self, color_token: int) -> str:
        try:
            color_token_start = self.colors_config['color_token_start']
            max_color_tokens = self.colors_config['max_color_tokens']
            
            # Check special color tokens
            if color_token == color_token_start:
                return "none"  # No color
            elif color_token == color_token_start + 1:
                return "currentColor"  # Special color
            
            color_index = color_token - (color_token_start + 2)
            if color_index < 0 or color_index >= max_color_tokens:
                print(f"Warning: Color token {color_token} out of range, using default color")
                return "#808080"  # Gray as default
            
            r = (color_index >> 8) & 0xF  
            g = (color_index >> 4) & 0xF  
            b = color_index & 0xF         
            
            r = (r << 4) | r
            g = (g << 4) | g
            b = (b << 4) | b
            
            return f"#{r:02x}{g:02x}{b:02x}"
            
        except Exception as e:
            print(f"Error in token_to_color: {e}")
            return "#808080"  
    
    def pixel_to_xy(self, pixel: int) -> np.ndarray:
        """Convert pixel token to xy coordinates"""
        base_offset = self.tokens_config['base_offset']
        pix_pad = self.coordinates_config['pix_pad_offset']
        svg_end = self.tokens_config['svg_end']
        
        if self.tokens_config['eom'] < pixel < pix_pad + svg_end:
            xy = np.array([pixel - base_offset, pixel - base_offset]).astype(int)
            return xy
        elif pix_pad + svg_end <= pixel < self.colors_config['cmd_fill'] + base_offset + svg_end:
            pixel_index = pixel - pix_pad - svg_end
            if pixel_index in self.pixel2xy:
                return self.pixel2xy[pixel_index] - base_offset
            else:
                raise ValueError(f"Invalid pixel index: {pixel_index}")
        else:
            raise ValueError(f"Invalid pixel token: {pixel}")
    
    def raster_svg(self, pixels: np.ndarray) -> List[List[torch.Tensor]]:
        """Convert pixel sequence to SVG tensor representation"""
        try:
            adjustment = self.tokens_config['num_end_token'] + self.tokens_config['svg_end'] + 2  # 8
            pixels = pixels - adjustment
            
            svg_tensors = []
            path_tensor = []
            i = 0
            
            while i < len(pixels):
                try:
                    pix = pixels[i]
                    
                    if pix[0] == self.svg_commands['move']:  # Move command
                        cmd_tensor = np.zeros(14)
                        cmd_tensor[0] = 0
                        
                        if i + 2 >= len(pixels):
                            break  
                            
                        cmd_tensor[12:14] = pixels[i+2]
                        start_pos = pixels[i+1]
                        end_pos = pixels[i+2]
                        
                        if np.all(start_pos == end_pos) and path_tensor:
                            svg_tensors.append(torch.tensor(path_tensor))
                            path_tensor = []
                        path_tensor.append(cmd_tensor.tolist())
                        i += 3
                        
                    elif pix[0] == self.svg_commands['line']:  # Line command
                        cmd_tensor = np.zeros(14)
                        cmd_tensor[0] = 1
                        
                        if i + 1 >= len(pixels):
                            break  
                            
                        cmd_tensor[12:14] = pixels[i+1]
                        path_tensor.append(cmd_tensor.tolist())
                        i += 2
                        
                    elif pix[0] == self.svg_commands['curve']:  # Curve command
                        cmd_tensor = np.zeros(14)
                        cmd_tensor[0] = 2
                        
                        if i + 3 >= len(pixels):
                            break  
                            
                        cmd_tensor[8:10] = pixels[i+1]
                        cmd_tensor[10:12] = pixels[i+2]
                        cmd_tensor[12:14] = pixels[i+3]
                        path_tensor.append(cmd_tensor.tolist())
                        i += 4
                        
                    elif pix[0] == self.svg_commands['arc']:  # Arc command
                        cmd_tensor = np.zeros(14)
                        cmd_tensor[0] = 3
                        
                        if i + 5 >= len(pixels):
                            break 
                            
                        radius = pixels[i+1]
                        x_axis_rot = pixels[i+2][0]
                        large_arc_flg = pixels[i+3][0]
                        sweep_flg = pixels[i+4][0]
                        end_pos = pixels[i+5]
                        
                        cmd_tensor[1:3] = radius
                        cmd_tensor[3] = x_axis_rot
                        cmd_tensor[4] = large_arc_flg
                        cmd_tensor[5] = sweep_flg
                        cmd_tensor[12:14] = end_pos
                        path_tensor.append(cmd_tensor.tolist())
                        i += 6
                        
                    elif pix[0] == self.svg_commands['close']:  # Close command
                        cmd_tensor = np.zeros(14)
                        cmd_tensor[0] = 6
                        
                        if i + 1 >= len(pixels):
                            break  
                            
                        cmd_tensor[12:14] = pixels[i+1]
                        path_tensor.append(cmd_tensor.tolist())
                        i += 2
                    else:
                        i += 1  
                        
                except IndexError:
                    print(f"Index error at position {i}, stopping SVG processing")
                    break
                    
            if path_tensor:
                svg_tensors.append(torch.tensor(path_tensor))
                
            return [svg_tensors]
            
        except Exception as e:
            print(f"Error in raster_svg: {e}")
            return []
    
    def extract_colors_from_tokens(self, tokens: List[int]) -> List[int]:
        colors = []
        base_offset = self.tokens_config['base_offset']
        color_start = self.colors_config['color_start_offset']
        color_end = self.colors_config['color_end_offset']
        
        for token in tokens:
            if color_start <= token < color_end:
                colors.append(token - 1 - base_offset)
                
        return colors
    
    def process_generated_tokens(self, output_ids: torch.Tensor) -> Tuple[np.ndarray, List[int]]:
        # Remove <bos> and <eos> tokens
        generated_pixels = output_ids[:, 1:-1].tolist()
        
        generated_xy = []
        generated_colors = []
        
        for pixel_sequence in generated_pixels:
            xy_sequence = []
            colors = []
            
            for pixel in pixel_sequence:
                try:
                    if self.tokens_config['eom'] < pixel < self.coordinates_config['pix_pad_offset'] + self.tokens_config['svg_end']:
                        xy = self.pixel_to_xy(pixel)
                        xy_sequence.append(xy)
                    elif self.coordinates_config['pix_pad_offset'] + self.tokens_config['svg_end'] <= pixel < self.colors_config['cmd_fill'] + self.tokens_config['base_offset'] + self.tokens_config['svg_end']:
                        xy = self.pixel_to_xy(pixel)
                        xy_sequence.append(xy)
                    elif self.colors_config['color_start_offset'] <= pixel < self.colors_config['color_end_offset']:
                        colors.append(pixel - 1 - self.tokens_config['base_offset'])
                except ValueError as e:
                    print(f"Error processing pixel {pixel}: {e}")
                    continue
                    
            if xy_sequence:
                generated_xy = np.vstack(xy_sequence)
            generated_colors = colors
            
        return generated_xy, generated_colors
    
    def apply_colors_to_svg(self, svg_tensors: Union[List[torch.Tensor], List[List[torch.Tensor]]], colors: Optional[List[int]]) -> SVG:
        paths = []
        bbox = self.coordinates_config['bbox']
        
        flat_tensors = []
        if svg_tensors and isinstance(svg_tensors[0], list):
            for tensor_list in svg_tensors:
                flat_tensors.extend(tensor_list)
        else:
            flat_tensors = svg_tensors
        
        if not flat_tensors:
            raise ValueError("No valid SVG tensors provided")
        
        if colors is None:
            colors = []
                    
        for i, path_tensor in enumerate(flat_tensors):
            try:
                path = SVGTensor.from_data(path_tensor)
                path = SVG.from_tensor(path.data, viewbox=Bbox(bbox))
                
                if i < len(colors):
                    color_token = colors[i]
                    actual_color = self.token_to_color(color_token)
                else:
                    actual_color = "none"
                
                for path_group in path:
                    path_group.color = actual_color
                    path_group.stroke_color = "none"
                    
                path.fill_(True)
                paths.append(path)
        
                
            except Exception as e:
                print(f"Error processing path {i}: {e}")
                continue
        
        if not paths:
            raise ValueError("No valid paths could be generated")
        path_groups = paths[0].svg_path_groups
        for i in range(1, len(paths)):
            if i < len(paths): 
                path_groups.extend(paths[i].svg_path_groups)
        
        svg = SVG(path_groups, viewbox=Bbox(bbox))
        
        return svg