import torch.nn as nn
import torch
import yaml
import os
from transformers import AutoConfig
from accelerate import init_empty_weights,infer_auto_device_map
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
  
from .util import load_checkpoint_and_dispatch_
from safetensors.torch import load_file

def load_config(config_path=None):
    """Load configuration from config.yaml"""
    if config_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(current_dir, "config.yaml"),
            os.path.join(current_dir, "..", "config.yaml"),
            "config.yaml"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        if config_path is None:
            raise FileNotFoundError("config.yaml not found")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


class SketchDecoder(nn.Module):
    """
    Autoregressive generative model
    """
    def __init__(self, config_path=None, model_path=None,clip_path=None,repo="",attn="sdpa", **kwargs):
        super().__init__()
        
        config_data = load_config(config_path)
        model_size=kwargs["model_size"]
        use_accelerate=kwargs["use_accelerate"]

        model_config = config_data.get('model', {})
        huggingface_config = config_data.get('huggingface', {})
        self.bos_token_id = model_config['bos_token_id']
        self.eos_token_id = model_config['eos_token_id']
        self.pad_token_id = model_config['pad_token_id']
        
        self.vocab_size = config_data.get("models",{}).get(model_size,"8B").get("model",{}).get(
            'vocab_size', 
            max(self.bos_token_id, self.eos_token_id, self.pad_token_id) + 1
        )

        # 方案1：先用原始配置加载模型，然后再调整词表大小
        if (model_path is None and clip_path is None) or model_path is not None:
            print("Loading qwen repo ...")
            if model_path is None: 
                model_path = huggingface_config['qwen_model']   
            
            self.transformer = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16, 
                attn_implementation=attn,
                device_map="auto" if use_accelerate else None,
            )             
        else: # 单体模型加载方案 
            print("Loading DIT model...")
            if "3b" in clip_path.lower() and model_size=="8B":
                raise ValueError("3B DIT model cannot be used with 8B model")
            elif "7b" in clip_path.lower() and model_size=="4B":
                raise ValueError("7B DIT model cannot be used with 4B model")
            config=AutoConfig.from_pretrained(repo)
            if use_accelerate:
                with init_empty_weights():
                    transformer = Qwen2_5_VLForConditionalGeneration._from_config(config)
            else:
                transformer = Qwen2_5_VLForConditionalGeneration._from_config(config)
            state_dict = torch.load(clip_path,weights_only=False, map_location='cpu')   if not clip_path.endswith('.safetensors') else load_file(clip_path)
            has_language_keys = any("language_model" in k for k in transformer.state_dict().keys()) # new transformer vesrion load language model dict
            if has_language_keys:
                print("New transformer version,try to load language model dict")
                new_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("transformer.visual."):
                        new_k = k.replace("transformer.visual.", "transformer.model.visual.")
                    elif k.startswith("transformer.model.layers.") or k.startswith("transformer.model.embed_tokens.") or k.startswith("transformer.model.norm.") :
                        new_k = k.replace("transformer.model.", "transformer.model.language_model.")
                    else:
                        new_k = k
                    new_dict[new_k] = v
                if use_accelerate:
                    self.transformer=load_checkpoint_and_dispatch_(transformer, new_dict, device_map="auto",dtype=torch.bfloat16)
                else:
                    self.transformer.load_state_dict(new_dict, strict=False)
                del state_dict,new_dict
            else:
                if use_accelerate:
                    self.transformer=load_checkpoint_and_dispatch_(transformer, state_dict, device_map="auto",dtype=torch.bfloat16)
                else:
                    self.transformer.load_state_dict(state_dict, strict=False)
                del state_dict
            self.transformer.eval() 
      
        try:
            # try standard resize
            self.transformer.resize_token_embeddings(self.vocab_size)
        except RuntimeError:  
            print("Model has meta tensors, using manual resize to avoid meta tensor issues...")
            # When we have meta tensors, we can't use the standard resize_token_embeddings
            # because it tries to compute statistics on the tensors (mean, covariance)
            # which requires accessing tensor values (.item() calls)
            # Instead, we use our manual resize method
            self._manual_resize_token_embeddings(self.vocab_size)
 
        
        # 更新特殊 token id
        self.transformer.config.bos_token_id = self.bos_token_id
        self.transformer.config.eos_token_id = self.eos_token_id
        self.transformer.config.pad_token_id = self.pad_token_id

    def _manual_resize_token_embeddings(self, new_num_tokens, hidden_size=3584):
        """Manually resize token embeddings when standard method fails on meta tensors"""
        model = self.transformer
        
        # 获取设备信息
        model_device = None
        model_dtype = None
        for param in model.parameters():
            if param.device.type != 'meta':
                model_device = param.device
                model_dtype = param.dtype
                break
        
        if model_device is None:
            model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_dtype = torch.bfloat16
        
        # 获取旧的嵌入层
        old_input_embeddings = model.get_input_embeddings()
        old_num_tokens = old_input_embeddings.weight.size(0) if hasattr(old_input_embeddings, 'weight') and old_input_embeddings.weight.device.type != 'meta' else 0
        
        # 创建新的嵌入层，使用正确的隐藏维度
        new_input_embeddings = nn.Embedding(new_num_tokens, hidden_size)
        new_input_embeddings = new_input_embeddings.to(device=model_device, dtype=model_dtype)
        
        # 如果可能，复制旧权重
        if old_num_tokens > 0 and old_input_embeddings.weight.device.type != 'meta':
            try:
                old_hidden_size = old_input_embeddings.weight.size(1)
                if old_hidden_size == hidden_size:
                    # 形状匹配，可以直接复制
                    copy_tokens = min(old_num_tokens, new_num_tokens)
                    new_input_embeddings.weight.data[:copy_tokens] = old_input_embeddings.weight.data[:copy_tokens]
                else:
                    # 形状不匹配，需要适应
                    print(f"Adapting embedding dimensions from {old_hidden_size} to {hidden_size}")
                    copy_tokens = min(old_num_tokens, new_num_tokens)
                    if hidden_size > old_hidden_size:
                        # 扩展维度（填充0）
                        new_input_embeddings.weight.data[:copy_tokens, :old_hidden_size] = old_input_embeddings.weight.data[:copy_tokens]
                    else:
                        # 缩减维度
                        new_input_embeddings.weight.data[:copy_tokens] = old_input_embeddings.weight.data[:copy_tokens, :hidden_size]
            except Exception as e:
                print(f"Warning: Could not copy embeddings: {e}")
        else:
            # 初始化权重
            nn.init.xavier_uniform_(new_input_embeddings.weight.data)
        
        # 替换嵌入层
        model.set_input_embeddings(new_input_embeddings)
        
        # 处理lm_head
        if hasattr(model, 'lm_head') and model.lm_head is not None:
            if getattr(model.config, 'tie_word_embeddings', False):
                model.lm_head.weight = new_input_embeddings.weight
                # 确保没有偏置项（绑定权重时不应有偏置）
                if hasattr(model.lm_head, 'bias') and model.lm_head.bias is not None:
                    delattr(model.lm_head, 'bias')
            else:
                old_lm_head = model.lm_head
                old_out_features = old_lm_head.out_features if hasattr(old_lm_head, 'out_features') else old_num_tokens
                old_in_features = old_lm_head.in_features if hasattr(old_lm_head, 'in_features') else hidden_size
                
                # 创建新的lm_head，注意偏置项
                new_lm_head = nn.Linear(hidden_size, new_num_tokens, bias=hasattr(old_lm_head, 'bias') and old_lm_head.bias is not None)
                new_lm_head = new_lm_head.to(device=model_device, dtype=model_dtype)
                
                # 如果可能，复制权重
                if (hasattr(old_lm_head, 'weight') and old_lm_head.weight.device.type != 'meta' and
                    old_out_features > 0 and old_in_features > 0):
                    try:
                        copy_out = min(old_out_features, new_num_tokens)
                        copy_in = min(old_in_features, hidden_size)
                        
                        if copy_out > 0 and copy_in > 0:
                            if hidden_size > old_in_features:
                                new_lm_head.weight.data[:copy_out, :old_in_features] = old_lm_head.weight.data[:copy_out, :old_in_features]
                            else:
                                new_lm_head.weight.data[:copy_out] = old_lm_head.weight.data[:copy_out, :hidden_size]
                                
                            # 复制偏置（如果有）
                            if (hasattr(old_lm_head, 'bias') and old_lm_head.bias is not None and 
                                hasattr(new_lm_head, 'bias') and new_lm_head.bias is not None):
                                new_lm_head.bias.data[:copy_out] = old_lm_head.bias.data[:copy_out]
                    except Exception as e:
                        print(f"Warning: Could not copy lm_head weights: {e}")
                else:
                    # 初始化权重
                    nn.init.xavier_uniform_(new_lm_head.weight.data)
                    if new_lm_head.bias is not None:
                        nn.init.zeros_(new_lm_head.bias.data)
                
                model.lm_head = new_lm_head
        
        # 更新配置
        model.config.vocab_size = new_num_tokens
        
        # 安全更新text_config（如果存在）
        if hasattr(model.config, 'text_config') and model.config.text_config is not None:
            model.config.text_config.vocab_size = new_num_tokens
        
        return new_input_embeddings

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward pass not included in open-source version")
