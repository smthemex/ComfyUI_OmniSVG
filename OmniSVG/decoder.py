import torch.nn as nn
import torch
from transformers import AutoConfig
try :
   from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
except:
   from transformers import Qwen2_5_VLForConditionalGeneration
 
class SketchDecoder(nn.Module):
  """
  Autoregressive generative model 
  """
  
  def __init__(self,repo,**kwargs):
    super().__init__()
    self.vocab_size = 196042
    self.bos_token_id = 151643
    self.eos_token_id = 196041
    self.pad_token_id = 151643

    config = AutoConfig.from_pretrained(
          repo,
          vocab_size=self.vocab_size,
          bos_token_id=self.bos_token_id,
          eos_token_id=self.eos_token_id,
          pad_token_id=self.pad_token_id)

    self.transformer = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        repo, 
        config=config,
        ignore_mismatched_sizes=True)

    self.transformer.resize_token_embeddings(self.vocab_size)

  def forward(self, *args, **kwargs):
      raise NotImplementedError("Forward pass not included in open-source version")
    