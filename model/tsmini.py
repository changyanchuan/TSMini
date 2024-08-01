import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from model.msa_llm import ModelArgs, LlamaModel as MSALLM
from model.embeder import ConvEmbeder

from utils.traj import *

class TSMini(nn.Module):
    def __init__(self):
        super(TSMini, self).__init__()
        
        self.embeder = ConvEmbeder(7, 
                                Config.tsmini_conv_channel_dim, 
                                Config.tsmini_conv_kernel_size,
                                Config.tsmini_conv_stride,
                                None,
                                Config.tsmini_conv_hidden_in_dim,
                                Config.tsmini_patch_emb_dim)
            
        _msa_args = ModelArgs(dim = Config.seq_embedding_dim,
                            n_layers = Config.tsmin_trans_attention_layer, 
                            n_heads = Config.tsmin_trans_attention_head,
                            hidden_dim = Config.tsmin_trans_hidden_dim, 
                            max_seq_len = Config.max_traj_len,
                            device = Config.device,
                            dropout = Config.tsmin_trans_attention_dropout)
        
        self.trajenc = MSALLM(_msa_args)
        
        self.output = nn.Linear(Config.seq_embedding_dim, Config.seq_embedding_dim)
        

    def forward(self, trajs, trajs_len):
        device = next(self.parameters()).device
        
        patch_embs, patch_len = self.embeder(trajs, trajs_len)
        patch_len_max = patch_len.max().item()
        src_padding_mask = torch.arange(patch_len_max, device = device)[None, :] >= patch_len[:, None]
        
        kwargs = {'src': patch_embs.permute(1, 0, 2), 
                    'attn_mask': None, 
                    'src_padding_mask': src_padding_mask, 
                    'src_len': patch_len}
        trajs_embs = self.trajenc(**kwargs)
        trajs_embs = F.normalize(self.output(trajs_embs), dim=1)
        return trajs_embs
