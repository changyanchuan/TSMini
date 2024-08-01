import sys
sys.path.append('..')

import torch
import torch.nn as nn


class StackedConv(nn.Module):
    def __init__(self, conv_in_dim, conv_out_dim, conv_k, conv_s):
        super(StackedConv, self).__init__()
        self.conv_in_dim = conv_in_dim
        self.conv_out_dim = conv_out_dim
        self.conv_k = conv_k
        self.conv_s = conv_s
        
        self.layers = nn.Sequential(
                nn.Conv1d(conv_in_dim, conv_out_dim, conv_k, conv_s),
                nn.BatchNorm1d(conv_out_dim),
                nn.LeakyReLU(),
        )

    def forward(self, src, src_len = None):
        # src: [bsz, dim, seq_len]
        src = self.layers(src)
        
        if src_len != None:
            patch_len = torch.floor( (src_len - self.conv_k) / self.conv_s + 1 ).long()
            return src, patch_len
        else:
            return src
        
        
class ConvEmbeder(nn.Module):
    def __init__(self, cin_dim, cout_dim, kernel_size, stride, patch_length, 
                 hidden_dim, out_dim, dropout = 0.1):
        # cin_dim = 7
        # hidden_dim = 16
        # cout_dim = 64
        # out_dim = 128
        super(ConvEmbeder, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.linear_in = nn.Linear(cin_dim, hidden_dim) # 7 16
        
        self.conv1 = StackedConv(hidden_dim, cout_dim//2, kernel_size, stride)
        self.conv2 = StackedConv(cout_dim//2, cout_dim, kernel_size, stride)
        self.conv3 = StackedConv(cout_dim, out_dim, kernel_size, stride)
        
        self.linear_out = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        # src = [bsz, psz, p, 7]
        # src = [bsz, seq, 7], tensor
        # src_len = [bsz], tensor
        
        src = self.linear_in(src)
        src = src.permute(0, 2, 1) # [bsz, seq_len, dim] -> [bsz, dim, seq_len]
        
        src, patch_len = self.conv1(src, src_len)
        src, patch_len = self.conv2(src, patch_len)
        src, patch_len = self.conv3(src, patch_len)
        
        src = src.permute(0, 2, 1) # -> [bsz, seq_len, dim]

        patch_embs = src + self.dropout(torch.relu(self.linear_out(src)))
        return patch_embs, patch_len
    
  