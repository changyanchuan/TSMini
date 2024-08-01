# References:
# https://github.com/meta-llama/llama/
# https://github.com/hkproj/pytorch-llama
# https://github.com/JAYANDJEAN/From_Transformer_to_GPTs

import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

@dataclass
class ModelArgs:
    dim: int = 4096 
    n_layers: int = 32
    n_heads: int = 32
    hidden_dim: int = 11008
    norm_eps: float = 1e-5
    max_seq_len: int = 2048

    device: str = None
    dropout: float = 0.1


def precompute_freqs_cis(head_dim: int, seq_len: int, device: str, base: float = 10000.0):
    assert head_dim % 2 == 0, "dimension must be divisible by 2"
    # Shape: (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (head_dim / 2)
    theta = 1.0 / (base ** (theta_numerator / head_dim)).to(device)
    # Shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    # Shape: (seq_len) outer_product* (head_dim / 2) -> (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    # Shape: (seq_len, head_dim / 2) -> (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embeddings(x: Tensor, freqs_complex: Tensor, device: str):
    # Shape: (batch_size, seq_len, n_head, head_dim) -> (batch_size, seq_len, n_head, head_dim/2)
    # n_head * head_dim = dim
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Shape: (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Shape: (batch_size, seq_len, n_head, head_dim/2) * (1, seq_len, 1, head_dim/2) 
    # = (batch_size, seq_len, n_head, head_dim/2)
    x_rotated = x_complex * freqs_complex
    # Shape: (batch_size, seq_len, n_head, head_dim/2) -> (batch_size, seq_len, n_head, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # Shape: (batch_size, seq_len, n_head, head_dim/2, 2) -> (batch_size, seq_len, n_head, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor):
        # (batch_size, seq_len, dim) * (batch_size, seq_len, 1) = (batch_size, seq_len, dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor):
        # (dim) * (batch_size, seq_len, dim) = (batch_size, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads 
        self.head_dim = args.dim // args.n_heads  # dim of every head

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        # Linear transformation for output.
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: Tensor, freqs_complex: Tensor, mask: Optional[Tensor]):
        # x: (batch_size, seq_len, dim)
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # -> (batch_size, seq_len, n_head_q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # -> (batch_size, seq_len, n_head_q, head_dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, self.args.device)
        # -> (batch_size, seq_len, n_head_kv, head_dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, self.args.device)

        # -> (batch_size, n_head_q, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        # -> (batch_size, n_head_q, seq_len_kv, head_dim)
        xk = xk.transpose(1, 2)
        # -> (batch_size, n_head_q, seq_len_kv, head_dim)
        xv = xv.transpose(1, 2)

        # (batch_size, n_head_q, seq_len, head_dim) @ (batch_size, n_head_q, head_dim, seq_len_kv)
        # -> (batch_size, n_head_q, seq_len, seq_len_kv)
        # print(f'score = {xq.shape} * {xk.transpose(2, 3).shape}')
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.view(batch_size, 1, 1, seq_len).expand(-1, self.n_heads, -1, -1)
            scores = scores + mask  # (batch_size, n_head_q, seq_len, seq_len_kv)
        # -> (batch_size, n_head_q, seq_len, seq_len_kv)
        scores = self.dropout(F.softmax(scores.float(), dim=-1).type_as(xq))

        # (batch_size, n_head_q, seq_len, seq_len_kv) @ (batch_size, n_head_q, seq_len_kv, head_dim)
        # -> (batch_size, n_head_q, seq_len, head_dim)
        output = torch.matmul(scores, xv)
        # (batch_size, n_head_q, seq_len, head_dim) -> (batch_size, seq_len, n_head_q, head_dim)
        # -> (batch_size, seq_len, dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def forward(self, x: Tensor):
        # 为什么设计成这样? 看 GLU Variants Improve Transformer
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, hidden_dim)
        swish = F.silu(self.w1(x))
        # (batch_size, seq_len, dim) -> (batch_size, seq_len, hidden_dim)
        x_V = self.w3(x)
        # (batch_size, seq_len, hidden_dim) * (batch_size, seq_len, hidden_dim)
        # -> (batch_size, seq_len, hidden_dim)
        x = swish * x_V
        # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, dim)
        x = self.w2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Attention, FeedForward 都是输入：(batch_size, seq_len, dim)
        # 输出：(batch_size, seq_len, dim)
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.feed_forward = FeedForward(args)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: Tensor, freqs_complex: Tensor, mask: Optional[Tensor]):
        # (batch_size, seq_len, dim) + (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        h = x + self.attention(self.attention_norm(x), freqs_complex, mask)
        # (batch_size, seq_len, dim) + (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(DecoderBlock(args))

        self.freqs_complex = precompute_freqs_cis(self.args.dim // self.args.n_heads,
                                                  self.args.max_seq_len * 2,
                                                  self.args.device)

    def forward(self, src, attn_mask, src_padding_mask, src_len):
        # src: [seq_len, batch_size, emb_size]
        # attn_mask: [seq_len, seq_len]
        # src_padding_mask: [batch_size, seq_len]
        # src_len: [batch_size]

        seqlen, _, _  = src.shape
        
        freqs_complex = self.freqs_complex[0: seqlen]
        
        mask = torch.zeros_like(src_padding_mask, dtype = src_padding_mask.dtype, device = src_padding_mask.device) \
                    .masked_fill_(src_padding_mask, float('-inf'))

        h = src.transpose(0, 1) # -> batch_size, seq_len, emb_dim
        for layer in self.layers:
            h = layer(h, freqs_complex, mask)
        
        mask = 1 - src_padding_mask.unsqueeze(-1).expand(h.shape).float() # [batch_size, seq_len, emb_dim]
        rtn = torch.sum(mask * h, 1) # [batch_size, emb_dim]
        rtn = rtn / src_len.unsqueeze(-1).expand(rtn.shape)

        return rtn