import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, context_length, dropout, num_heads, qkv_bias=False, use_sdpa=True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_sdpa = use_sdpa

        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        if T > self.mask.size(0):
            raise ValueError(f"T={T} exceeds context_length={self.mask.size(0)}")

        qkv = self.W_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.attn_dropout.p if self.training else 0.0
            )
        else:
            att = (q @ k.transpose(-2, -1)) * self.scale
            mask = self.mask[:T, :T]
            mask_value = -1e4 if att.dtype in (torch.float16, torch.bfloat16) else -1e9
            att = att.masked_fill(mask, mask_value)
            att = self.attn_dropout(torch.softmax(att, dim=-1))
            out = att @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out
