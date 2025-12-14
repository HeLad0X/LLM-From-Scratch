import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.scale = self.head_dim ** 0.5

        self.W_qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        B, T, _ = x.shape

        qkv = self.W_qkv(x)                       # (B, T, 3*d_out)
        q, k, v = qkv.chunk(3, dim=-1)            # each (B, T, d_out)

        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, T, hd)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = q @ k.transpose(-2, -1)     # (B, h, T, T)
        attn_scores = attn_scores / self.scale

        mask = self.mask[:T, :T]                  # (T, T) bool
        attn_scores = attn_scores.masked_fill(mask, torch.finfo(attn_scores.dtype).min)

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v                    # (B, h, T, hd)
        out = out.transpose(1, 2).contiguous().reshape(B, T, self.d_out)  # (B, T, d_out)

        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out
