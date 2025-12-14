import torch
from torch import nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out, device = torch.device("cuda"), qkv_bias = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias).to(device)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias).to(device)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias).to(device)

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_score = queries @ keys.T
        attn_weights = torch.softmax(
            attn_score / (keys.shape[-1] ** 0.5), dim = -1
        )

        context_vector = attn_weights @ values

        return context_vector