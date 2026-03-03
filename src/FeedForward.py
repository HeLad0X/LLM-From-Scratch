import torch
import torch.nn as nn
import torch.nn.functional as F

from GPTConfig import GPTConfig


def _round_up(n: int, multiple: int = 8) -> int:
    if multiple <= 1:
        return max(1, n)
    return max(multiple, ((n + multiple - 1) // multiple) * multiple)


class FeedForward(nn.Module):
    """
    SwiGLU MLP that respects cfg.mlp_ratio and cfg.bias.

    We shrink the hidden width to ~2/3 of the old GELU FFN hidden size so the
    parameter count stays close to a standard 4x GELU MLP.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        emb_dim = int(cfg.emb_dim)
        mlp_ratio = float(getattr(cfg, "mlp_ratio", 4.0) or 4.0)
        use_bias = bool(getattr(cfg, "bias", False))

        base_hidden = max(1, int(round(mlp_ratio * emb_dim)))
        hidden_dim = _round_up(int(round((2.0 * base_hidden) / 3.0)), multiple=8)

        self.gate_proj = nn.Linear(emb_dim, hidden_dim, bias=use_bias)
        self.up_proj = nn.Linear(emb_dim, hidden_dim, bias=use_bias)
        self.down_proj = nn.Linear(hidden_dim, emb_dim, bias=use_bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x
