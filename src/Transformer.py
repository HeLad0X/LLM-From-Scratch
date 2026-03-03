import torch.nn as nn
from AttentionHead import MultiHeadAttention
from FeedForward import FeedForward
from LayerNorm import LayerNorm
from GPTConfig import GPTConfig

class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

        self.norm1 = LayerNorm(cfg.emb_dim, eps=float(getattr(cfg, "norm_eps", 1e-5)))
        self.att = MultiHeadAttention(
            d_model=cfg.emb_dim,
            context_length=cfg.max_context_length,
            dropout=cfg.dropout,
            num_heads=cfg.n_heads,
            qkv_bias=bool(getattr(cfg, "bias", False)),
            out_bias=bool(getattr(cfg, "bias", False)),
            use_sdpa=bool(getattr(getattr(cfg, "model", None), "use_sdpa", True))
        )

        self.norm2 = LayerNorm(cfg.emb_dim, eps=float(getattr(cfg, "norm_eps", 1e-5)))
        self.ff = FeedForward(cfg)

        self.drop_shortcut = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))
        return x
