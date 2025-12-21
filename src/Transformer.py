import torch.nn as nn
from AttentionHead import MultiHeadAttention
from FeedForward import FeedForward
from LayerNorm import LayerNorm
from GPTConfig import GPTConfig

class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

        self.norm1 = LayerNorm(cfg.emb_dim)
        self.att = MultiHeadAttention(
            d_model=cfg.emb_dim,
            context_length=cfg.max_context_length,
            dropout=cfg.dropout,
            num_heads=cfg.n_heads,
            use_sdpa=False
        )

        self.norm2 = LayerNorm(cfg.emb_dim)
        self.ff = FeedForward(cfg)

        # keep this if you want extra residual dropout; otherwise remove
        self.drop_shortcut = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))
        return x
