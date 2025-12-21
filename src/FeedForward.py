import torch.nn as nn
from GPTConfig import GPTConfig
class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
            nn.Dropout(cfg.dropout)
        )

    def forward(self, x):
        return self.layers(x)