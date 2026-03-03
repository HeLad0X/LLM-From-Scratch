import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        # Delegate to PyTorch's optimized implementation (better performance + AMP behavior).
        self.norm = nn.LayerNorm(emb_dim, eps=eps)

    def forward(self, x):
        return self.norm(x)
