import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
from GPTConfig import GPTConfig
from Transformer import TransformerBlock
from LayerNorm import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

        # Embedding layer
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.max_context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.dropout)
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg=cfg) for _ in range(cfg.n_layers)]
        )
        self.activation_checkpointing = bool(
            getattr(getattr(cfg, "model", None), "activation_checkpointing", False)
        )

        self.final_norm = LayerNorm(cfg.emb_dim, eps=float(getattr(cfg, "norm_eps", 1e-5)))
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        if self.activation_checkpointing and self.training:
            for block in self.trf_blocks:
                try:
                    x = checkpoint(block, x, use_reentrant=False)
                except TypeError:
                    x = checkpoint(block, x)
        else:
            x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        return logits
