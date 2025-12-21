import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, tokenizer, d_model, max_context_length, dropout):
        super().__init__()
        vocab_size = len(tokenizer)
        pad_id = getattr(tokenizer, "stoi", {}).get("<PAD>", None)

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_context_length, d_model)
        self.drop = nn.Dropout(dropout)

        self.max_context_length = max_context_length
        self.register_buffer("pos_ids", torch.arange(max_context_length, dtype=torch.long), persistent=False)

    def forward(self, x):
        B, T = x.shape
        if T > self.max_context_length:
            raise ValueError(f"T={T} exceeds max_context_length={self.max_context_length}")

        tok = self.tok_emb(x)                          # [B, T, d]
        pos = self.pos_emb(self.pos_ids[:T])[None, :, :]  # [1, T, d]
        return self.drop(tok + pos)
