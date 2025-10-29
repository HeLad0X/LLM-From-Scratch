import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, tokenizer, d_model: int = 64, max_context_length: int = 128, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = len(tokenizer)
        self.d_model = d_model
        self.max_context_length = max_context_length

        pad_id = tokenizer.stoi.get("<PAD>", None)
        self.tok_emb = nn.Embedding(self.vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_context_length, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: Tensor of shape [B, T] with token IDs
        returns: Tensor [B, T, d_model]
        """
        B, T = x.shape
        tok = self.tok_emb(x) * (self.d_model ** 0.5)
        pos_ids = torch.arange(T, device=x.device)
        pos = self.pos_emb(pos_ids)[None, :, :]  # shape [1, T, d_model]
        return self.dropout(tok + pos)
