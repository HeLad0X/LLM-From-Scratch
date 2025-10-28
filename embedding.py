import torch

max_length = 4
output_dim = 256

pos_embedding_layer = torch.nn.Embedding(max_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))

print(pos_embeddings.shape)
