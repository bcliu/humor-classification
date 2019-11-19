import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, weights_matrix):
        super().__init__()
        num_embeddings, embedding_dim = weights_matrix.shape
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.load_state_dict({'weight': weights_matrix})
        self.embedding.weight.requires_grad = False

    def forward(self):
        pass