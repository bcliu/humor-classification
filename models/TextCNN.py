import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, weights_matrix, num_filters, window_sizes, out_classes, dropout_rate=0.1):
        super().__init__()
        num_embeddings, embedding_dim = weights_matrix.shape
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.load_state_dict({'weight': weights_matrix})
        self.embedding.weight.requires_grad = False

        # List of convs with different window sizes
        self.convs = [nn.Conv2d(1, num_filters, (s, embedding_dim)) for s in window_sizes]
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(num_filters * len(window_sizes), out_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, train=True):
        embedded = self.embedding(input).unsqueeze(1)
        conv_outputs = [F.relu(conv(embedded)).squeeze() for conv in self.convs]
        pool_outputs = [F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2) for conv_out in conv_outputs]
        concatenated = torch.cat(pool_outputs, 1)
        if train:
            concatenated = self.dropout(concatenated)
        return self.softmax(self.fc1(concatenated))
