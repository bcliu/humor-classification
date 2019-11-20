import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, weights_matrix, num_filters, window_sizes, out_classes):
        super().__init__()
        num_embeddings, embedding_dim = weights_matrix.shape
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.load_state_dict({'weight': weights_matrix})
        self.embedding.weight.requires_grad = False

        # List of convs with different window sizes
        self.convs = [nn.Conv2d(1, num_filters, (s, embedding_dim)) for s in window_sizes]
        self.pools = [nn.MaxPool1d(conv.kernel_size[1]) for conv in self.convs]
        self.fc1 = nn.Linear(num_filters * len(window_sizes), out_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        embedded = self.embedding(input).unsqueeze(1)
        conv_outputs = [F.relu(conv(embedded)).squeeze() for conv in self.convs]
        pool_outputs = [self.pools[i](conv_outputs[i]).squeeze() for i in range(len(conv_outputs))]
        concatenated = torch.cat(pool_outputs, 1)
        return self.softmax(self.fc1(concatenated))
