import torch
import torch.nn as nn
from torchtyping import TensorType
import numpy as np

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        pass
        self.vocabulary_size = vocabulary_size
        self.embeddings = nn.Embedding(self.vocabulary_size, 16)
        self.linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        pass
        x = self.embeddings(x)
        print(x.shape)
        x = torch.mean(x, axis=1)
        print(x.shape)
        x = self.linear(x)
        x = self.sigmoid(x)

        return x

