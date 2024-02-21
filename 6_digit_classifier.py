import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        pass
        # Define the architecture here
        self.first_linear = nn.Linear(28*28, 512)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.second_linear = nn.Linear(512, 10)

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        pass
        # Return the model's prediction to 4 decimal places
        # x = self.first_linear(self.activation_fn(self.dropout(self.second_linear(images))))
        x = self.first_linear(images)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.second_linear(x)
        x = self.sigmoid(x)
        return x

