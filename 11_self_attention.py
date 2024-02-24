import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.key_linear = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_linear = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_linear = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        key = self.key_linear(embedded)
        query = self.query_linear(embedded)
        value = self.value_linear(embedded)

        key_t = torch.transpose(key, 1, 2)

        scores = torch.matmul(query, key_t) # @ is the same as torch.matmul()
        cl, ad= key.shape[1], key.shape[2]
        scores = scores / (ad ** 0.5)

        lower_triangular = torch.tril(torch.ones(cl, cl))
        mask = lower_triangular == 0
        scores = scores.masked_fill(mask, float('-inf'))
        scores = nn.functional.softmax(scores, dim = 2)
        attention_output = torch.matmul(scores, value)

        return torch.round(attention_output, decimals=4)
