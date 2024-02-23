import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]]]:
        # You must start by generating batch_size different random indices in the appropriate range
        # using a single call to torch.randint()
        torch.manual_seed(0)
        data = raw_dataset.split()
        indices_list = torch.randint(
            low = 0,
            high = len(data) - context_length,
            size = (batch_size,)
        )
        print(indices_list)
        X = []
        Y = []
        for idx in indices_list:
            X.append(data[idx : idx + context_length])
            print(X)
            idx += 1
            Y.append(data[idx: idx + context_length])
            print(Y)

        return X, Y


