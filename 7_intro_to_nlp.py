import torch
import torch.nn as nn
from torchtyping import TensorType

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        unique_words = set()
        for sentence in positive:
            positive_words = sentence.split()
            for each_word in positive_words:
                unique_words.add(each_word)
        for sentence in negative:
            negative_words = sentence.split()
            for each_word in negative_words:
                unique_words.add(each_word)

        # Mapping of unique words to numbers
        unique_words_sorted = sorted(list(unique_words))
        mapping = {}
        for i in range(len(unique_words_sorted)):
            mapping[unique_words_sorted[i]] = i + 1

        # Final list of 2 * N x T
        output = []
        for sentence in positive:
            pos_words = []
            for word in sentence.split():
                pos_words.append(mapping[word])
            output.append(torch.tensor(pos_words))
        for sentence in negative:
            neg_words = []
            for word in sentence.split():
                neg_words.append(mapping[word])
            output.append(torch.tensor(neg_words))

        return nn.utils.rnn.pad_sequence(output, batch_first=True,  padding_value=0.0)


