from collections import Counter

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def text_to_one_hot(texts, vocab_size):
    # Step 1: Build a vocabulary and map words to integer indices
    counter = Counter(word for text in texts for word in text.split())
    word_to_index = {word: idx + 1 for idx, (word, _) in enumerate(counter.most_common(vocab_size))}  # Start indices from 1

    # Step 2: Encode the texts as integer sequences
    integer_encoded = [[word_to_index.get(word, vocab_size + 1) for word in text.split()] for text in texts]
    return integer_encoded
