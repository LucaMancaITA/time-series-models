import torch
import torch.nn as nn


class FakeNewsModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embd = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=100,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(
            in_features=100,
            out_features=1
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.embd(x)
        h0 = torch.zeros(1, x.size(0), 100).to("cpu")
        c0 = torch.zeros(1, x.size(0), 100).to("cpu")
        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.sig(self.fc(x))[:, -1]
        return x


if __name__ == "__main__":
    vocab_size = 5000
    embedding_dim = 40
    seq_len = 20
    x = torch.rand(size=(64, seq_len))
    x = x.long()
    print(x.size())
    model = FakeNewsModel(vocab_size, seq_len, embedding_dim)
    y = model(x)
    print(y.shape)
