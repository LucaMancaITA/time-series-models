import torch
import torch.nn as nn
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_stacked_layers,
            batch_first=True
        )

        self.fc = nn.Linear(
            in_features=hidden_size,
            out_features=1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = F.tanh(out)
        return out


if __name__ == "__main__":

    x = torch.rand(size=(32, 100, 5))

    model = RNN(
        input_size=5,
        hidden_size=4,
        num_stacked_layers=1)
    model.to(device)

    print(x.size())
    y = model(x)
    print(y.size())
