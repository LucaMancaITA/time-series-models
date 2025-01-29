import numpy as np

import torch
import torch.nn as nn


device = "cuda" if torch.cuda.is_available() else "cpu"


class TSTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_encoder_layers, dim_feedforward, dropout, seq_length):
        super(TSTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = self._generate_positional_encoding(seq_length, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.fc_out = nn.Linear(d_model, 1)

    def _generate_positional_encoding(self, seq_length, d_model):
        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        # Imput embedding
        x = self.input_projection(x) + self.positional_encoding[:, :x.size(1), :]
        # Encoder block
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        # Fully connected layer
        x = self.fc_out(x)
        return x

if __name__ == "__main__":

    x = torch.rand(size=(32, 100, 5))

    model = TSTransformer(
        input_dim=5,
        d_model=32,
        n_heads=1,
        num_encoder_layers=2,
        dim_feedforward=1,
        dropout=0.2,
        seq_length=100
    )

    print(x.size())
    y = model(x)
    print(y.size())
