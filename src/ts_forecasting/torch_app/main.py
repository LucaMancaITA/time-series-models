import sys
sys.path.append("/Users/lucamanca/Codes/lstm")
sys.path.append("/Users/lucamanca/Codes/lstm/ts_forecasting")
sys.path.append("/Users/lucamanca/Codes/lstm/ts_forecasting/tf_app")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch_app.dataloader import TimeSeriesDataset
from ts_forecasting.torch_app.net.lstm import LSTM
from torch_app.train import train_one_epoch, validate_one_epoch

from utils import df_preprocessing

# Set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparams
batch_size = 16
learning_rate = 0.001
num_epochs = 10

# Tensors
X_train, X_test, y_train, y_test = df_preprocessing()
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Dataset
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

# Dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
model = LSTM(
    input_size=1,
    hidden_size=4,
    num_stacked_layers=1)

# Training loop
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    train_one_epoch(model, train_loader, epoch, loss_function, optimizer, device)
    validate_one_epoch(model, test_loader, loss_function, device)
