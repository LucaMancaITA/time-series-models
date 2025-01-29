import tensorflow as tf

from ts_forecasting.tf_app.dataloader import TimeSeriesDataset
from ts_forecasting.tf_app.net.lstm import LSTM
from ts_forecasting.tf_app.train import train_one_epoch, validate_one_epoch

from ts_forecasting.utils import df_preprocessing


# Hyperparams
batch_size = 16
learning_rate = 0.001
num_epochs = 10

# Tensors
X_train, X_test, y_train, y_test = df_preprocessing()
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Dataset
train_dataset = TimeSeriesDataset(X_train, y_train, batch_size)
test_dataset = TimeSeriesDataset(X_test, y_test, batch_size)

# Model
model = LSTM(
    hidden_size=4,
    num_stacked_layers=1)

# Training loop
loss_function = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
for epoch in range(num_epochs):
    train_one_epoch(model, train_dataset, epoch, loss_function, optimizer)
    validate_one_epoch(model, test_dataset, loss_function)
