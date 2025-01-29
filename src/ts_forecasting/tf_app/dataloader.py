import tensorflow as tf
import numpy as np


class TimeSeriesDataset(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.indices = np.arange(self.X.shape[0])

    def __len__(self):
        return (len(self.X) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, i):
        start_idx = i * self.batch_size
        end_idx = min((i + 1) * self.batch_size, len(self.X))
        batch_indices = tf.convert_to_tensor(self.indices[start_idx:end_idx], dtype=tf.int32)
        x_batch = tf.gather(self.X, batch_indices)
        y_batch = tf.gather(self.y, batch_indices)
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
