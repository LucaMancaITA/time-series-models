import numpy as np

import tensorflow as tf


class TSTransformer(tf.keras.Model):
    def __init__(self, input_dim, d_model, n_heads, num_encoder_layers, dim_feedforward, seq_length):
        super(TSTransformer, self).__init__()
        self.input_projection = tf.keras.layers.Dense(d_model)
        self.positional_encoding = self._generate_positional_encoding(seq_length, d_model)

        self.encoder_layers = [
            tf.keras.layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model) for _ in range(num_encoder_layers)
        ]
        self.feedforward_layers = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(dim_feedforward, activation="relu"),
                tf.keras.layers.Dense(d_model)
            ]) for _ in range(num_encoder_layers)
        ]

        self.fc_out = tf.keras.layers.Dense(1)

    def _generate_positional_encoding(self, seq_length, d_model):
        position = np.arange(seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = np.zeros((seq_length, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return tf.convert_to_tensor(pe, dtype=tf.float32)

    def call(self, x):
        # Input embedding
        x = self.input_projection(x) + self.positional_encoding[tf.newaxis, :x.shape[1], :]
        # Encoder block
        for encoder_layer, feedforward_layer in zip(self.encoder_layers, self.feedforward_layers):
            attn_output = encoder_layer(x, x)
            x = attn_output + x  # Skip connection
            x = feedforward_layer(x) + x  # Skip connection
        x = tf.reduce_mean(x, axis=1)  # Global average pooling
        # Fully connected layer
        x = self.fc_out(x)
        return x


if __name__ == "__main__":

    x = tf.random.uniform(shape=(32, 100, 5))

    model = TSTransformer(
        input_dim=5,
        d_model=32,
        n_heads=1,
        num_encoder_layers=2,
        dim_feedforward=1,
        seq_length=100
    )

    print(x.shape)
    y = model(x)
    print(y.shape)
