import tensorflow as tf


class LSTM(tf.keras.Model):
    def __init__(self, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm_layers = [
            tf.keras.layers.LSTM(
                units=hidden_size,
                return_sequences=(i < num_stacked_layers - 1),
                return_state=False,
                stateful=False,
                recurrent_initializer='glorot_uniform')
            for i in range(self.num_stacked_layers)
        ]

        self.fc = tf.keras.layers.Dense(1)

    def call(self, x):
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x)
        out = self.fc(x)
        return out


if __name__ == "__main__":

    x = tf.random.uniform(shape=(32, 100, 5))

    model = LSTM(
        hidden_size=4,
        num_stacked_layers=1)

    print(x.shape)
    y = model(x)
    print(y.shape)
