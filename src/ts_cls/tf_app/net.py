import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Model params
vocab_size = 5000
embedding_dim = 40
seq_len = 20

# Sequential model
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=seq_len))
model.add(LSTM(100))
model.add(Dense(1, activation="sigmoid"))
print(model.summary())

# Custom model
class FakeNewsModel(tf.keras.Model):
    def __init__(self, vocab_size, seq_len, embedding_vector_features):
        super().__init__()
        self.emb_features = embedding_vector_features
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embd = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.emb_features
        )
        self.lstm = LSTM(
            units=100,
            activation="tanh",
            return_sequences=False
        )
        self.fc = Dense(
            units=1,
            activation="sigmoid"
        )

    def call(self, x):
        x = self.embd(x)
        x = self.lstm(x)
        out = self.fc(x)
        return out


if __name__ == "__main__":
    x = tf.random.uniform(shape=(200, 20))
    model = FakeNewsModel(seq_len, embedding_dim)
    y = model(x)
    print(y.shape)
