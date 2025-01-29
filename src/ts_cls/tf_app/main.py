import sys
sys.path.append("/Users/lucamanca/Codes/lstm")
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from ts_cls.tf_app.net import FakeNewsModel


# Read the df
df = pd.read_csv("data/ts-cls/train.csv")
df = df.dropna()
print(df.head())
print(df.shape)

X = df.drop("label", axis=1)
y = df["label"]

# Vocab size
vocab_size = 5000
sent_length = 20
embedding_dim = 40

# OneHot representation
messages = X.copy()
messages.reset_index(inplace=True)

# Dataset preprocessing
# 1. Keep only letters from a to z
# 2. Lower each word
# 3. Remove stop word
# 4. Stemming: to reduce inflectional forms to their stem or root form
# 5. Tokenizer (one hot encoding)
# 6. Padding

nltk.download("stopwords")
ps = PorterStemmer()
corpus = []
for i in tqdm(range(0, len(messages))):
    review = re.sub("[^a-zA-Z]", " ", messages["title"][i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words("english")]
    review = " ".join(review)
    corpus.append(review)

one_hot_reprs = [one_hot(words, vocab_size) for words in corpus]
print(one_hot_reprs)

embedded_docs = pad_sequences(one_hot_reprs, padding="pre", maxlen=sent_length)
print(embedded_docs)

# Model
model = FakeNewsModel(vocab_size, sent_length, embedding_dim)

# Train and test dataset
X = np.array(embedded_docs)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, y_train.shape)

# Training
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.BinaryCrossentropy()
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics="accuracy")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)

# Evaluation
y_pred = model.predict(X_test)
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
