import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from tensorflow.keras.preprocessing.text import one_hot

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from ts_cls.torch_app.dataloader import CustomDataset
from ts_cls.torch_app.net import FakeNewsModel
from ts_cls.torch_app.train import train_val_loop


# Device
device = "gpu" if torch.cuda.is_available() else "cpu"

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
one_hot_tensors = [torch.tensor(seq) for seq in one_hot_reprs]

embedded_docs = pad_sequence(one_hot_tensors, padding_side='left', batch_first=True)
print(embedded_docs)

# Train and test dataset
X = embedded_docs.detach()
y = torch.tensor(y, dtype=torch.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, y_train.shape)
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Model
model = FakeNewsModel(
    vocab_size=vocab_size,
    seq_len=sent_length,
    embedding_dim=embedding_dim
)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
train_val_loop(model, train_dataloader, test_dataloader, optimizer,
               criterion, max_epochs=10, device=device)

# Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
y_pred = (y_pred.squeeze() > 0.5).int()  # Convert to binary 0 or 1
y_pred = y_pred.cpu()
y_test = y_test.cpu()
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

