import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#load dataset
max_words = 10000
max_len = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_words)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

word_index = imdb.get_word_index()

reverse_word_index = {value:key for key,value in word_index.items()}

#convert reviews back to exit
def decode_review(sequence):
    return " ".join(
        reverse_word_index.get(i - 3, "?")
        for i in sequence
        if i >= 3
    )

train_text = [decode_review(x) for x in X_train]
test_text = [decode_review(x) for x in X_test]

#bag of words model
vectorizer = CountVectorizer(max_features=5000)

X_train_bow = vectorizer.fit_transform(train_text)
X_test_bow = vectorizer.transform(test_text)

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train_bow, y_train)

bow_pred = lr.predict(X_test_bow)

bow_acc = accuracy_score(y_test, bow_pred)

print("Bag of Words Accuracy:", bow_acc)

#creating LSTM model
lstm_model = Sequential()

lstm_model.add(Embedding(max_words,128,input_length=max_len))

lstm_model.add(LSTM(128))

lstm_model.add(Dense(1,activation="sigmoid"))

lstm_model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

#training
early = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = lstm_model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    callbacks=[early]
)

#evaluating
loss, lstm_acc = lstm_model.evaluate(X_test,y_test)
print("LSTM Accuracy:",lstm_acc)


#save model
os.makedirs("model",exist_ok=True)
lstm_model.save("model/lstm_model.keras")

#BiLSTM
bilstm = Sequential()

bilstm.add(
    Embedding(max_words,128,input_length=max_len)
)

bilstm.add(
    Bidirectional(
        LSTM(128)
    )
)

bilstm.add(Dense(1,activation="sigmoid"))

bilstm.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

#Training
history2 = bilstm.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    callbacks=[early]
)

#Evaluate
loss,bilstm_acc = bilstm.evaluate(X_test,y_test)
print("BiLSTM Accuracy:",bilstm_acc)

#Save
bilstm.save("model/bilstm_model.keras")

#predictions
pred = (bilstm.predict(X_test)>0.5).astype(int)

report = classification_report(y_test,pred)

print(report)

os.makedirs("outputs",exist_ok=True)

with open("outputs/classification_report.txt","w") as f:
    f.write(report)

#confusion matrix
cm = confusion_matrix(y_test,pred)

disp = ConfusionMatrixDisplay(cm)

disp.plot()

plt.savefig("outputs/confusion_matrix.png")

plt.close()

#Training graph
plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"],label="Train")

plt.plot(history.history["val_accuracy"],label="Validation")

plt.legend()

plt.savefig("outputs/training_history.png")

plt.close()

#Loading GloVe
embedding_dim = 100

embeddings_index = {}

with open(
    "data/glove.6B.100d.txt",
    encoding="utf8"
) as f:

    for line in f:

        values = line.split()

        word = values[0]

        vector = np.asarray(values[1:], dtype="float32")

        embeddings_index[word] = vector

#creating Embedding Matrix
embedding_matrix = np.zeros((max_words, embedding_dim))

for word, index in word_index.items():

    if index >= max_words:

        continue

    vector = embeddings_index.get(word)

    if vector is not None:

        embedding_matrix[index] = vector

#saving matrix
np.save(
    "model/glove_embedding.npy",
    embedding_matrix
)

#creating model
glove_model = Sequential()

glove_model.add(
    Embedding(
        max_words,
        embedding_dim,
        weights=[embedding_matrix],
        trainable=False
    )
)

glove_model.add(LSTM(128))

glove_model.add(Dense(1, activation="sigmoid"))

glove_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

#training
history3 = glove_model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
    callbacks=[early]
)

#evaluating
loss, glove_acc = glove_model.evaluate(
    X_test,
    y_test
)

print("GloVe Accuracy:", glove_acc)

#comparision
with open("outputs/comparison_results.txt", "w") as f:
    f.write(f"Bag of Words : {bow_acc:.4f}\n")
    f.write(f"LSTM : {lstm_acc:.4f}\n")
    f.write(f"BiLSTM : {bilstm_acc:.4f}\n")
    f.write(f"GloVe + LSTM : {glove_acc:.4f}\n")

#saving model
glove_model.save(
    "model/glove_model.keras"
)