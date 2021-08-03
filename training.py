import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Lemmatization of the words used in the training data

Word_lemmitizer = WordNetLemmatizer()
intentions = json.loads(open('intents.json').read())

words = []
classes = []
docs = []
skip_chars = ["?", "!", "/", ",", "."]

for intent in intentions['intents']:
    for pattern in intent['patterns']:
        list_words = nltk.word_tokenize(pattern)
        words.extend(list_words)
        docs.append((list_words, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [Word_lemmitizer.lemmatize(w) for w in words if w not in skip_chars]
words = sorted(set(words))
classes = sorted(set(classes))


pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
template = [0] * len(classes)

for d in docs:
    store = []
    w_p = d[0]
    w_p = [Word_lemmitizer.lemmatize(w.lower()) for w in w_p]
    for wo in words:
        store.append(1) if wo in w_p else store.append(0)

    row = list(template)
    row[classes.index(d[1])] = 1
    training.append([store, row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# ----Neural net model --------------------------------

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)

