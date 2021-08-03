import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')


# ------ Functions ------------------------------------------


def clean(line):
    wrds = nltk.word_tokenize(line)
    wrds = [lemmatizer.lemmatize(word.lower()) for word in wrds]
    return wrds

def flagging(line):
    wrds = clean(line)
    store = [0] * len(words)
    for w in wrds:
        for i, word in enumerate(words):
            if str(word) == str(w):
                store[i] = 1
    return np.array(store)

def predict_class(line):
    store = flagging(line)
    result = model.predict(np.array([store]))[0]
    er_threshold = 0.25
    result = [[i, r] for i, r in enumerate(result) if r > er_threshold]

    result.sort(key=lambda x: x[1], reverse=True)
    output_list = []
    for r in result:
        output_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return output_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list = intents_json['intents']
    for i in list:
        if i['tag'] == tag:
            answer = random.choice(i['responses'])
            break
    return answer

