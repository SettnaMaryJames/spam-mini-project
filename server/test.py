from flask import Flask, request, jsonify
import jsonpickle
from keras.models import load_model
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import numpy as np

lmodel = load_model("my_model.h5")

sentences = ["spam spam spam spam spam"]
print(sentences)
one_hot_sent = [one_hot(i, 10000) for i in sentences]
padsequences = pad_sequences(one_hot_sent, maxlen=80)
label_pred = lmodel.predict(padsequences)
label_pred_ = [np.argmax(i, axis=0) for i in label_pred]
label_pred_
print("result is")
print(label_pred_)
print(type(label_pred_))

keywords = ["spam", "winner"]
stringtocheck = sentences[0]


def contains_keywords(stringtocheck, keywords):
    return any(keyword in stringtocheck for keyword in keywords)


if contains_keywords(stringtocheck, keywords) == True:
    label_pred_[0] = 1

if label_pred_[0] == 0:
    print("returning 0 safe")
else:
    print("return 1 not safe")
