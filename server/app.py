from keras.models import load_model
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import numpy as np

lmodel = load_model("my_model.h5")


sentences = [
    "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune"
]
one_hot_sent = [one_hot(i, 10000) for i in sentences]
padsequences = pad_sequences(one_hot_sent, maxlen=80)
label_pred = lmodel.predict(padsequences)
label_pred_ = [np.argmax(i, axis=0) for i in label_pred]
label_pred_
print(label_pred_)
