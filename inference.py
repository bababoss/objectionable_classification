
import numpy as np # linear algebra
import pandas as pd 

# DataSET

from subprocess import check_output
print(check_output(["ls", "/home/cogknit/project/hatespeech/input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

max_features = 20000
maxlen = 100

batch_size = 1
epochs = 1
file_path="/home/cogknit/project/objectionable_classification/weights_base.best.hdf5"

def get_model():
    embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def prediction_run(input_text):
    """
    # Prediction
    """
    model = get_model()

    list_sentences_train=np.array([input_text])
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    X_data = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)

    model.load_weights(file_path)

    pred = model.predict(X_data)


    pridiction=pred[0]
    result = dict(zip(list_classes, pridiction))
    return result