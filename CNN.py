# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 18:12:37 2019

@author: Rémy
"""

import numpy as np
#import pandas as pd
import pickle
import re
#from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, BatchNormalization
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score

def clean_str(string):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

#os.environ['KMP_DUPLICATE_LIB_OK']='True'

MAX_SEQUENCE_LENGTH =300
MAX_NB_WORDS = 25000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.1

def cattoint(v):
    return np.argmax(v,axis=0)


def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

labels={
        "romance" : 2,
        "achievement" : 3,
        "health" : 4,
        "learning" : 5,
        "future/dreams" : 6,
        "art" : 7,
        "dark toughts" : 10,
        "society/politics" : 11,
        "money" : 12,
        "media" : 13,
        "technology" : 14,
        "nature" : 15,
        "religion" : 0,
        "science/history": 8,
        "wisdom" : 1,
        "war" : 9,      
        }
num_labels=len(labels)

rlabels=dict((v, k) for k, v in labels.items())

def loadQuotes(filename):#laod all but three quotes
    return np.genfromtxt(filename,delimiter=";",dtype="str")

q=loadQuotes('newquotes6.csv')
texts=[]
lab=[]
for i in range(len(q)):
    texts.append(clean_str(q[i,0]))
    lab.append(np.zeros(num_labels))
    lab[-1][labels[q[i,2]]]+=1
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

save_obj(tokenizer,"tokenizer")


indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
lab = np.array(lab)[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-2*nb_validation_samples]
y_train = lab[:-2*nb_validation_samples]
x_val = data[-2*nb_validation_samples:-nb_validation_samples]
y_val = lab[-2*nb_validation_samples:-nb_validation_samples]
x_test = data[-nb_validation_samples:]
y_test = lab[-nb_validation_samples:]



def compemb():
    embeddings_index = {}
    f = open('glove.6B.200d.txt',encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    save_obj(embeddings_index,"embdict200")
embeddings_index = load_obj("embdict200")
print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))



embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector



def buildmodel():
    embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,trainable=True)
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_cov1= Conv1D(128, 5, activation='relu',padding='same')(embedded_sequences)
    bn1=BatchNormalization()(l_cov1)
    l_pool1 = MaxPooling1D(5)(bn1)
    l_cov2 = Conv1D(128, 5, activation='relu',padding='same')(l_pool1)
    bn2=BatchNormalization()(l_cov2)
    l_pool2 = MaxPooling1D(5)(bn2)
    l_cov3 = Conv1D(128, 5, activation='relu',padding='same')(l_pool2)
    bn3=BatchNormalization()(l_cov3)
    l_pool3 = MaxPooling1D(12)(bn3)  # global max pooling
    l_flat = Flatten()(l_pool3)
    l_dense = Dense(128, activation='relu')(l_flat)
    d=Dropout(0.4)(l_dense)
    preds = Dense(len(labels), activation='softmax')(d)
    
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=0.0005),
                  metrics=['acc'])
    
    print("Simplified convolutional neural network")
    model.summary()
    cp=ModelCheckpoint('model_cnntest.hdf5',monitor='val_acc',verbose=1,save_best_only=True)
    return model,cp


def train(model,cp):
    history=model.fit(x_train, y_train, validation_data=(x_val, y_val),epochs=10, batch_size=8,callbacks=[cp])
    return history

def plot(history):
    fig1 = plt.figure()
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves :CNN',fontsize=16)
    fig1.savefig('loss_cnn.pdf')
    plt.show()
    fig2=plt.figure()
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves : CNN',fontsize=16)
    fig2.savefig('accuracy_cnn.pdf')
    plt.show()

#model,cp=buildmodel()
#history=train(model,cp)

#plot(history)
#model.save('cnn.h5')

model=load_model('model_cnn.hdf5')
plot_model(model, to_file='model_cnn.pdf')
"""
x_test=np.load("x_test_cnn.npy")
y_test=np.zeros((x_test.shape[0],len(labels)))
for i in range(len(data)):
    for j in range(len(x_test)):
        if (data[i]==x_test[j]).all():
            y_test[j]=lab[i]
y_test=np.array(y_test)
ypred=model.predict(x_test)
Ypred = np.argmax(ypred, axis=1)
Ytrue=np.argmax(y_test, axis=1)
cf=confusion_matrix(Ytrue,Ypred)
np.save("confusion_matrix_cnn.npy",cf)
np.save("x_test_cnn.npy",x_test)


print(accuracy_score(Ytrue,Ypred))
print(cf)
"""



