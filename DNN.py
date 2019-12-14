# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 22:42:05 2019

@author: Rémy
"""
import numpy as np
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.optimizers as optimizers
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from tensorflow.keras.utils import  plot_model

x_train=np.load("x_train.npy")
#x_train=x_train/np.max(x_train,axis=0)
y_train=np.load("y_train.npy")
n=len(x_train)//10
x_test,y_test=x_train[-n:],y_train[-n:]
x_val,y_val=x_train[-2*n:-n],y_train[-2*n:-n]
x_train,y_train=x_train[:-2*n],y_train[:-2*n]
d=len(x_train[0])
o=len(y_train[0])
m=(d+o)//2

model=Sequential()
model.add(Dense(6*o,input_dim=d,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(4*o,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(2*o,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(o,activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

cp=ModelCheckpoint('model_dense_softmax.hdf5',monitor='val_acc',verbose=1,save_best_only=True)

history=model.fit(x_train, y_train,
          epochs=10,
          batch_size=16,
          validation_data=(x_val,y_val),
          callbacks=[cp])

#score = model.evaluate(x_test, y_test, batch_size=32)

def predic(probabilities):
    y=np.zeros(len(probabilities))
    y[np.argmax(probabilities)]+=1
    return y

model.save('softmaxdense_NN.h5')
np.save("x_test_dense.npy",x_test)


def plot(history):
    fig1 = plt.figure()
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves :Dense',fontsize=16)
    fig1.savefig('loss_dense.pdf')
    plt.show()
    fig2=plt.figure()
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves : Dense',fontsize=16)
    fig2.savefig('accuracy_dense.pdf')
    plt.show()
plot(history)

model=load_model('model_dense_softmax3.hdf5')
plot_model(model, to_file='model_dnn.pdf')
"""ypred=model.predict(x_test)
Ypred = np.argmax(ypred, axis=1)
Ytrue=np.argmax(y_test, axis=1)
cf=confusion_matrix(Ytrue,Ypred)
np.save("confusion_matrix_dense.npy",cf)
np.save("x_test_dense.npy",x_test)
print(accuracy_score(Ytrue,Ypred))

print(cf)
"""