import DataLoader
import Model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping as es
import tensorflow as tf

#dataset
x_train, y_train, x_test, y_test, x_valid, y_valid = DataLoader.DataLoader()
y_train = np.reshape(y_train, (len(y_train), 1))
y_test = np.reshape(y_test, (len(y_test), 1))
y_valid = np.reshape(y_valid, (len(y_valid), 1))

#model and params
model = Model.CreateModel()
loss, opt, callback = Model.HyperPar()

# compile 
model.compile(optimizer=opt,loss = loss,metrics = ["accuracy"])
# train
history = model.fit(x_train, y_train, validation_data= (x_valid,y_valid) , verbose= 1,batch_size=50, epochs = 150, callbacks= callback )