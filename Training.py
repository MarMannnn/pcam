import DataLoader
import Model
import numpy as np
import tensorflow as tf
import h5py
from tensorflow import keras
from keras import layers
from keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping as es
import tensorflow as tf

#dataset
x_train = h5py.File('train_x.h5', 'r')['x']
y_train = h5py.File('train_y.h5', 'r')['y']

x_test = h5py.File('test_x.h5', 'r')['x']
y_test = h5py.File('test_y.h5', 'r')['y']

x_valid = h5py.File('valid_x.h5', 'r')['x']
y_valid = h5py.File('valid_y.h5', 'r')['y']


    # 
x_train = np.asarray(x_train).astype('float32')
x_test = np.asarray(x_test).astype('float32')
x_valid = np.asarray(x_valid).astype('float32')

y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')
y_valid = np.asarray(y_valid).astype('float32')  

#model and params
model = Model.CreateModel()
loss, opt, callback = Model.HyperPar()

# compile 
model.compile(optimizer=opt,loss = loss,metrics = ["accuracy"])
# train
history = model.fit(x_train, y_train, validation_data= (x_valid,y_valid) , verbose= 1,batch_size=50, epochs = 150, callbacks= callback )