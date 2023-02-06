import DataLoader
import Model

# prepare the dataset
<<<<<<< HEAD
x_train, y_train, x_test, y_test, x_valid, y_valid = DataLoader.DataLoader()

# get the model
model = Model.CreateModel()

# get the hyperparms for the training
loss, opt, callback = Model.HyperPar()

#train
model.compile(optimizer=opt,
              loss=loss, #true because values [-1,1]
              metrics=[keras.metrics.BinaryAccuracy()])
model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid), verbose=1)
=======

# get the model

# get the hyperparms for the training
>>>>>>> 476fbe1058e99b2025e44a5c8cdcaff5361ba12a
