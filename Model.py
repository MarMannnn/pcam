def CreateModel():
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    from keras import Sequential
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(96, 96, 3)),
        layers.RandomFlip(),

        layers.Conv2D(16, 3, padding='same', strides=(2,2),activation='relu', kernel_initializer='glorot_uniform'),
        layers.Conv2D(16, 3, padding='same', strides=(2,2),activation='relu', kernel_initializer='glorot_uniform'),
        layers.MaxPooling2D(),
        layers.Dropout(0.1),
        
        layers.Conv2D(32, 3, padding='same', strides=(2,2),activation='relu', kernel_initializer='glorot_uniform'),
        layers.Conv2D(32, 3, padding='same', strides=(2,2),activation='relu', kernel_initializer='glorot_uniform'),
        layers.MaxPooling2D(),


        layers.Conv2D(64, 3, padding='same', strides=(2,2),activation='relu', kernel_initializer='glorot_uniform'),
        layers.Conv2D(64, 3, padding='same', strides=(2,2),activation='relu', kernel_initializer='glorot_uniform'),
        layers.MaxPooling2D(),
        
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),

        layers.Dense(1, activation='softmax')
    ])
    return model

def HyperPar():
    from tensorflow.keras.callbacks import EarlyStopping as es
    import tensorflow as tf
    loss = tf.losses.binary_crossentropy()
    opt = tf.optimizers.Adam(learning_rate = 0.001)
    callback = es(monitor='loss', patience = 20)
    return loss, opt, callback
    