def CreateModel():
    import keras
    base_model = keras.applications.Xception(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(96, 96, 3),
    include_top=False) 

    #freeze the train of a base_model (pre-trained)
    base_model.trainable = False

    # create a model 
    inputs = keras.Input(shape=(96, 96, 3))
    scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1) # scaling all the values [-1, 1]
    x = scale_layer(inputs)
    x = base_model(x, training=False)
    x = keras.layers.Conv2D(16, 3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(16, 3, padding='same', activation='relu')(x)
    x = keras.layers.Conv2D(16, 3, padding='same', activation='relu')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(1)(x)
    outputs = keras.layers.Dropout(0.2)(x) 
    model = keras.Model(inputs, outputs)
    return model

def HyperPar():
    from tensorflow.keras.callbacks import EarlyStopping as es
    import tensorflow as tf
    loss = tf.losses.binary_crossentropy(from_logits=True)
    opt = tf.optimizers.Adam(learning_rate = 0.001)
    callback = es(monitor='loss', patience = 20)
    return loss, opt, callback
    