"""
Classify MNIST dataset using any pertained model
"""
#pip install tensorflow numpy

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)





# Ensure 'tf' is defined by running the import cell (BklCHBJ18IBL) first.
x_train = tf.image.resize(x_train, [32, 32])
x_test  = tf.image.resize(x_test, [32, 32])


x_train = tf.image.grayscale_to_rgb(x_train)
x_test  = tf.image.grayscale_to_rgb(x_test)


x_train = x_train / 255.0
x_test  = x_test / 255.0

y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test, 10)


base_model = keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)

base_model.trainable = False  # Freeze pretrained layers

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')   # MNIST has 10 classes
])

model.summary()


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Ensure model is defined and compiled by running preceding cells first.
history = model.fit(
    x_train, y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.1
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)