"""
Implement CNN for classifying MNIST dataset
"""
# pip install notebook tensorflow numpy

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)


# Reshape to (28, 28, 1) for CNN input
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test  = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test, 10)


model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)


test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)


preds = model.predict(x_test[:5])
print(np.argmax(preds, axis=1))
print("Actual:", np.argmax(y_test[:5], axis=1))