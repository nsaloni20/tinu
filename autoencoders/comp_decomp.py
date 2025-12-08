"""
PRACTICAL: Image Compression and Decompression using Autoencoders
"""
# pip install tensorflow numpy matplotlib

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Data Preparation
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize pixel values (0 to 1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape to (28, 28, 1) to match Conv2D requirements
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# 2. Build the Model (Encoder-Decoder)
input_img = layers.Input(shape=(28, 28, 1))

#   ENCODER (Compression)
#   We reduce filters to 8 to force compression
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

#   DECODER (Decompression)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Combine into Autoencoder Model
autoencoder = models.Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

# 3. Train the Model
# Note: Input is x_train, Target is ALSO x_train
autoencoder.fit(
    x_train, x_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, x_test)
)

# 4. Visualize Results
decoded_imgs = autoencoder.predict(x_test)

n = 5  # Number of images to display
plt.figure(figsize=(10, 4))
for i in range(n):
    # Display Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.set_title("Original")
    ax.axis('off')

    # Display Decompressed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.set_title("Decompressed")
    ax.axis('off')

plt.show()