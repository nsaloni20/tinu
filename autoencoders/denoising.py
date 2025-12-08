"""
Create a Denoising Autoencoder using Keras
"""
# pip install tensorflow numpy matplotlib

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ==========================================
# 1. Prepare Data (Add Noise)
# ==========================================
# Load MNIST data (we don't need labels y_train/y_test for autoencoders)
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape to (28, 28, 1) for Conv2D layers
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# Function to add random noise to images
def add_noise(images, noise_factor=0.5):
    noisy_imgs = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    # Clip values to ensure they stay between 0 and 1
    return np.clip(noisy_imgs, 0., 1.)

x_train_noisy = add_noise(x_train)
x_test_noisy = add_noise(x_test)

# ==========================================
# 2. Build the Autoencoder Model
# ==========================================
input_img = layers.Input(shape=(28, 28, 1))

# --- Encoder (Compress the image) ---
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# --- Decoder (Reconstruct the image) ---
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# Combine Encoder and Decoder
autoencoder = models.Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

# ==========================================
# 3. Train the Model
# ==========================================
# NOTICE: Input is NOISY, Target is CLEAN (x_train)
autoencoder.fit(
    x_train_noisy, x_train,
    epochs=10,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test_noisy, x_test)
)

# ==========================================
# 4. Visualize Results
# ==========================================
decoded_imgs = autoencoder.predict(x_test_noisy)

n = 5  # How many digits we will display
plt.figure(figsize=(10, 4))
for i in range(n):
    # Display Noisy Input
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 2: ax.set_title("Noisy Input")

    # Display Denoised Output
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 2: ax.set_title("Denoised Output")

plt.show()