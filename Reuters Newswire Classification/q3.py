"""
Deep Neural Network for Reuters Newswire Classification (Multi-Class)
"""
# pip install tensorflow numpy matplotlib

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import reuters
from tensorflow.keras import models, layers, utils
import matplotlib.pyplot as plt

# 1. Load Data
# We restrict the data to the top 10,000 most frequently occurring words.
# Rare words are discarded to manage vector size.
MAX_WORDS = 10000
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=MAX_WORDS)

# ==========================================
# 2. Data Preprocessing (Vectorization)
# ==========================================
# Neural networks cannot ingest lists of integers (which vary in length).
# We must convert them into tensors (fixed size vectors of 0s and 1s).
# This process is often called "Multi-hot encoding" or "Bag of Words".
def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # Set specific indices to 1 if the word exists
    return results

# Convert training and test data into 10k-dimensional vectors
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# One-hot encode the labels (e.g., category 3 becomes [0, 0, 0, 1, ...])
# We use 46 because there are 46 possible topics in Reuters.
y_train = utils.to_categorical(train_labels)
y_test = utils.to_categorical(test_labels)

# Create a validation set (first 1000 samples) to monitor training performance
# This helps us detect overfitting during the training process.
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

# ==========================================
# 3. Build the Model
# ==========================================
model = models.Sequential([
    # Input layer: Must match our vocab size (10,000).
    # We use 64 hidden units (neurons) for the network to learn complex patterns.
    layers.Dense(64, activation='relu', input_shape=(MAX_WORDS,)),
    
    # Second hidden layer for added capacity.
    layers.Dense(64, activation='relu'),
    
    # Output layer: 46 units for the 46 different topics.
    # 'softmax' activation outputs a probability distribution summing to 1.
    # e.g., "70% Topic A, 20% Topic B, 10% Topic C"
    layers.Dense(46, activation='softmax')
])

model.summary()

# ==========================================
# 4. Compile and Train
# ==========================================
model.compile(
    optimizer='rmsprop',
    # 'categorical_crossentropy' is the standard loss for multi-class classification
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
# We use 9 epochs because this specific dataset tends to overfit after that.
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=9,
    batch_size=512,
    validation_data=(x_val, y_val)
)

# ==========================================
# 5. Evaluate and Visualize
# ==========================================
results = model.evaluate(x_test, y_test)
print(f"Test Loss: {results[0]}, Test Accuracy: {results[1]}")

# Plotting Loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting Accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.subplot(1, 2, 2)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()





"""
"The objective of this program is to build a Deep Neural Network that classifies news wires from the Reuters dataset into 46 mutually 
exclusive topics, such as 'grain,' 'shipping,' or 'earnings.' This is a classic example of a Single-Label, Multi-Class Classification 
problem.

The first challenge we face is that neural networks cannot understand raw text or lists of integers with variable lengths. To solve 
this, I performed Data Vectorization (often called Multi-hot encoding). I restricted the vocabulary to the top 10,000 most frequently 
occurring words to keep the input manageable. I then wrote a function to convert each news wire into a fixed-size vector of 10,000 
dimensions. If a specific word appears in the article, its corresponding index in the vector is set to 1; otherwise, it is 0. 
Similarly, the labels were One-Hot Encoded (categorical encoding) so that the network could predict a vector for the 46 classes.

For the model architecture, I used a Sequential model with three Dense layers. The first two hidden layers have 64 neurons each with 
ReLU activation. I chose 64 units to give the model enough 'capacity' to learn the complex relationships between words and topics. The 
final output layer is the most critical part: it consists of 46 neurons (matching the 46 topics) and uses the Softmax activation 
function. This ensures the model outputs a probability distribution where the sum of all 46 outputs equals 1, allowing us to pick the 
topic with the highest probability.

Finally, I compiled the model using the RMSprop optimizer and Categorical Crossentropy loss function, which is the standard math used 
for multi-class classification. I set aside 1,000 samples as a validation set to monitor the training. I specifically stopped training 
at 9 epochs because, typically with this dataset, the model begins to overfit (memorizing the training data instead of learning) after 
that point, which results in worse performance on new data."
"""