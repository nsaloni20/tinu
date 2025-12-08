import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------------------
# 1. Load Reuters dataset
# -------------------------------------
from tensorflow.keras.datasets import reuters

num_words = 10000
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=num_words)

print("Training samples:", len(x_train))
print("Test samples:", len(x_test))
print("Number of classes:", np.max(y_train) + 1)

# -------------------------------------
# 2. Vectorize the sequences (One-hot)
# -------------------------------------
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, seq in enumerate(sequences):
        results[i, seq] = 1.0      # float (safer & cleaner)
    return results

x_train_vec = vectorize_sequences(x_train)
x_test_vec  = vectorize_sequences(x_test)

# -------------------------------------
# 3. One-hot encode labels
# -------------------------------------
num_classes = np.max(y_train) + 1

y_train_oh = keras.utils.to_categorical(y_train, num_classes)
y_test_oh  = keras.utils.to_categorical(y_test, num_classes)

# -------------------------------------
# 4. Build Deep Neural Network
# -------------------------------------
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10000,),
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.5),

    layers.Dense(32, activation='relu',
                 kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation='softmax')
])

model.summary()

# -------------------------------------
# 5. Compile model
# -------------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping to reduce overfitting
callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

# -------------------------------------
# 6. Train model
# -------------------------------------
history = model.fit(
    x_train_vec, y_train_oh,
    epochs=20,
    batch_size=512,
    validation_split=0.2,
    callbacks=[callback],
    verbose=1
)

# -------------------------------------
# 7. Evaluate model
# -------------------------------------
test_loss, test_acc = model.evaluate(x_test_vec, y_test_oh)
print("\nTest Accuracy:", test_acc)

# -------------------------------------
# 8. Predictions
# -------------------------------------
predictions = model.predict(x_test_vec[:5])
predicted_classes = np.argmax(predictions, axis=1)

print("\nPredicted classes:", predicted_classes)
print("Actual classes:", y_test[:5])
