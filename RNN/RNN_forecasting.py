"""
Create a model for time-series forecasting using RNN (SimpleRNN)
"""
# pip install tensorflow numpy matplotlib

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt

# 1. Generate Synthetic Data (Sine Wave)
t = np.linspace(0, 100, 1000)
data = np.sin(t)

plt.figure(figsize=(10,4))
plt.plot(data)
plt.title("Raw Data (Sine Wave)")
plt.show()

# 2. Data Preprocessing (Sliding Window)
def create_dataset(dataset, look_back=10):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back)])
        y.append(dataset[i + look_back])
    return np.array(X), np.array(y)

look_back = 20
X, y = create_dataset(data, look_back)

# Reshape input to be [samples, time_steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split Data
train_size = int(len(X) * 0.8)
x_train, x_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

# 3. Build the SimpleRNN Model
# The only difference from LSTM is using 'SimpleRNN' layer here
model = Sequential([
    SimpleRNN(50, activation='tanh', input_shape=(look_back, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# 4. Train the Model
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(x_test, y_test),
    verbose=1
)

# 5. Make Predictions & Visualize
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

plt.figure(figsize=(12,6))
plt.plot(np.arange(len(data)), data, label='Original Data', alpha=0.6)

# Plot Training
train_plot = np.empty_like(data)
train_plot[:] = np.nan
train_plot[look_back:len(train_predict)+look_back] = train_predict.flatten()
plt.plot(train_plot, label='Training Prediction')

# Plot Testing
test_plot = np.empty_like(data)
test_plot[:] = np.nan
# Align test predictions (indices calculation)
test_plot[len(train_predict)+(look_back):len(data)] = test_predict.flatten()
plt.plot(test_plot, label='Testing Prediction', color='red')

plt.legend()
plt.title("Time-Series Prediction: SimpleRNN vs Actual")
plt.show()