import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load Dataset from Online Source (Teacher Requirement)

url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

print("Dataset shape:", data.shape)
print(data.head())

# 2. Split Features and Target
X = data.drop("medv", axis=1).values    # features
y = data["medv"].values                 # target (price)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled  = scaler.transform(x_test)

# 4. Build Deep Neural Network
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # regression output
])

model.summary()

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# 5. Train the Model
history = model.fit(
    x_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=1
)

# 6. Evaluate
test_loss, test_mae = model.evaluate(x_test_scaled, y_test, verbose=0)
print(f"Test MSE: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")
