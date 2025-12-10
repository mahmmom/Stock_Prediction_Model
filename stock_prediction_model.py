import os
# Suppress TensorFlow warnings - MUST be set before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # Changed from StandardScaler - better for LSTM
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Read the stock market data
data = pd.read_csv('TSLA.csv')
# Display and quick summary of the data
# print(data.head())
# print(data.info())
# print(data.describe())

# Initial Data Visualization
# Plot 1 - Open and Close Prices over Time
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Open'], label="Open", color='blue')
plt.plot(data['Date'], data['Close'], label="Close", color='red')  # Fixed: was plotting Open twice
plt.title('Open and Close Prices over Time')
plt.legend()
plt.close()  # Close this figure to prevent it from showing

# Plot 2 - Volume Traded over Time (check for outliers)
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Volume'], label="Volume", color='purple')
plt.title('Volume Traded over Time')
plt.close()  # Close this figure to prevent it from showing

# Drop non-numeric Columns for correlation analysis
numeric_data = data.select_dtypes(include=["int64", "float64"])

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title('Correlation Heatmap')
# plt.show()
plt.close()

# Convert the data into Datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Prepare for the LSTM Model (sequential Data)
stock_close = data.filter(['Close'])
dataset = stock_close.values
training_data_len = int(len(dataset) * 0.80)  # Changed to 80/20 split (from 95/5)

# Preprocessing Stages - FIX DATA LEAKAGE: Fit scaler only on training data
scaler = MinMaxScaler(feature_range=(0, 1))  # Changed to MinMaxScaler - better for LSTM
training_data = dataset[:training_data_len]  # Get training data first
scaled_training = scaler.fit_transform(training_data)  # Fit only on training data

# Transform test data using the same scaler
test_raw = dataset[training_data_len:]
scaled_test = scaler.transform(test_raw)

X_train, Y_train = [], []

# Create a sliding window of 60 time-steps
for i in range(60, len(scaled_training)):
    X_train.append(scaled_training[i-60:i, 0])
    Y_train.append(scaled_training[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM Model
model = keras.Sequential()

# First layer
model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))

# second layer
model.add(keras.layers.LSTM(32, return_sequences=False))

# third layer (Dense)
model.add(keras.layers.Dense(128, activation='relu'))

# 4th layer (Dropout for overfitting)
model.add(keras.layers.Dropout(0.5))

# Output layer
model.add(keras.layers.Dense(1))

model.summary() # Print the model summary

# Compile the model - Changed to MSE (better for regression)
model.compile(optimizer='adam', 
              loss='mse',  # Changed from 'mae' to 'mse' - penalizes large errors more
              metrics=[keras.metrics.RootMeanSquaredError()])

# Add early stopping to prevent overfitting
early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# Train the model - Increased epochs with early stopping
print(f"\n📊 Dataset: {len(dataset)} samples")
print(f"Training: {training_data_len} samples (80%)")
print(f"Test: {len(dataset) - training_data_len} samples (20%)\n")
print("🚀 Training the model...\n")

training = model.fit(X_train, Y_train, 
                     batch_size=32, 
                     epochs=50,
                     callbacks=[early_stop],
                     verbose=1)

print(f"\n✅ Training completed! Total epochs: {len(training.history['loss'])}")

# Prep for test - combine training and test scaled data for windowing
scaled_data = np.vstack([scaled_training, scaled_test])
test_data = scaled_data[training_data_len - 60:]
X_test, Y_test = [], dataset[training_data_len:]

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Get the model's predicted price values
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Calculate and print evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(Y_test, predictions)
rmse = np.sqrt(mean_squared_error(Y_test, predictions))
mape = np.mean(np.abs((Y_test - predictions) / Y_test)) * 100
r2 = r2_score(Y_test, predictions)

print("\n" + "="*50)
print("📊 MODEL EVALUATION METRICS")
print("="*50)
print(f"Mean Absolute Error (MAE):  ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"R² Score: {r2:.4f}")
print("="*50 + "\n")

# Plotting data
train = data[:training_data_len]
test = data[training_data_len:]

test = test.copy()

test['Predictions'] = predictions

plt.figure(figsize=(12, 8))
plt.plot(train['Date'], train['Close'], label='Train Actual', color='blue')
plt.plot(test['Date'], test['Close'], label='Test Actual', color='orange')
plt.plot(test['Date'], test['Predictions'], label='Test Predictions', color='red')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Price Prediction using LSTM')
plt.legend()
plt.show()
