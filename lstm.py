import pandas as pd
import yfinance as yf
import datetime

# Set date range for the last 700 days
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=700)

# Download historical gold prices
gold_data = yf.download(
    'GC=F',
    start=start_date.strftime('%Y-%m-%d'),
    end=end_date.strftime('%Y-%m-%d'),
    interval='1h'
)
gold_data.reset_index(inplace=True)

# Handle missing values
gold_data.ffill(inplace=True)  # Updated to avoid FutureWarning

# Ensure 'Close' is a float Series
gold_data['Close'] = gold_data['Close'].astype(float)

# Calculate moving averages (before scaling)
gold_data['MA10'] = gold_data['Close'].rolling(window=10).mean()
gold_data['MA50'] = gold_data['Close'].rolling(window=50).mean()

# Custom RSI calculation
window_length = 14

# Calculate the difference in closing prices
delta = gold_data['Close'].diff()

# Separate gains and losses
gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()

# Calculate the RSI based on the average gain and average loss
rs = gain / loss
gold_data['RSI'] = 100 - (100 / (1 + rs))

# Fill NaN values resulted from calculations
gold_data.fillna(0, inplace=True)

# Features and target variable
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50', 'RSI']

# Scale the features (after computing technical indicators)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(gold_data[features])

# Convert scaled_data back to DataFrame
scaled_df = pd.DataFrame(
    scaled_data,
    columns=features,
    index=gold_data.index
)

# Assign back to gold_data
gold_data[features] = scaled_df

# Prepare X and y
X = gold_data[features]
y = gold_data['Close'].shift(-1)  # Predict the next closing price

# Remove the last row as it doesn't have a target
X = X[:-1]
y = y[:-1]

# Split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Reshape data for LSTM [samples, time steps, features]
X_train_lstm = np.reshape(
    X_train.values, (X_train.shape[0], 1, X_train.shape[1])
)
X_test_lstm = np.reshape(
    X_test.values, (X_test.shape[0], 1, X_test.shape[1])
)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(1, X_train.shape[1])))
model.add(LSTM(50))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_lstm, y_train, epochs=10, batch_size=32)

# Predictions
predictions = model.predict(X_test_lstm)

# Inverse scaling for predictions
predictions_full = np.concatenate(
    (predictions, X_test.iloc[:, 1:].values), axis=1
)
predictions_inv = scaler.inverse_transform(predictions_full)[:, 0]

# Inverse scaling for y_test
y_test_full = np.concatenate(
    (y_test.values.reshape(-1, 1), X_test.iloc[:, 1:].values), axis=1
)
y_test_inv = scaler.inverse_transform(y_test_full)[:, 0]

# Calculate performance metrics
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
print(f'Root Mean Squared Error: {rmse}')

# Inverse scaling adjustments
col_idx = X_test.columns.get_loc('Close')

# Inverse scaling for predictions
predictions_full = X_test.copy()
predictions_full.iloc[:, col_idx] = predictions.flatten()

predictions_inv_full = scaler.inverse_transform(predictions_full)
predictions_inv = predictions_inv_full[:, col_idx]

# Inverse scaling for y_test
y_test_full = X_test.copy()
y_test_full.iloc[:, col_idx] = y_test.values

y_test_inv_full = scaler.inverse_transform(y_test_full)
y_test_inv = y_test_inv_full[:, col_idx]

# Ensure predictions_inv and y_test_inv are 1D arrays
predictions_inv = predictions_inv.reshape(-1)
y_test_inv = y_test_inv.reshape(-1)

# Simple strategy: Buy if the predicted price is higher than the current price
signals = []
for i in range(len(predictions_inv)):
    predicted_price = float(predictions_inv[i])
    current_close = float(X_test['Close'].values[i])

    if predicted_price > current_close:
        signals.append(1)  # Buy
    else:
        signals.append(0)  # Sell

# Backtesting the strategy
investment = 10000  # Starting with $10,000
positions = 0
portfolio = []

for i in range(len(signals)):
    price = y_test_inv[i]  # Use the actual price
    if signals[i] == 1 and investment > price:
        # Buy one unit
        investment -= price
        positions += 1
    elif signals[i] == 0 and positions > 0:
        # Sell all positions
        investment += positions * price
        positions = 0
    portfolio.append(investment + positions * price)

# Plot the portfolio value over time
import matplotlib.pyplot as plt

plt.plot(portfolio)
plt.title('Portfolio Value Over Time')
plt.xlabel('Time')
plt.ylabel('Portfolio Value ($)')
plt.show()
