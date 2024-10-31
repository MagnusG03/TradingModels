import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Set the date range for the last 700 days
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

# Handle missing values and ensure data types are correct
gold_data.ffill(inplace=True)
gold_data['Close'] = gold_data['Close'].astype(float)

# Calculate moving averages and RSI
gold_data['MA10'] = gold_data['Close'].rolling(window=10).mean()
gold_data['MA50'] = gold_data['Close'].rolling(window=50).mean()

window_length = 14
delta = gold_data['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=window_length).mean()
loss = -delta.where(delta < 0, 0).rolling(window=window_length).mean()
rs = gain / loss
gold_data['RSI'] = 100 - (100 / (1 + rs))

# Fill NaN values
gold_data.fillna(0, inplace=True)

# Features and target variable
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50', 'RSI']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(gold_data[features])
scaled_df = pd.DataFrame(scaled_data, columns=features, index=gold_data.index)
gold_data[features] = scaled_df

X = gold_data[features]
y = gold_data['Close'].shift(-1)
X = X[:-1]
y = y[:-1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Reshape data for LSTM
X_train_lstm = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

# Build and train the LSTM model
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(1, X_train.shape[1])))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_lstm, y_train, epochs=50, batch_size=32, validation_split=0.01, callbacks=[early_stop])

# Make predictions and inverse scaling
predictions = model.predict(X_test_lstm)
col_idx = X_test.columns.get_loc('Close')

predictions_full = X_test.copy()
predictions_full.iloc[:, col_idx] = predictions.flatten()
predictions_inv_full = scaler.inverse_transform(predictions_full)
predictions_inv = predictions_inv_full[:, col_idx]

y_test_full = X_test.copy()
y_test_full.iloc[:, col_idx] = y_test.values
y_test_inv_full = scaler.inverse_transform(y_test_full)
y_test_inv = y_test_inv_full[:, col_idx]

predictions_inv = predictions_inv.reshape(-1)
y_test_inv = y_test_inv.reshape(-1)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
print(f'Root Mean Squared Error: {rmse}')

# Generate buy and sell signals
signals = []
for i in range(len(predictions_inv)):
    predicted_price = predictions_inv[i]
    current_close = X_test['Close'].values[i]
    if predicted_price > current_close:
        signals.append(1)  # Buy
    else:
        signals.append(0)  # Sell

# Backtest the strategy
investment = 10000
positions = 0
portfolio = []

for i in range(len(signals)):
    price = y_test_inv[i]
    if signals[i] == 1 and investment >= price:
        investment -= price
        positions += 1
    elif signals[i] == 0 and positions > 0:
        investment += positions * price
        positions = 0
    portfolio.append(investment + positions * price)

# Buy-and-hold strategy
initial_price = y_test_inv[0]
units_buy_hold = 10000 / initial_price
buy_hold_portfolio = units_buy_hold * y_test_inv

# Plot portfolio values over time
plt.figure(figsize=(12, 6))
plt.plot(portfolio, label='Model Portfolio')
plt.plot(buy_hold_portfolio, label='Buy and Hold Portfolio')
plt.title('Portfolio Value Over Time')
plt.xlabel('Time')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.show()

# Plot difference between strategies
difference = np.array(portfolio) - buy_hold_portfolio
plt.figure(figsize=(12, 6))
plt.plot(difference, label='Profit/Loss vs Buy and Hold')
plt.title('Profit/Loss Compared to Buy and Hold Strategy')
plt.xlabel('Time')
plt.ylabel('Profit/Loss ($)')
plt.legend()
plt.show()
