import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warning messages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Ensure reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Set the date ranges
end_date = datetime.datetime.today()
start_date_daily = end_date - datetime.timedelta(days=365 * 23)  # 23 years
start_date_hourly = end_date - datetime.timedelta(days=719)      # 719 days
start_date_2min = end_date - datetime.timedelta(days=59)         # 59 days

# Download daily data
daily_data = yf.download(
    'CL=F',
    start=start_date_daily.strftime('%Y-%m-%d'),
    end=end_date.strftime('%Y-%m-%d'),
    interval='1d',
    progress=False
)
# Flatten MultiIndex columns if present in daily_data
if isinstance(daily_data.columns, pd.MultiIndex):
    daily_data.columns = daily_data.columns.get_level_values(0)
daily_data.reset_index(inplace=True)
daily_data.set_index('Date', inplace=True)

# Download hourly data
hourly_data = yf.download(
    'CL=F',
    start=start_date_hourly.strftime('%Y-%m-%d'),
    end=end_date.strftime('%Y-%m-%d'),
    interval='1h',
    progress=False
)
# Flatten MultiIndex columns if present
if isinstance(hourly_data.columns, pd.MultiIndex):
    hourly_data.columns = hourly_data.columns.get_level_values(0)
hourly_data.reset_index(inplace=True)
if 'Datetime' in hourly_data.columns:
    hourly_data.set_index('Datetime', inplace=True)
elif 'Date' in hourly_data.columns:
    hourly_data.set_index('Date', inplace=True)
else:
    print("Datetime column not found in hourly_data.")

# Download 2-minute data
data_2min = yf.download(
    'CL=F',
    start=start_date_2min.strftime('%Y-%m-%d'),
    end=end_date.strftime('%Y-%m-%d'),
    interval='2m',
    progress=False
)
# Flatten MultiIndex columns if present
if isinstance(data_2min.columns, pd.MultiIndex):
    data_2min.columns = data_2min.columns.get_level_values(0)
data_2min.reset_index(inplace=True)
if 'Datetime' in data_2min.columns:
    data_2min.set_index('Datetime', inplace=True)
elif 'Date' in data_2min.columns:
    data_2min.set_index('Date', inplace=True)
else:
    print("Datetime column not found in data_2min.")

# Function to aggregate higher-frequency data
def aggregate_data(high_freq_data, freq):
    available_cols = high_freq_data.columns.tolist()
    agg_dict = {}
    if 'Open' in available_cols:
        agg_dict['Open'] = 'first'
    if 'High' in available_cols:
        agg_dict['High'] = 'max'
    if 'Low' in available_cols:
        agg_dict['Low'] = 'min'
    if 'Close' in available_cols:
        agg_dict['Close'] = 'last'
    if 'Volume' in available_cols:
        agg_dict['Volume'] = 'sum'

    if not agg_dict:
        print(f"No columns to aggregate in data with frequency {freq}")
        return pd.DataFrame()

    agg_data = high_freq_data.resample(freq).agg(agg_dict)
    agg_data.dropna(subset=['Close'], inplace=True)
    return agg_data

# Aggregate hourly data
agg_hourly = aggregate_data(hourly_data, 'D')
agg_hourly = agg_hourly.loc[start_date_hourly.strftime('%Y-%m-%d'):]

# Aggregate 2-minute data
agg_2min = aggregate_data(data_2min, 'D')
agg_2min = agg_2min.loc[start_date_2min.strftime('%Y-%m-%d'):]

# Calculate volatility from higher-frequency data
def calculate_volatility(high_freq_data):
    if 'Close' not in high_freq_data.columns:
        print("No 'Close' column available for volatility calculation.")
        return pd.Series(dtype=float)
    returns = high_freq_data['Close'].pct_change()
    volatility = returns.resample('D').std()
    return volatility

# Calculate volatility for hourly and 2-minute data
volatility_hourly = calculate_volatility(hourly_data).rename('Volatility_hourly')
volatility_2min = calculate_volatility(data_2min).rename('Volatility_2min')

# Convert volatility Series to DataFrame and reset index
volatility_hourly = volatility_hourly.to_frame().reset_index()
volatility_hourly.rename(columns={'Datetime': 'Date'}, inplace=True)

volatility_2min = volatility_2min.to_frame().reset_index()
volatility_2min.rename(columns={'Datetime': 'Date'}, inplace=True)

# Reset index of daily_data for merging
daily_data.reset_index(inplace=True)

# Merge higher-frequency features with daily data
daily_data = daily_data.merge(volatility_hourly, on='Date', how='left')
daily_data = daily_data.merge(volatility_2min, on='Date', how='left')

# Set 'Date' back as index
daily_data.set_index('Date', inplace=True)

# Fill missing volatility values
daily_data['Volatility_hourly'] = daily_data['Volatility_hourly'].ffill()
daily_data['Volatility_2min'] = daily_data['Volatility_2min'].ffill()

# **Handle Negative 'Close' Values with Forward Fill**
# 1. Identify negative or zero 'Close' values
negative_close_mask = daily_data['Close'] <= 0
print(f"Number of days with negative or zero 'Close': {negative_close_mask.sum()}")

# 2. Replace negative 'Close' values with NaN to use ffill
daily_data.loc[negative_close_mask, 'Close'] = np.nan

# 3. Use forward fill and backfill to replace NaN with the previous valid 'Close' value
daily_data['Close'] = daily_data['Close'].ffill().bfill()

# Fill other missing values
daily_data = daily_data.ffill()
daily_data['Close'] = daily_data['Close'].astype(float)

# Calculate moving averages and RSI
daily_data['MA10'] = daily_data['Close'].rolling(window=10).mean()
daily_data['MA50'] = daily_data['Close'].rolling(window=50).mean()

window_length = 14
delta = daily_data['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=window_length).mean()
loss = -delta.where(delta < 0, 0).rolling(window=window_length).mean()
rs = gain / loss
daily_data['RSI'] = 100 - (100 / (1 + rs))

daily_data.fillna(0, inplace=True)

# Features and target variable
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50', 'RSI', 'Volatility_hourly', 'Volatility_2min']
missing_features = [feat for feat in features if feat not in daily_data.columns]
if missing_features:
    print(f"Missing features in daily_data: {missing_features}")
    features = [feat for feat in features if feat in daily_data.columns]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(daily_data[features])
scaled_df = pd.DataFrame(scaled_data, columns=features, index=daily_data.index)
daily_data[features] = scaled_df

X = daily_data[features]
y = daily_data['Close']

# **Set the prediction horizon**
prediction_horizon = 10  # Predict 10 days ahead

# Shift the target variable to predict N days ahead
y = y.shift(-prediction_horizon)
X = X[:-prediction_horizon]
y = y[:-prediction_horizon]

def create_sequences(X, y, time_steps=10):
    Xs, ys, indices = [], [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps - 1])  # Adjusted index
        indices.append(y.index[i + time_steps - 1])
    return np.array(Xs), np.array(ys), indices

time_steps = 10
X_seq, y_seq, y_indices = create_sequences(X, y, time_steps)

# Split the data
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]
indices_train, indices_test = y_indices[:split], y_indices[split:]

# Check if the model file exists
if os.path.exists('./TrainedModels/LSTM/lstm_crude_oil_model.h5'):
    print("Model file exists. Loading model...")
    model = load_model('./TrainedModels/LSTM/lstm_crude_oil_model.h5')
    history = None  # No training history
else:
    print("Model file not found. Building and training the model...")
    # Build and train the LSTM model
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model = Sequential()
    model.add(Input(shape=(time_steps, X_train.shape[2])))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(
        X_train, y_train,
        epochs=500,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop]
    )
    # Save the trained model
    model.save('./TrainedModels/LSTM/lstm_crude_oil_model.h5')

# Make predictions and inverse scaling
predictions = model.predict(X_test)
X_test_flat = X.iloc[-len(y_test):]
col_idx = X_test_flat.columns.get_loc('Close')

# Prepare data for inverse scaling
predictions_full = X_test_flat.copy()
predictions_full.iloc[:, col_idx] = predictions.flatten()
predictions_inv_full = scaler.inverse_transform(predictions_full)
predictions_inv_full = pd.DataFrame(predictions_inv_full, columns=features, index=X_test_flat.index)
predictions_inv = predictions_inv_full['Close']

y_test_full = X_test_flat.copy()
y_test_full.iloc[:, col_idx] = y_test
y_test_inv_full = scaler.inverse_transform(y_test_full)
y_test_inv_full = pd.DataFrame(y_test_inv_full, columns=features, index=X_test_flat.index)
y_test_inv = y_test_inv_full['Close']

predictions_inv = predictions_inv.values.reshape(-1)
y_test_inv = y_test_inv.values.reshape(-1)

X_test_inv_full = scaler.inverse_transform(X_test_flat)
X_test_inv_full = pd.DataFrame(X_test_inv_full, columns=features, index=X_test_flat.index)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
print(f'Root Mean Squared Error: {rmse}')

# **Modify Trading Strategy and Plotting**

# Calculate expected percentage changes over the prediction horizon
expected_pct_changes = []
for i in range(len(predictions_inv)):
    current_price = X_test_inv_full.iloc[i]['Close']
    predicted_price = predictions_inv[i]
    expected_pct_change = (predicted_price - current_price) / current_price
    expected_pct_changes.append(expected_pct_change)

# Define dynamic thresholds based on historical data or desired sensitivity
buy_threshold = 0.02  # 2% increase
sell_threshold = -0.1  # 5% decrease

signals = []
for expected_pct_change in expected_pct_changes:
    if expected_pct_change >= buy_threshold:
        signals.append(1)  # Buy
    elif expected_pct_change <= sell_threshold:
        signals.append(-1)  # Sell
    else:
        signals.append(0)  # Hold

# Define maximum expected changes for normalization
max_expected_increase = 0.01  # 5% increase
max_expected_decrease = -0.01  # -5% decrease

trade_sizes = []
for i, expected_pct_change in enumerate(expected_pct_changes):
    if expected_pct_change >= buy_threshold:
        trade_size = min(expected_pct_change / max_expected_increase, 1.0)
    elif expected_pct_change <= sell_threshold:
        if expected_pct_change <= -0.1:  # 7% drop
            trade_size = 1.0
        else:
            trade_size = min(-expected_pct_change / max_expected_decrease, 1.0)
    else:
        trade_size = 0.0
    trade_sizes.append(trade_size)

num_buy_signals = signals.count(1)
num_sell_signals = signals.count(-1)
num_hold_signals = signals.count(0)

print(f"Number of Buy signals: {num_buy_signals}")
print(f"Number of Sell signals: {num_sell_signals}")
print(f"Number of Hold signals: {num_hold_signals}")

# **Initial Investment Separately for Buy-and-Hold**
initial_investment = 10000.0
buy_hold_investment = initial_investment  # Use initial_investment for buy-and-hold
initial_price_buy_hold = y_test_inv[0]
units_buy_hold = buy_hold_investment / initial_price_buy_hold
buy_hold_portfolio = units_buy_hold * y_test_inv

# **Trading Simulation**
# Re-initialize investment and positions for trading simulation
investment = initial_investment
positions = 0.0
portfolio = []
transaction_fee = 0.0

print("\nStarting Trading Simulation...\n")

for i in range(len(signals)):
    price = X_test_inv_full.iloc[i]['Close']  # Use current price
    trade_signal = signals[i]
    trade_size = trade_sizes[i]
    date = indices_test[i].strftime('%Y-%m-%d')

    if price <= 0:
        print(f"Warning: Invalid price on {date}: {price}. Skipping trade.")
        portfolio_value = investment + positions * price
        portfolio.append(portfolio_value)
        continue

    if trade_signal == 1 and investment > 0:
        amount_to_invest = investment * trade_size
        units = (amount_to_invest - amount_to_invest * transaction_fee) / price
        if units > 0:
            cost = units * price
            fee = cost * transaction_fee
            total_cost = cost + fee
            investment_before = investment
            investment -= total_cost
            positions += units
            investment_after = investment
            print(f"Buy on {date}:\n  Expected increase over {prediction_horizon} days: {expected_pct_changes[i]*100:.2f}%\n  Trade size: {trade_size*100:.2f}%\n  Investment before: ${investment_before:.2f}\n  Units bought: {units:.6f}\n  Price: ${price:.2f}\n  Investment after: ${investment_after:.2f}\n")

    elif trade_signal == -1 and positions > 0:
        units_to_sell = positions * trade_size
        if units_to_sell > 0:
            revenue = units_to_sell * price
            fee = revenue * transaction_fee
            total_revenue = revenue - fee
            investment_before = investment
            investment += total_revenue
            positions -= units_to_sell
            investment_after = investment
            print(f"Sell on {date}:\n  Expected decrease over {prediction_horizon} days: {expected_pct_changes[i]*100:.2f}%\n  Trade size: {trade_size*100:.2f}%\n  Investment before: ${investment_before:.2f}\n  Units sold: {units_to_sell:.6f}\n  Price: ${price:.2f}\n  Investment after: ${investment_after:.2f}\n")

    portfolio_value = investment + positions * price
    portfolio.append(portfolio_value)

# **Plot portfolio values over time**
plt.figure(figsize=(12, 6))
plt.plot(indices_test, portfolio, label='Model Portfolio')
plt.plot(indices_test, buy_hold_portfolio, label='Buy and Hold Portfolio')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.show()

# Plot difference between strategies
difference = np.array(portfolio) - buy_hold_portfolio
plt.figure(figsize=(12, 6))
plt.plot(indices_test, difference, label='Profit/Loss vs Buy and Hold')
plt.title('Profit/Loss Compared to Buy and Hold Strategy')
plt.xlabel('Date')
plt.ylabel('Profit/Loss ($)')
plt.legend()
plt.show()

# Print final values for both strategies
print(f'\nFinal Model Portfolio Value: ${portfolio[-1]:.2f}')
print(f'Final Buy and Hold Portfolio Value: ${buy_hold_portfolio[-1]:.2f}')

# Visualize Predictions
plt.figure(figsize=(12, 6))
plt.plot(indices_test, y_test_inv.flatten(), label='Actual')
plt.plot(indices_test, predictions_inv.flatten(), label='Predicted')
plt.title('Actual vs Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot training and validation loss
if history is not None:
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
