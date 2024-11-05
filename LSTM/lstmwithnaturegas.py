import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Set the date ranges
end_date = datetime.datetime.today()
start_date_daily = end_date - datetime.timedelta(days=365 * 23)  # 23 years
start_date_hourly = end_date - datetime.timedelta(days=720)      # 720 days
start_date_2min = end_date - datetime.timedelta(days=60)         # 60 days

# Download daily data for Natural Gas
daily_data = yf.download(
    'NG=F',
    start=start_date_daily.strftime('%Y-%m-%d'),
    end=end_date.strftime('%Y-%m-%d'),
    interval='1d'
)
# Flatten MultiIndex columns if present in daily_data
if isinstance(daily_data.columns, pd.MultiIndex):
    daily_data.columns = daily_data.columns.get_level_values(0)
daily_data.reset_index(inplace=True)
daily_data.set_index('Date', inplace=True)

# Download hourly data for Natural Gas
hourly_data = yf.download(
    'NG=F',
    start=start_date_hourly.strftime('%Y-%m-%d'),
    end=end_date.strftime('%Y-%m-%d'),
    interval='1h'
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

# Download 2-minute data for Natural Gas
data_2min = yf.download(
    'NG=F',
    start=start_date_2min.strftime('%Y-%m-%d'),
    end=end_date.strftime('%Y-%m-%d'),
    interval='2m'
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
daily_data['Volatility_hourly'].fillna(method='ffill', inplace=True)
daily_data['Volatility_2min'].fillna(method='ffill', inplace=True)

daily_data.ffill(inplace=True)
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
y = daily_data['Close'].shift(-1)
X = X[:-1]
y = y[:-1]

def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10
X_seq, y_seq = create_sequences(X, y, time_steps)

# Split the data
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# Build and train the LSTM model
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model = Sequential()
model.add(Input(shape=(time_steps, X_train.shape[2])))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=16,
    validation_split=0.1,
    callbacks=[early_stop]
)

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

# Generate buy and sell signals with a more aggressive threshold to encourage buying and selling
signals = []
for i in range(len(predictions_inv)):
    if predictions_inv[i] > X_test_inv_full.iloc[i]['Close'] * 1.15:
        signals.append(2)  # Strong Buy
    elif predictions_inv[i] > X_test_inv_full.iloc[i]['Close']:
        signals.append(1)  # Buy
    elif predictions_inv[i] < X_test_inv_full.iloc[i]['Close'] * 0.96:
        signals.append(-1)  # Strong Sell
    else:
        signals.append(0)  # Hold

# Backtest the strategy
investment = 10000
positions = 0
portfolio = []

for i in range(len(signals)):
    price = y_test_inv[i]
    if signals[i] == 2:  # Strong Buy
        units = (investment // price) * 2  # Double the usual buy amount
        if units > 0:
            investment -= units * price
            positions += units
    elif signals[i] == 1:  # Buy
        units = investment // price  # Buy as many units as possible
        if units > 0:
            investment -= units * price
            positions += units
    elif signals[i] == -1 and positions > 0:  # Strong Sell
        investment += positions * price
        positions = 0
    portfolio.append(investment + positions * price)

# Buy-and-hold strategy
initial_price = y_test_inv[0]
units_buy_hold = 10000 / initial_price
buy_hold_portfolio = units_buy_hold * y_test_inv

# Plot portfolio values over time
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv_full.index, portfolio, label='Model Portfolio')
plt.plot(y_test_inv_full.index, buy_hold_portfolio, label='Buy and Hold Portfolio')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.show()

# Plot difference between strategies
difference = np.array(portfolio) - buy_hold_portfolio
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv_full.index, difference, label='Profit/Loss vs Buy and Hold')
plt.title('Profit/Loss Compared to Buy and Hold Strategy')
plt.xlabel('Date')
plt.ylabel('Profit/Loss ($)')
plt.legend()
plt.show()

# Print final values for both strategies
print(f'Final Model Portfolio Value: {portfolio[-1]}')
print(f'Final Buy and Hold Portfolio Value: {buy_hold_portfolio[-1]}')

