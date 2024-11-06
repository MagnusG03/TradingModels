import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Set date ranges
end_date = datetime.datetime.today()
start_date_daily = end_date - datetime.timedelta(days=365 * 23)  # 23 years

# Function to download daily data with error handling
def download_daily_data(ticker, start, end):
    try:
        data = yf.download(
            ticker,
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d'),
            interval='1d',
            progress=False  # Hide download progress
        )
        if data.empty:
            print(f"Data for {ticker} with interval 1d is empty.")
        return data
    except Exception as e:
        print(f"Error downloading {ticker} with interval 1d: {e}")
        return pd.DataFrame()

# Download daily data for S&P 500
daily_data = download_daily_data('^GSPC', start_date_daily, end_date)

# Flatten MultiIndex columns if present in daily_data
if isinstance(daily_data.columns, pd.MultiIndex):
    daily_data.columns = daily_data.columns.get_level_values(0)
daily_data.reset_index(inplace=True)
daily_data.set_index('Date', inplace=True)

# Check for missing columns
required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
missing_columns = [col for col in required_columns if col not in daily_data.columns]
if missing_columns:
    raise ValueError(f"Missing columns in daily_data: {missing_columns}")

# Calculate technical indicators
# Moving Averages
daily_data['MA10'] = daily_data['Close'].rolling(window=10, min_periods=10).mean()
daily_data['MA50'] = daily_data['Close'].rolling(window=50, min_periods=50).mean()

# Relative Strength Index (RSI)
window_length = 14
delta = daily_data['Close'].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(window=window_length, min_periods=window_length).mean()
avg_loss = loss.rolling(window=window_length, min_periods=window_length).mean()

# Avoid division by zero
epsilon = 1e-10
rs = avg_gain / (avg_loss + epsilon)
daily_data['RSI'] = 100 - (100 / (1 + rs))

# Volatility (Standard Deviation of Daily Returns)
daily_data['Returns'] = daily_data['Close'].pct_change()
daily_data['Volatility'] = daily_data['Returns'].rolling(window=10, min_periods=10).std()

# Remove infinite values and NaNs
daily_data.replace([np.inf, -np.inf], np.nan, inplace=True)
daily_data.dropna(inplace=True)

# Define features and target variable
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50', 'RSI', 'Volatility']
missing_features = [feat for feat in features if feat not in daily_data.columns]
if missing_features:
    print(f"Missing features in daily_data: {missing_features}")
    features = [feat for feat in features if feat in daily_data.columns]

# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(daily_data[features])
scaled_df = pd.DataFrame(scaled_data, columns=features, index=daily_data.index)
daily_data[features] = scaled_df

# Diagnostic prints to check scaled data
print("Scaled features (first 5 rows):")
print(daily_data[features].head())
print("Any NaNs in scaled data:", np.isnan(daily_data[features].values).any())

# Prepare input and target
X = daily_data[features]
y = daily_data['Close'].shift(-1)

# Remove the last row with NaN in y
X = X[:-1]
y = y[:-1]

# Function to create sequences
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10
X_seq, y_seq = create_sequences(X, y, time_steps)

# Check if there is sufficient data
if len(X_seq) == 0:
    raise ValueError("Sequences are empty. Check time_steps and data.")

print(f"Number of sequences: {X_seq.shape[0]}")

# Split the data into training and testing sets
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

print(f"Train set: {X_train.shape[0]} sequences")
print(f"Test set: {X_test.shape[0]} sequences")

# Build and train the LSTM model
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(time_steps, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=16,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# Make predictions and inverse scaling
predictions = model.predict(X_test)
X_test_flat = X.iloc[-len(y_test):]

if 'Close' in X_test_flat.columns:
    col_idx = X_test_flat.columns.get_loc('Close')
else:
    raise KeyError("'Close' column not found in X_test_flat.")

# Prepare data for inverse scaling
predictions_full = X_test_flat.copy()
predictions_full['Close'] = predictions.flatten()
predictions_inv_full = scaler.inverse_transform(predictions_full)
predictions_inv_full = pd.DataFrame(predictions_inv_full, columns=features, index=X_test_flat.index)
predictions_inv = predictions_inv_full['Close']

y_test_full = X_test_flat.copy()
y_test_full['Close'] = y_test
y_test_inv_full = scaler.inverse_transform(y_test_full)
y_test_inv_full = pd.DataFrame(y_test_inv_full, columns=features, index=X_test_flat.index)
y_test_inv = y_test_inv_full['Close']

predictions_inv = predictions_inv.values.reshape(-1)
y_test_inv = y_test_inv.values.reshape(-1)

# Check for NaNs in predictions and target
print("Any NaNs in predictions:", np.isnan(predictions_inv).any())
print("Any NaNs in target variable:", np.isnan(y_test_inv).any())

# Remove any remaining NaN values
valid_indices = ~np.isnan(predictions_inv) & ~np.isnan(y_test_inv)
predictions_inv = predictions_inv[valid_indices]
y_test_inv = y_test_inv[valid_indices]

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
mae = mean_absolute_error(y_test_inv, predictions_inv)
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Error: {mae}')

# Generate buy and sell signals with more aggressive thresholds to encourage buying and selling
signals = []
for i in range(len(predictions_inv)):
    if predictions_inv[i] > y_test_inv[i] * 1.03:
        signals.append(2)  # Strong Buy
    elif predictions_inv[i] > y_test_inv[i] * 1.01:
        signals.append(1)  # Buy
    elif predictions_inv[i] < y_test_inv[i] * 0.98:
        signals.append(-2)  # Sell
    elif predictions_inv[i] < y_test_inv[i] * 0.95:
        signals.append(-1)  # Strong Sell
    else:
        signals.append(0)  # Hold

# Backtest the strategy with correct leverage handling and detailed transaction logging
initial_investment = 10000
leverage = 4  # 4x leverage
portfolio_value = initial_investment
portfolio = []
invested = False  # Flag to indicate if the portfolio is currently invested

print("\nStarting Backtest...\n")

for i in range(len(signals)):
    signal = signals[i]
    price = y_test_inv[i]
    date = X_test_flat.index[i]
    
    # Calculate market return
    if i == 0:
        previous_price = price
        market_return = 0
    else:
        previous_price = y_test_inv[i - 1]
        market_return = (price / previous_price) - 1
    
    if invested:
        # Apply leveraged return
        leveraged_return = market_return * leverage
        portfolio_value *= (1 + leveraged_return)
    
    # Handle signals
    if signal in [2, 1] and not invested:  # Buy signals
        units_bought = (initial_investment * leverage) / price
        print(f"[{date.date()}] BUY SIGNAL: {'Strong Buy' if signal == 2 else 'Buy'}")
        print(f"Portfolio Before Transaction: ${portfolio_value:.2f}")
        print(f"Transaction Price: ${price:.2f}")
        print(f"Units Bought: {units_bought:.4f}")
        # Portfolio value remains the same; returns are leveraged
        print(f"Portfolio After Transaction: ${portfolio_value:.2f}\n")
        invested = True  # Set flag to invested
    
    elif signal in [-1, -2] and invested:  # Sell signals
        units_sold = (initial_investment * leverage) / price
        print(f"[{date.date()}] SELL SIGNAL: {'Strong Sell' if signal == -1 else 'Sell'}")
        print(f"Portfolio Before Transaction: ${portfolio_value:.2f}")
        print(f"Transaction Price: ${price:.2f}")
        print(f"Units Sold: {units_sold:.4f}")
        # Portfolio value remains the same; leverage is removed
        print(f"Portfolio After Transaction: ${portfolio_value:.2f}\n")
        invested = False  # Reset flag to not invested
    
    # Append current portfolio value
    portfolio.append(portfolio_value)

# Buy-and-hold strategy
initial_price_bh = y_test_inv[0]
units_buy_hold = initial_investment / initial_price_bh
buy_hold_portfolio = pd.Series(units_buy_hold * y_test_inv[:len(portfolio)], index=X_test_flat.index[:len(portfolio)])

# Align dates for plotting
plot_dates = X_test_flat.index[-len(portfolio):]
buy_hold_portfolio.index = plot_dates  # Ensure indices match

# Plot portfolio values over time
plt.figure(figsize=(12, 6))
plt.plot(plot_dates, portfolio, label='Model Portfolio')
plt.plot(plot_dates, buy_hold_portfolio, label='Buy and Hold Portfolio')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.show()

# Plot difference between strategies
difference = np.array(portfolio) - buy_hold_portfolio
plt.figure(figsize=(12, 6))
plt.plot(plot_dates, difference, label='Difference (Model - Buy and Hold)')
plt.title('Difference Between Model and Buy & Hold Strategy')
plt.xlabel('Date')
plt.ylabel('Difference in Portfolio Value ($)')
plt.legend()
plt.show()

# Print final portfolio values
final_portfolio_value = portfolio[-1] if portfolio else initial_investment
final_buy_hold_value = buy_hold_portfolio.iloc[-1] if not buy_hold_portfolio.empty else initial_investment

print(f'Final Portfolio Value: ${final_portfolio_value:.2f}')
print(f'Final Buy and Hold Portfolio Value: ${final_buy_hold_value:.2f}')
