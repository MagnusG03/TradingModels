import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warning messages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Input, Dropout, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, Layer
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import AdamW
#from tensorflow_addons.optimizers import AdamW  # Import from tensorflow_addons
from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow.keras.losses import Huber, MeanSquaredError
from tensorflow.keras.regularizers import l2

# Ensure reproducibility
np.random.seed(42)
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

# **Håndtere Negative 'Close'-Verdier med Forward Fill**
# 1. Identifiser negative eller null 'Close'-verdier
negative_close_mask = daily_data['Close'] <= 0
print(f"Antall dager med negative eller null 'Close': {negative_close_mask.sum()}")

# 2. Erstatt negative 'Close' verdier med NaN for å bruke ffill
daily_data.loc[negative_close_mask, 'Close'] = np.nan

# 3. Bruk forward fill og backfill for å erstatte NaN med forrige gyldige 'Close' verdi
daily_data['Close'] = daily_data['Close'].ffill().bfill()

# Fyll andre manglende verdier
daily_data = daily_data.ffill()

# Convert 'Close' to float
daily_data['Close'] = daily_data['Close'].astype(float)

# Calculate moving averages and RSI
daily_data['MA10'] = daily_data['Close'].rolling(window=10).mean()
daily_data['MA20'] = daily_data['Close'].rolling(window=20).mean()  # For Bollinger Bands
daily_data['MA50'] = daily_data['Close'].rolling(window=50).mean()

window_length = 14
delta = daily_data['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=window_length).mean()
loss = -delta.where(delta < 0, 0).rolling(window=window_length).mean()
rs = gain / loss
daily_data['RSI'] = 100 - (100 / (1 + rs))

# Calculate Bollinger Bands
daily_data['BB_upper'] = daily_data['MA20'] + (daily_data['Close'].rolling(window=20).std() * 2)
daily_data['BB_lower'] = daily_data['MA20'] - (daily_data['Close'].rolling(window=20).std() * 2)

# Calculate MACD
exp1 = daily_data['Close'].ewm(span=12, adjust=False).mean()
exp2 = daily_data['Close'].ewm(span=26, adjust=False).mean()
daily_data['MACD'] = exp1 - exp2
daily_data['Signal_Line'] = daily_data['MACD'].ewm(span=9, adjust=False).mean()

# Add lagged Close prices
for lag in range(1, 4):
    daily_data[f'Close_lag_{lag}'] = daily_data['Close'].shift(lag)

# Legg til flere tekniske indikatorer
daily_data['ATR'] = daily_data['Close'].rolling(window=14).apply(lambda x: np.mean(np.abs(x - x.shift(1))), raw=False)
low_min = daily_data['Low'].rolling(window=14).min()
high_max = daily_data['High'].rolling(window=14).max()
daily_data['Stochastic'] = 100 * ((daily_data['Close'] - low_min) / (high_max - low_min))
daily_data['OBV'] = (np.sign(daily_data['Close'].diff()) * daily_data['Volume']).fillna(0).cumsum()

# Add more lagged features
for feature in ['Open', 'High', 'Low', 'Volume', 'MA10', 'MA20', 'MA50', 'RSI', 
                'Volatility_hourly', 'Volatility_2min', 'BB_upper', 'BB_lower',
                'MACD', 'Signal_Line', 'ATR', 'Stochastic', 'OBV']:
    for lag in range(1, 4):
        daily_data[f'{feature}_lag_{lag}'] = daily_data[feature].shift(lag)

# Fyll manglende verdier
daily_data.bfill(inplace=True)

# **Initial Investering Separat for Buy-and-Hold**
initial_investment = 10000.0
investment = initial_investment  # For trading-simuleringen

# Features and target variable
features = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA20', 'MA50', 'RSI',
    'Volatility_hourly', 'Volatility_2min', 'BB_upper', 'BB_lower',
    'MACD', 'Signal_Line', 'ATR', 'Stochastic', 'OBV',
    'Close_lag_1',
    'Open_lag_1',
    'High_lag_1',
    'Low_lag_1',
    'Volume_lag_1',
    'MA10_lag_1',
    'MA20_lag_1',
    'MA50_lag_1',
    'RSI_lag_1',
    'Volatility_hourly_lag_1',
    'Volatility_2min_lag_1',
    'BB_upper_lag_1',
    'BB_lower_lag_1',
    'MACD_lag_1',
    'Signal_Line_lag_1',
    'ATR_lag_1',
    'Stochastic_lag_1',
    'OBV_lag_1'
]

# Remove any potential missing features
missing_features = [feat for feat in features if feat not in daily_data.columns]
if missing_features:
    print(f"Missing features in daily_data: {missing_features}")
    features = [feat for feat in features if feat in daily_data.columns]

# Scale features and target separately
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = feature_scaler.fit_transform(daily_data[features])
y_scaled = target_scaler.fit_transform(daily_data[['Close']])

X = pd.DataFrame(X_scaled, columns=features, index=daily_data.index)
y = pd.Series(y_scaled.flatten(), index=daily_data.index)

# Shift target variable to predict next day's close
y = y.shift(-1)
X = X[:-1]
y = y[:-1]

def create_sequences(X, y, time_steps=30):
    Xs, ys, indices = [], [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
        indices.append(y.index[i + time_steps])
    return np.array(Xs), np.array(ys), indices

time_steps = 30  # Increased sequence length
X_seq, y_seq, y_indices = create_sequences(X, y, time_steps)

# Split the data
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]
indices_train, indices_test = y_indices[:split], y_indices[split:]

# Build and train the Transformer model
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim // self.num_heads)
        self.ffn = Sequential([
            Dense(self.ff_dim, activation='relu', kernel_regularizer=l2(1e-4)),
            Dense(self.embed_dim, kernel_regularizer=l2(1e-4)),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(self.rate)
        self.dropout2 = Dropout(self.rate)
        
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, training=training)  # Self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Add & Norm
        ffn_output = self.ffn(out1)  # Feed Forward
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)  # Add & Norm

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
        })
        return config

class PositionalEncoding(Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'position': self.position,
            'd_model': self.d_model,
        })
        return config

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )

        # Apply sin to even indices; cos to odd indices
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        # Handle potential dimension mismatch if d_model is odd
        sines = tf.pad(sines, [[0, 0], [0, tf.shape(cosines)[1] - tf.shape(sines)[1]]])
        cosines = tf.pad(cosines, [[0, 0], [0, tf.shape(sines)[1] - tf.shape(cosines)[1]]])

        # Concatenate sines and cosines
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]  # (1, position, d_model)

        return pos_encoding

    def get_angles(self, position, i, d_model):
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angle_rates

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]

# Modellparametere
embed_dim = X_train.shape[2]  # Embedding size for each token
num_heads = 12  # Number of attention heads
ff_dim = 512  # Feed-forward network dimension
num_layers = 4  # Number of Transformer blocks
time_steps = X_train.shape[1]  # Should be 30 based on your sequence length

inputs = Input(shape=(time_steps, embed_dim))
x = PositionalEncoding(time_steps, embed_dim)(inputs)
x = LayerNormalization(epsilon=1e-6)(x)  # Optional: LayerNorm after Positional Encoding

for _ in range(num_layers):
    x = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, rate=0.3)(x)  # Increased dropout rate

x = GlobalAveragePooling1D()(x)
x = Dropout(0.3)(x)  # Increased dropout rate
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)

# Optimizer and loss function
optimizer = AdamW(learning_rate=1e-4, weight_decay=1e-5)
model.compile(optimizer=optimizer, loss='huber')

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
callbacks = [early_stop, reduce_lr, checkpoint]

# **Trening**
history = model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.1,
    callbacks=callbacks
)

# Make predictions and inverse scaling
predictions_scaled = model.predict(X_test)
predictions_inv = target_scaler.inverse_transform(predictions_scaled)
y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))

# For generating signals, we need the X_test data in original scale
X_test_flat = X.iloc[-(len(y_test)):]
X_test_inv_full = feature_scaler.inverse_transform(X_test_flat)
X_test_inv_full = pd.DataFrame(X_test_inv_full, columns=features, index=X_test_flat.index)

# **Generer Signals**
signals = []
for i in range(len(predictions_inv)):
    if predictions_inv[i] > X_test_inv_full.iloc[i]['Close'] * 1.01:
        signals.append(1)  # Buy
    elif predictions_inv[i] < X_test_inv_full.iloc[i]['Close'] * 0.99:
        signals.append(-1)  # Sell
    else:
        signals.append(0)  # Hold

num_buy_signals = signals.count(1)
num_sell_signals = signals.count(-1)
num_hold_signals = signals.count(0)

print(f"Number of Buy signals: {num_buy_signals}")
print(f"Number of Sell signals: {num_sell_signals}")
print(f"Number of Hold signals: {num_hold_signals}")

# **Buy-and-Hold Strategy med Separat Initial Investering**
buy_hold_investment = initial_investment  # Bruk initial_investment for buy-and-hold
initial_price_buy_hold = y_test_inv[0][0]
units_buy_hold = buy_hold_investment / initial_price_buy_hold
buy_hold_portfolio = units_buy_hold * y_test_inv.flatten()

# Verifiser at den første verdien er lik investeringen
print(f"\nInitial pris (Buy and Hold): {initial_price_buy_hold}")
print(f"Units kjøpt (Buy and Hold): {units_buy_hold}")
print(f"Første verdi i buy_hold_portfolio: {buy_hold_portfolio[0]}")  # Skal være ~$10,000

# Sjekk de første 5 verdiene i buy_hold_portfolio
print("\nFørste 5 verdier i buy_hold_portfolio:")
print(buy_hold_portfolio[:5])

# **Trading Simulation**
# Re-initialize investment and positions for trading simulation
investment = initial_investment
positions = 0
portfolio = []
transaction_fee = 0.0  # Set transaction fee (e.g., 0.001 for 0.1%)

print("\nStarting Trading Simulation...\n")

for i in range(len(signals)):
    price = y_test_inv[i][0]
    trade_signal = signals[i]
    trade_size = 0.5  # Trade 50% of available funds or positions
    date = indices_test[i].strftime('%Y-%m-%d')

    # Sjekk at prisen er positiv før handel
    if price <= 0:
        print(f"Advarsel: Ikke gyldig pris på {date}: {price}. Hopper over handel.")
        portfolio_value = investment + positions * price
        portfolio.append(portfolio_value)
        continue

    if trade_signal == 1 and investment >= price:
        amount_to_invest = investment * trade_size
        units = (amount_to_invest - amount_to_invest * transaction_fee) / price
        units = int(units)
        if units > 0:
            cost = units * price
            fee = cost * transaction_fee
            total_cost = cost + fee
            investment_before = investment
            investment -= total_cost
            positions += units
            investment_after = investment
            print(f"Buy on {date}:\n  Investment before: ${investment_before:.2f}\n  Units bought: {units}\n  Price: ${price:.2f}\n  Investment after: ${investment_after:.2f}\n")
    
    elif trade_signal == -1 and positions > 0:
        units_to_sell = int(positions * trade_size)
        if units_to_sell > 0:
            revenue = units_to_sell * price
            fee = revenue * transaction_fee
            total_revenue = revenue - fee
            investment_before = investment
            investment += total_revenue
            positions -= units_to_sell
            investment_after = investment
            print(f"Sell on {date}:\n  Investment before: ${investment_before:.2f}\n  Units sold: {units_to_sell}\n  Price: ${price:.2f}\n  Investment after: ${investment_after:.2f}\n")
    
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
plt.figure(figsize=(12,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
