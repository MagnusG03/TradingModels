# Import necessary libraries
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization, Dropout, Add, Conv1D,
    MultiHeadAttention, GlobalAveragePooling1D
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import ta  # For technical indicators


# Enable GPU usage
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Set date range for the last 700 days
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=700)

# Download historical data
gold_data = yf.download('GC=F', start=start_date, end=end_date, interval='1h', group_by='ticker')
silver_data = yf.download('SI=F', start=start_date, end=end_date, interval='1h', group_by='ticker')
oil_data = yf.download('CL=F', start=start_date, end=end_date, interval='1h', group_by='ticker')

# Reset index to ensure 'Datetime' is a column
gold_data.reset_index(inplace=True)
silver_data.reset_index(inplace=True)
oil_data.reset_index(inplace=True)

# Flatten MultiIndex columns if necessary
def flatten_columns(df):
    new_columns = []
    for col in df.columns.values:
        if isinstance(col, tuple):
            if 'Datetime' in col:
                new_columns.append('Datetime')
            else:
                new_columns.append('_'.join(filter(None, col)).strip('_'))
        else:
            new_columns.append(col)
    df.columns = new_columns
    return df

gold_data = flatten_columns(gold_data)
silver_data = flatten_columns(silver_data)
oil_data = flatten_columns(oil_data)

# Rename 'Datetime_' back to 'Datetime' if necessary
gold_data.rename(columns={'Datetime_': 'Datetime'}, inplace=True)
silver_data.rename(columns={'Datetime_': 'Datetime'}, inplace=True)
oil_data.rename(columns={'Datetime_': 'Datetime'}, inplace=True)

# Renaming columns to avoid conflicts
gold_data = gold_data.rename(columns={
    'GC=F_Open': 'Gold_Open',
    'GC=F_High': 'Gold_High',
    'GC=F_Low': 'Gold_Low',
    'GC=F_Close': 'Gold_Close',
    'GC=F_Adj Close': 'Gold_Adj_Close',
    'GC=F_Volume': 'Gold_Volume'
})

silver_data = silver_data.rename(columns={
    'SI=F_Open': 'Silver_Open',
    'SI=F_High': 'Silver_High',
    'SI=F_Low': 'Silver_Low',
    'SI=F_Close': 'Silver_Close',
    'SI=F_Adj Close': 'Silver_Adj_Close',
    'SI=F_Volume': 'Silver_Volume'
})

oil_data = oil_data.rename(columns={
    'CL=F_Open': 'Oil_Open',
    'CL=F_High': 'Oil_High',
    'CL=F_Low': 'Oil_Low',
    'CL=F_Close': 'Oil_Close',
    'CL=F_Adj Close': 'Oil_Adj_Close',
    'CL=F_Volume': 'Oil_Volume'
})

# Verify that 'Datetime' is in the columns
print("Gold Data Columns:", gold_data.columns.tolist())
print("Silver Data Columns:", silver_data.columns.tolist())
print("Oil Data Columns:", oil_data.columns.tolist())

# Merge dataframes on 'Datetime' column
merged_data = gold_data.merge(silver_data, on='Datetime', how='inner')
merged_data = merged_data.merge(oil_data, on='Datetime', how='inner')

# Handle missing values
merged_data.ffill(inplace=True)
merged_data.bfill(inplace=True)

# Ensure closing prices are float Series
merged_data['Gold_Close'] = merged_data['Gold_Close'].astype(float)
merged_data['Silver_Close'] = merged_data['Silver_Close'].astype(float)
merged_data['Oil_Close'] = merged_data['Oil_Close'].astype(float)

# Calculate additional features
merged_data['Gold_Silver_Ratio'] = merged_data['Gold_Close'] / merged_data['Silver_Close']
merged_data['Gold_Oil_Ratio'] = merged_data['Gold_Close'] / merged_data['Oil_Close']

# Technical Indicators (for all commodities)
for commodity in ['Gold', 'Silver', 'Oil']:
    # Moving Averages
    merged_data[f'{commodity}_MA10'] = merged_data[f'{commodity}_Close'].rolling(window=10).mean()
    merged_data[f'{commodity}_MA50'] = merged_data[f'{commodity}_Close'].rolling(window=50).mean()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=merged_data[f'{commodity}_Close'], window=20)
    merged_data[f'{commodity}_BB_High'] = bollinger.bollinger_hband()
    merged_data[f'{commodity}_BB_Low'] = bollinger.bollinger_lband()
    
    # MACD
    macd = ta.trend.MACD(close=merged_data[f'{commodity}_Close'])
    merged_data[f'{commodity}_MACD'] = macd.macd()
    
    # RSI
    rsi = ta.momentum.RSIIndicator(close=merged_data[f'{commodity}_Close'], window=14)
    merged_data[f'{commodity}_RSI'] = rsi.rsi()

# Fill NaN values resulted from calculations
merged_data.ffill(inplace=True)
merged_data.bfill(inplace=True)

# Features and target variables
features = merged_data.columns.drop(['Datetime', 'Gold_Adj_Close', 'Silver_Adj_Close', 'Oil_Adj_Close'])
targets = ['Gold_Close', 'Silver_Close', 'Oil_Close']

# Scale the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(merged_data[features])

# Convert scaled_data back to DataFrame
scaled_df = pd.DataFrame(
    scaled_data,
    columns=features,
    index=merged_data.index
)

# Assign back to merged_data
merged_data[features] = scaled_df

# Prepare sequences
sequence_length = 50  # Increased sequence length for better context

def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

X = merged_data[features]
y = merged_data[targets]  # Scaled prices for all commodities

X_sequences, y_sequences = create_sequences(X.values, y.values, sequence_length)

# Split the data into training and testing sets
split = int(0.8 * len(X_sequences))
X_train, X_test = X_sequences[:split], X_sequences[split:]
y_train, y_test = y_sequences[:split], y_sequences[split:]

# Build the Transformer model with hyperparameter tuning
# Hyperparameters
head_size = 64
num_heads = 4
ff_dim = 128
num_transformer_blocks = 2
dropout_rate = 0.5

# Positional Encoding Function
def positional_encoding(sequence_length, d_model):
    position = np.arange(sequence_length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = position * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads, dtype=tf.float32)

# Transformer Encoder Block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    # Multi-Head Attention
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-Forward Network
    x_ff = Dense(ff_dim, activation='relu')(x)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x_ff = Dropout(dropout)(x_ff)
    x = Add()([x_ff, x])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

# Input shape: (sequence_length, num_features)
input_shape = X_train.shape[1:]  # (sequence_length, num_features)
inputs = Input(shape=input_shape)

# Positional Encoding
position_encoding = positional_encoding(sequence_length, input_shape[-1])
embedded_inputs = inputs + position_encoding

# Optional: Add Conv1D layer for local feature extraction
x = Conv1D(filters=128, kernel_size=3, padding='causal', activation='relu')(embedded_inputs)

# Transformer Encoder Blocks
for _ in range(num_transformer_blocks):
    x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout_rate)

# Global Average Pooling
x = GlobalAveragePooling1D(data_format='channels_first')(x)

# Output Layer
outputs = Dense(len(targets))(x)

# Build Model
model = Model(inputs, outputs)

# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Enable early stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Predictions
predictions = model.predict(X_test)

# Inverse scaling
def inverse_transform_predictions(predictions, X_test_last, scaler, features, target_cols):
    predictions_full = X_test_last.copy()
    for idx, col in enumerate(target_cols):
        col_idx = features.get_loc(col)
        predictions_full[:, col_idx] = predictions[:, idx]
    predictions_inv_full = scaler.inverse_transform(predictions_full)
    predictions_inv = predictions_inv_full[:, [features.get_loc(col) for col in target_cols]]
    return predictions_inv

# Inverse scaling for predictions
X_test_last = X_test[:, -1, :]  # Get the last time step from each sequence
predictions_inv = inverse_transform_predictions(predictions, X_test_last, scaler, X.columns, targets)

# Inverse scaling for y_test
y_test_inv = inverse_transform_predictions(y_test, X_test_last, scaler, X.columns, targets)

# Calculate performance metrics for each commodity
for idx, commodity in enumerate(targets):
    rmse = np.sqrt(mean_squared_error(y_test_inv[:, idx], predictions_inv[:, idx]))
    mae = mean_absolute_error(y_test_inv[:, idx], predictions_inv[:, idx])
    r2 = r2_score(y_test_inv[:, idx], predictions_inv[:, idx])
    print(f'Performance Metrics for {commodity}:')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R-squared Score: {r2}')
    print('---')

# Plot actual vs predicted prices for each commodity
for idx, commodity in enumerate(targets):
    plt.figure(figsize=(14, 7))
    plt.plot(y_test_inv[:, idx], label=f'Actual {commodity} Prices')
    plt.plot(predictions_inv[:, idx], label=f'Predicted {commodity} Prices')
    plt.title(f'Actual vs Predicted {commodity} Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Trading Strategy for Each Commodity
def unified_trading_strategy(predictions_inv, y_test_inv, initial_investment=10000):
    num_assets = predictions_inv.shape[1]
    cash = initial_investment
    positions = np.zeros(num_assets)
    portfolio_values = []
    
    for i in range(len(predictions_inv)):
        predicted_prices = predictions_inv[i]
        current_prices = y_test_inv[i - 1] if i > 0 else y_test_inv[i]
        signals = predicted_prices > current_prices  # Buy if predicted price is higher
        
        # Sell assets where signal is False
        for idx in range(num_assets):
            if not signals[idx] and positions[idx] > 0:
                cash += positions[idx] * y_test_inv[i, idx]
                positions[idx] = 0
        
        # Buy assets where signal is True
        num_signals = np.sum(signals)
        if num_signals > 0:
            investment_per_asset = cash / num_signals
            for idx in range(num_assets):
                if signals[idx]:
                    price = y_test_inv[i, idx]
                    units = investment_per_asset // price
                    cost = units * price
                    if units > 0 and cost <= cash:
                        cash -= cost
                        positions[idx] += units
        
        # Calculate portfolio value
        portfolio_value = cash + np.sum(positions * y_test_inv[i])
        portfolio_values.append(portfolio_value)
    
    return portfolio_values

# Backtesting the unified strategy
print("Backtesting Unified Strategy")
portfolio_values = unified_trading_strategy(predictions_inv, y_test_inv)

# Plot the portfolio value over time
plt.figure(figsize=(14, 7))
plt.plot(portfolio_values)
plt.title('Unified Portfolio Value Over Time')
plt.xlabel('Time')
plt.ylabel('Portfolio Value ($)')
plt.show()

# Print final portfolio value
print(f'Final Portfolio Value: ${portfolio_values[-1]:.2f}')
