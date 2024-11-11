# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import gym
from gym import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# For LSTM model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Data Collection

# Define the ticker symbol and date range
ticker = 'CL=F'
start_date = '2015-01-01'
end_date = '2023-10-01'

# Fetch historical market data using yfinance
data = yf.download(ticker, start=start_date, end=end_date)
print("Data columns before any processing:")
print(data.columns)

# If columns are MultiIndex, drop the ticker level
if isinstance(data.columns, pd.MultiIndex):
    # Drop the ticker level
    data.columns = data.columns.droplevel(1)
    print("Columns after dropping ticker level:")
    print(data.columns)

# Now 'Close' column is available
close_col = 'Close'

# Remove rows with non-positive 'Close' prices
data = data[data[close_col] > 0]

# Drop rows with NaN in 'Close' prices
data.dropna(subset=[close_col], inplace=True)

# Forward fill any remaining NaN values in features (if applicable)
data.fillna(method='ffill', inplace=True)

data.reset_index(drop=True, inplace=True)  # Reset index after dropping rows

# Verify that all 'Close' prices are positive
assert (data[close_col] > 0).all(), "There are non-positive 'Close' prices in the data after preprocessing."

# Technical Indicators

# Function to compute Relative Strength Index (RSI)
def compute_RSI(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=(window - 1), min_periods=window).mean()
    ema_down = down.ewm(com=(window - 1), min_periods=window).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to compute Moving Average Convergence Divergence (MACD)
def compute_MACD(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd - signal_line
    return macd_histogram

data['MA20'] = data[close_col].rolling(window=20).mean()
data['RSI'] = compute_RSI(data[close_col], window=14)
data['MACD'] = compute_MACD(data[close_col])

# Display the first few rows with new indicators
print(data[['Close', 'MA20', 'RSI', 'MACD']].head(20))

# Fundamental Data

# Function to fetch fundamental data using yfinance
def get_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    fundamental_data = {}
    # Get PE ratio
    try:
        pe_ratio = stock.info.get('trailingPE', np.nan)
    except KeyError:
        pe_ratio = np.nan
    fundamental_data['PE_ratio'] = pe_ratio
    # Get Price to Book ratio
    try:
        pb_ratio = stock.info.get('priceToBook', np.nan)
    except KeyError:
        pb_ratio = np.nan
    fundamental_data['PB_ratio'] = pb_ratio
    # Get Dividend Yield
    try:
        dividend_yield = stock.info.get('dividendYield', np.nan)
    except KeyError:
        dividend_yield = np.nan
    fundamental_data['Dividend_Yield'] = dividend_yield
    # Convert to DataFrame
    fundamental_df = pd.DataFrame(fundamental_data, index=[0])
    return fundamental_df

# Fetch fundamental data
fundamental_data = get_fundamental_data(ticker)

# Add fundamental data to market data
for col in fundamental_data.columns:
    data[col] = fundamental_data[col].values[0]

# Display the updated data
print(data.head())

# Data Preprocessing

# Define the features list before using it
features = ['Close', 'MA20', 'RSI', 'MACD', 'PE_ratio', 'PB_ratio', 'Dividend_Yield']

# Update 'Close' in features to match the actual column name
features[features.index('Close')] = close_col

# Remove 'Dividend_Yield' if it's entirely NaN
if data['Dividend_Yield'].isna().all():
    data.drop(columns=['Dividend_Yield'], inplace=True)
    features.remove('Dividend_Yield')

# Remove 'PE_ratio' and 'PB_ratio' if they are entirely NaN
for col in ['PE_ratio', 'PB_ratio']:
    if data[col].isna().all():
        data.drop(columns=[col], inplace=True)
        features.remove(col)
    else:
        # Optionally fill missing values with mean
        data[col].fillna(data[col].mean(), inplace=True)

# Drop initial rows with NaNs in technical indicators
data.dropna(subset=['MA20', 'RSI', 'MACD'], inplace=True)

# Now drop any remaining rows with NaNs in the features
data.dropna(subset=features, inplace=True)

# Reset index after dropping rows
data.reset_index(drop=True, inplace=True)

# Keep a copy of the original 'Close' price
data['Close_unscaled'] = data[close_col]

# Prepare Data for LSTM Model

# Set the target variable 'y' as the next day's 'Close' price
data['Target'] = data['Close_unscaled'].shift(-1)

# Remove last row with NaN in 'Target'
data.dropna(subset=['Target'], inplace=True)

# Feature Scaling for LSTM
lstm_scaler = MinMaxScaler()
lstm_scaled_data = lstm_scaler.fit_transform(data[['Close_unscaled', 'Target']])

# Prepare sequences
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, 0])  # Use 'Close_unscaled' price
        y.append(data[i+seq_length, 1])    # Use 'Target' price
    return np.array(X), np.array(y)

seq_length = 60  # You can adjust this
X_lstm, y_lstm = create_sequences(lstm_scaled_data, seq_length)

# Reshape X_lstm for LSTM input
X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

# Split into training and testing sets
split = int(0.8 * len(X_lstm))
X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

# Build and Train LSTM Model

lstm_model = Sequential()
lstm_model.add(LSTM(128, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(64))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Use early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
history = lstm_model.fit(X_train_lstm, y_train_lstm, epochs=100, batch_size=32,
                         validation_data=(X_test_lstm, y_test_lstm), callbacks=[early_stopping])

# Make Predictions

# Predict on the entire dataset
full_predictions = lstm_model.predict(X_lstm)

# Prepare LSTM predictions as a feature
# Since we shifted 'Target' by -1 and used sequences, we need to align the predictions properly

# Initialize lstm_predictions with NaNs
lstm_predictions = np.empty((len(data)))
lstm_predictions[:] = np.nan

# Inverse transform the predictions
inverse_predictions = lstm_scaler.inverse_transform(
    np.concatenate((lstm_scaled_data[seq_length:, 0].reshape(-1,1), full_predictions), axis=1)
)[:,1]

# Assign predictions to lstm_predictions from index 'seq_length' onward
lstm_predictions[seq_length:] = inverse_predictions

# Add LSTM predictions to the DataFrame
data['LSTM_Prediction_unscaled'] = lstm_predictions

# Drop any rows with NaN in 'LSTM_Prediction_unscaled'
data.dropna(subset=['LSTM_Prediction_unscaled'], inplace=True)

# Reset index after dropping rows
data.reset_index(drop=True, inplace=True)

# Keep unscaled versions for environment calculations
data['Close_unscaled'] = data['Close_unscaled']
data['LSTM_Prediction_unscaled'] = data['LSTM_Prediction_unscaled']

# **Rename 'LSTM_Prediction_unscaled' to 'LSTM_Prediction' for scaling**
data.rename(columns={'LSTM_Prediction_unscaled': 'LSTM_Prediction'}, inplace=True)

# Update features list to include 'LSTM_Prediction'
features.append('LSTM_Prediction')

# Define features to scale (including 'Close' and 'LSTM_Prediction')
features_to_scale = features.copy()

# Separate features that need special scaling
robust_features = ['MACD']
minmax_features = [feat for feat in features_to_scale if feat not in robust_features]

# Feature Scaling
from sklearn.preprocessing import RobustScaler

# Apply MinMaxScaler to minmax_features
scaler_minmax = MinMaxScaler()
data[minmax_features] = scaler_minmax.fit_transform(data[minmax_features])

# Apply RobustScaler to robust_features
scaler_robust = RobustScaler()
data[robust_features] = scaler_robust.fit_transform(data[robust_features])

# Verify scaling for minmax_features
tol = 1e-6
min_vals = data[minmax_features].min()
max_vals = data[minmax_features].max()

# Print min and max values for each feature
for feature in features_to_scale:
    min_val = data[feature].min()
    max_val = data[feature].max()
    print(f"Feature '{feature}' - min: {min_val}, max: {max_val}")

# Adjusted assertion
assert (min_vals >= -tol).all() and (max_vals <= 1 + tol).all(), "Features not scaled properly."

# Custom Trading Environment for Reinforcement Learning

class CustomTradingEnv(gym.Env):
    """A custom trading environment for reinforcement learning"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(CustomTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.df_total_steps = len(self.df) - 1

        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Adjust observation space limits
        obs_low = np.full(len(features), -np.inf)
        obs_high = np.full(len(features), np.inf)

        # Set limits for minmax scaled features
        for idx, feature in enumerate(features):
            if feature in minmax_features:
                obs_low[idx] = 0 - tol
                obs_high[idx] = 1 + tol
            elif feature in robust_features:
                # Since 'MACD' is scaled using RobustScaler, set bounds based on the data
                obs_low[idx] = data[feature].min() - tol
                obs_high[idx] = data[feature].max() + tol

        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Initialize state
        self.current_step = 0
        self.balance = 10000.0  # Starting with $10,000
        self.net_worth = 10000.0
        self.max_net_worth = 10000.0
        self.shares_held = 0
        self.cost_basis = 0.0
        self.total_shares_sold = 0
        self.total_sales_value = 0.0
        self.initial_net_worth = 10000.0
        self.returns = []

        # Define the transaction fee rate (1% in this case)
        self.transaction_fee_rate = 0.01  # 1% transaction fee

    def reset(self):
        self.current_step = 0
        self.balance = 10000.0
        self.net_worth = 10000.0
        self.max_net_worth = 10000.0
        self.shares_held = 0
        self.cost_basis = 0.0
        self.total_shares_sold = 0
        self.total_sales_value = 0.0
        self.initial_net_worth = 10000.0
        self.returns = []

        return self._next_observation()

    def _next_observation(self):
        obs = self.df.iloc[self.current_step][features].astype('float32').values
        return obs

    def step(self, action):
        # Execute one time step within the environment
        current_price = float(self.df.iloc[self.current_step]['Close_unscaled'])

        done = False

        # Action logic
        if action == 0:  # Hold
            pass

        elif action == 1:  # Buy
            # Calculate maximum shares we can buy considering the transaction fee
            max_shares = self.balance // (current_price * (1 + self.transaction_fee_rate))
            if max_shares > 0:
                # Calculate the total cost including the transaction fee
                total_cost = max_shares * current_price * (1 + self.transaction_fee_rate)
                self.balance -= total_cost
                prev_shares = self.shares_held
                self.shares_held += max_shares
                # Update cost basis to include the transaction fee
                if prev_shares == 0:
                    self.cost_basis = current_price * (1 + self.transaction_fee_rate)
                else:
                    self.cost_basis = ((self.cost_basis * prev_shares) + (current_price * (1 + self.transaction_fee_rate) * max_shares)) / self.shares_held

        elif action == 2:  # Sell
            if self.shares_held > 0:
                # Calculate the total proceeds after deducting the transaction fee
                total_proceeds = self.shares_held * current_price * (1 - self.transaction_fee_rate)
                self.balance += total_proceeds
                self.total_shares_sold += self.shares_held
                self.total_sales_value += total_proceeds
                self.shares_held = 0
                self.cost_basis = 0.0

        self.current_step += 1

        if self.current_step >= self.df_total_steps:
            done = True

        # Update net worth
        prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # Calculate reward (profit or loss)
        reward = self.net_worth - prev_net_worth

        # Append daily return for performance metrics
        daily_return = (self.net_worth - self.initial_net_worth) / self.initial_net_worth
        self.returns.append(daily_return)

        # Return step information
        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape)
        info = {}
        return obs, reward, done, info

    def render(self, mode='human', close=False):
        # Implement render logic (optional)
        profit = self.net_worth - self.initial_net_worth
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Net worth: {self.net_worth}')
        print(f'Profit: {profit}')

    def get_portfolio_performance(self):
        total_return = (self.net_worth - self.initial_net_worth) / self.initial_net_worth
        max_drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        returns = pd.Series(self.returns)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }

# Initialize Environment and Agent

# Prepare data for the environment
env_data = data.copy()
env_data.reset_index(inplace=True, drop=True)

# Initialize environment
env = CustomTradingEnv(env_data)

# Initialize agent
agent = PPO('MlpPolicy', env, verbose=1)

# Train agent
agent.learn(total_timesteps=100000)

# Backtesting and Evaluation

# Reset environment
obs = env.reset()

# Variables to track performance
net_worths = [env.net_worth]
balances = [env.balance]
held_shares = [env.shares_held]
actions = []
prices = []
profits = []

# Run the agent in the environment
for _ in range(len(env.df) - env.current_step - 1):
    # Record the current price before stepping
    current_price = float(env.df.iloc[env.current_step]['Close_unscaled'])
    prices.append(current_price)

    action, _states = agent.predict(obs)
    action = int(action)  # Convert action from array to scalar
    obs, reward, done, info = env.step(action)

    # Record the variables
    net_worths.append(env.net_worth)
    balances.append(env.balance)
    held_shares.append(env.shares_held)
    actions.append(action)
    profits.append(env.net_worth - env.initial_net_worth)

    if done:
        break

# Evaluate performance
portfolio_performance = env.get_portfolio_performance()
print(f"Total Return: {portfolio_performance['total_return'] * 100:.2f}%")
print(f"Maximum Drawdown: {portfolio_performance['max_drawdown'] * 100:.2f}%")
print(f"Sharpe Ratio: {portfolio_performance['sharpe_ratio']:.2f}")

# Plot Net Worth Over Time
plt.figure(figsize=(12,6))
plt.plot(net_worths, label='Net Worth')
plt.title('Net Worth Over Time')
plt.xlabel('Time Step')
plt.ylabel('Net Worth ($)')
plt.legend()
plt.show()

# Plot Balance and Equity Curve
plt.figure(figsize=(12,6))
plt.plot(balances, label='Balance')
equity = [net_worths[i] - balances[i] for i in range(len(balances))]
plt.plot(equity, label='Equity in Shares')
plt.title('Balance and Equity Over Time')
plt.xlabel('Time Step')
plt.ylabel('Amount ($)')
plt.legend()
plt.show()

# Plot Actions Over Time
action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
action_names = [action_dict[a] for a in actions]

plt.figure(figsize=(12,6))
plt.plot(prices, label='Price')
buy_signals = [prices[i] if actions[i] == 1 else np.nan for i in range(len(actions))]
sell_signals = [prices[i] if actions[i] == 2 else np.nan for i in range(len(actions))]
plt.scatter(range(len(actions)), buy_signals, marker='^', color='g', label='Buy Signal')
plt.scatter(range(len(actions)), sell_signals, marker='v', color='r', label='Sell Signal')
plt.title('Trading Actions Over Time')
plt.xlabel('Time Step')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Plot Returns Distribution
returns = pd.Series(env.returns)
plt.figure(figsize=(10,5))
plt.hist(returns, bins=50, edgecolor='black')
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.show()

# Compare with Buy-and-Hold Strategy
buy_and_hold_net_worth = [env.initial_net_worth]
num_shares = env.initial_net_worth / prices[0]
for price in prices[1:]:
    net_worth = num_shares * price
    buy_and_hold_net_worth.append(net_worth)

plt.figure(figsize=(12,6))
plt.plot(net_worths, label='Agent Net Worth')
plt.plot(buy_and_hold_net_worth, label='Buy and Hold Net Worth')
plt.title('Agent vs Buy and Hold Strategy')
plt.xlabel('Time Step')
plt.ylabel('Net Worth ($)')
plt.legend()
plt.show()
