import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler, RobustScaler
import gym
from gym import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# Data Collection

ticker = "GC=F"
start_date = "2000-01-01"
end_date = "2023-10-01"

# Fetch historical market data
data = yf.download(ticker, start=start_date, end=end_date)
print("Data columns before any processing:")
print(data.columns)

# If columns are MultiIndex, drop the ticker level
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)
    print("Columns after dropping ticker level:")
    print(data.columns)

close_col = "Close"

# Remove rows with non-positive 'Close' prices
data = data[data[close_col] > 0]

# Drop rows with NaN in 'Close' prices
data.dropna(subset=[close_col], inplace=True)

# Forward fill any remaining NaN values in features
data.fillna(method="ffill", inplace=True)

data.reset_index(drop=True, inplace=True)

# Verify that all 'Close' prices are positive
assert (
    data[close_col] > 0
).all(), "There are non-positive 'Close' prices in the data after preprocessing."

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


data["MA20"] = data[close_col].rolling(window=20).mean()
data["RSI"] = compute_RSI(data[close_col], window=14)
data["MACD"] = compute_MACD(data[close_col])

print(data[["Close", "MA20", "RSI", "MACD"]].head(20))

# Fundamental Data


# Function to fetch fundamental data
def get_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    fundamental_data = {}
    # Get PE ratio
    try:
        pe_ratio = stock.info.get("trailingPE", np.nan)
    except KeyError:
        pe_ratio = np.nan
    fundamental_data["PE_ratio"] = pe_ratio
    # Get Price to Book ratio
    try:
        pb_ratio = stock.info.get("priceToBook", np.nan)
    except KeyError:
        pb_ratio = np.nan
    fundamental_data["PB_ratio"] = pb_ratio
    # Get Dividend Yield
    try:
        dividend_yield = stock.info.get("dividendYield", np.nan)
    except KeyError:
        dividend_yield = np.nan
    fundamental_data["Dividend_Yield"] = dividend_yield
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
features = ["Close", "MA20", "RSI", "MACD", "PE_ratio", "PB_ratio", "Dividend_Yield"]

# Update 'Close' in features to match the actual column name
features[features.index("Close")] = close_col

# Remove 'Dividend_Yield' if it's entirely NaN
if data["Dividend_Yield"].isna().all():
    data.drop(columns=["Dividend_Yield"], inplace=True)
    features.remove("Dividend_Yield")

# Remove 'PE_ratio' and 'PB_ratio' if they are entirely NaN
for col in ["PE_ratio", "PB_ratio"]:
    if data[col].isna().all():
        data.drop(columns=[col], inplace=True)
        features.remove(col)
    else:
        data[col].fillna(data[col].mean(), inplace=True)

# Drop initial rows with NaNs in technical indicators
data.dropna(subset=["MA20", "RSI", "MACD"], inplace=True)

# Now drop any remaining rows with NaNs in the features
data.dropna(subset=features, inplace=True)

# Reset index after dropping rows
data.reset_index(drop=True, inplace=True)

# Keep a copy of the original 'Close' price
data["Close_unscaled"] = data[close_col]

# Split the Data into Training and Testing Sets
split_index = int(len(data) * 0.8)  # 80% training, 20% testing

# Split the data
train_data = data.iloc[:split_index].reset_index(drop=True)
test_data = data.iloc[split_index:].reset_index(drop=True)

# Prepare Data for LSTM Model on the Training Set

# Set the target variable 'y' as the next day's 'Close' price in training data
train_data["Target"] = train_data["Close_unscaled"].shift(-1)

# Remove last row with NaN in 'Target'
train_data.dropna(subset=["Target"], inplace=True)

# Feature Scaling for LSTM
lstm_scaler = StandardScaler()
lstm_scaled_data = lstm_scaler.fit_transform(train_data[["Close_unscaled", "Target"]])


# Prepare sequences
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length, 0])
        y.append(data[i + seq_length, 1])
    return np.array(X), np.array(y)


seq_length = 60  # Sequence length for LSTM
X_lstm, y_lstm = create_sequences(lstm_scaled_data, seq_length)

# Reshape X_lstm for LSTM input
X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

# Check if LSTM model exists
lstm_model_path = "./TrainedModels/LSTM&PPO_Gold.h5"
if os.path.exists(lstm_model_path):
    print("Loading existing LSTM model...")
    lstm_model = load_model(lstm_model_path)
else:
    lstm_model = Sequential()
    lstm_model.add(LSTM(128, return_sequences=True, input_shape=(X_lstm.shape[1], 1)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(64))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer="adam", loss="mean_squared_error")

    # Use early stopping
    early_stopping = EarlyStopping(
        monitor="loss", patience=10, restore_best_weights=True
    )

    # Train model
    history = lstm_model.fit(
        X_lstm, y_lstm, epochs=100, batch_size=32, callbacks=[early_stopping]
    )

    # Save the trained LSTM model
    lstm_model.save(lstm_model_path)
    print("LSTM model saved to disk.")

# Generate LSTM Predictions on Training Data

# Predict on the training data
train_full_predictions = lstm_model.predict(X_lstm)

# Prepare LSTM predictions as a feature
# Initialize lstm_predictions with NaNs
train_lstm_predictions = np.empty((len(train_data)))
train_lstm_predictions[:] = np.nan

# Inverse transform the predictions
train_inverse_predictions = lstm_scaler.inverse_transform(
    np.concatenate(
        (lstm_scaled_data[seq_length:, 0].reshape(-1, 1), train_full_predictions),
        axis=1,
    )
)[:, 1]

train_lstm_predictions[seq_length:] = train_inverse_predictions

# Add LSTM predictions to the training DataFrame
train_data["LSTM_Prediction_unscaled"] = train_lstm_predictions

# Drop any rows with NaN in 'LSTM_Prediction_unscaled'
train_data.dropna(subset=["LSTM_Prediction_unscaled"], inplace=True)

# Reset index after dropping rows
train_data.reset_index(drop=True, inplace=True)

# Prepare Test Data for LSTM Predictions

# Set the target variable 'y' as the next day's 'Close' price in test data
test_data["Target"] = test_data["Close_unscaled"].shift(-1)

# Remove last row with NaN in 'Target'
test_data.dropna(subset=["Target"], inplace=True)

# Feature Scaling for LSTM using the same scaler
test_lstm_scaled_data = lstm_scaler.transform(test_data[["Close_unscaled", "Target"]])

# Prepare sequences
X_test_lstm, y_test_lstm = create_sequences(test_lstm_scaled_data, seq_length)

# Reshape X_test_lstm for LSTM input
X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))

# Generate LSTM Predictions on Testing Data

# Predict on the test data
test_full_predictions = lstm_model.predict(X_test_lstm)

# Prepare LSTM predictions as a feature
# Initialize lstm_predictions with NaNs
test_lstm_predictions = np.empty((len(test_data)))
test_lstm_predictions[:] = np.nan

# Inverse transform the predictions
test_inverse_predictions = lstm_scaler.inverse_transform(
    np.concatenate(
        (test_lstm_scaled_data[seq_length:, 0].reshape(-1, 1), test_full_predictions),
        axis=1,
    )
)[:, 1]

# Assign predictions to lstm_predictions from index 'seq_length' onward
test_lstm_predictions[seq_length:] = test_inverse_predictions

# Add LSTM predictions to the testing DataFrame
test_data["LSTM_Prediction_unscaled"] = test_lstm_predictions

# Drop any rows with NaN in 'LSTM_Prediction_unscaled'
test_data.dropna(subset=["LSTM_Prediction_unscaled"], inplace=True)

# Reset index after dropping rows
test_data.reset_index(drop=True, inplace=True)

# Combine the data
combined_data = pd.concat([train_data, test_data], ignore_index=True)

# Keep unscaled versions for environment calculations
combined_data["Close_unscaled"] = combined_data["Close_unscaled"]
combined_data["LSTM_Prediction_unscaled"] = combined_data["LSTM_Prediction_unscaled"]

# Rename 'LSTM_Prediction_unscaled' to 'LSTM_Prediction' for scaling
combined_data.rename(
    columns={"LSTM_Prediction_unscaled": "LSTM_Prediction"}, inplace=True
)

# Update features list to include 'LSTM_Prediction'
features.append("LSTM_Prediction")

# Define features to scale (including 'Close' and 'LSTM_Prediction')
features_to_scale = features.copy()

# Separate features that need special scaling
robust_features = ["MACD"]
standard_features = [feat for feat in features_to_scale if feat not in robust_features]

# Feature Scaling

# Apply StandardScaler to standard_features
scaler_standard = StandardScaler()
combined_data[standard_features] = scaler_standard.fit_transform(
    combined_data[standard_features]
)

# Apply RobustScaler to robust_features
scaler_robust = RobustScaler()
combined_data[robust_features] = scaler_robust.fit_transform(
    combined_data[robust_features]
)

# Custom Trading Environment for Reinforcement Learning


class CustomTradingEnv(gym.Env):
    """A custom trading environment for reinforcement learning"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, df, training=True):
        super(CustomTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.df_total_steps = len(self.df) - 1
        self.training = training

        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        obs_low = np.full(len(features), -np.inf)
        obs_high = np.full(len(features), np.inf)

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # Initialize state
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
        obs = self.df.iloc[self.current_step][features].astype("float32").values
        return obs

    def step(self, action):
        current_price = float(self.df.iloc[self.current_step]["Close_unscaled"])

        done = False

        if action == 0:  # Hold
            pass

        elif action == 1:  # Buy
            max_shares = self.balance // (
                current_price * (1 + self.transaction_fee_rate)
            )
            if max_shares > 0:
                total_cost = (
                    max_shares * current_price * (1 + self.transaction_fee_rate)
                )
                self.balance -= total_cost
                prev_shares = self.shares_held
                self.shares_held += max_shares
                if prev_shares == 0:
                    self.cost_basis = current_price * (1 + self.transaction_fee_rate)
                else:
                    self.cost_basis = (
                        (self.cost_basis * prev_shares)
                        + (current_price * (1 + self.transaction_fee_rate) * max_shares)
                    ) / self.shares_held

        elif action == 2:  # Sell
            if self.shares_held > 0:
                total_proceeds = (
                    self.shares_held * current_price * (1 - self.transaction_fee_rate)
                )
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
        daily_return = (
            self.net_worth - self.initial_net_worth
        ) / self.initial_net_worth
        self.returns.append(daily_return)

        # Return step information
        obs = (
            self._next_observation()
            if not done
            else np.zeros(self.observation_space.shape)
        )
        info = {}
        return obs, reward, done, info

    def render(self, mode="human", close=False):
        profit = self.net_worth - self.initial_net_worth
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance}")
        print(f"Shares held: {self.shares_held}")
        print(f"Net worth: {self.net_worth}")
        print(f"Profit: {profit}")

    def get_portfolio_performance(self):
        total_return = (
            self.net_worth - self.initial_net_worth
        ) / self.initial_net_worth
        max_drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
        returns = pd.Series(self.returns)
        sharpe_ratio = (
            returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        )
        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
        }


# Initialize Environment and Agent

# Prepare data for the environment
# Training Data
train_env_data = combined_data.iloc[: len(train_data)].reset_index(drop=True)

# Initialize environment with training data
train_env = CustomTradingEnv(train_env_data, training=True)

# Path to save/load the DRL agent
agent_model_path = "./TrainedModels/LSTM&PPO_Gold.zip"

if os.path.exists(agent_model_path):
    # Load existing agent
    print("Loading existing DRL agent...")
    agent = PPO.load(agent_model_path, env=train_env)
else:
    # Initialize agent
    agent = PPO("MlpPolicy", train_env, verbose=1)

    # Train agent
    agent.learn(total_timesteps=300000)

    # Save the trained agent
    agent.save(agent_model_path)
    print("DRL agent saved to disk.")

# Testing Data
test_env_data = combined_data.iloc[len(train_data) :].reset_index(drop=True)

# Initialize environment with testing data
test_env = CustomTradingEnv(test_env_data, training=False)

# Backtesting and Evaluation

# Reset environment
obs = test_env.reset()

# Variables to track performance
net_worths = [test_env.net_worth]
balances = [test_env.balance]
held_shares = [test_env.shares_held]
actions = []
prices = []
profits = []

# Run the agent in the environment
while True:
    # Record the current price before stepping
    current_price = float(test_env.df.iloc[test_env.current_step]["Close_unscaled"])
    prices.append(current_price)

    action, _states = agent.predict(obs)
    action = int(action)
    obs, reward, done, info = test_env.step(action)

    # Record the variables
    net_worths.append(test_env.net_worth)
    balances.append(test_env.balance)
    held_shares.append(test_env.shares_held)
    actions.append(action)
    profits.append(test_env.net_worth - test_env.initial_net_worth)

    if done:
        break

# Evaluate performance
portfolio_performance = test_env.get_portfolio_performance()
print(f"Total Return: {portfolio_performance['total_return'] * 100:.2f}%")
print(f"Maximum Drawdown: {portfolio_performance['max_drawdown'] * 100:.2f}%")
print(f"Sharpe Ratio: {portfolio_performance['sharpe_ratio']:.2f}")

# Plot Net Worth Over Time
plt.figure(figsize=(12, 6))
plt.plot(net_worths, label="Net Worth")
plt.title("Net Worth Over Time")
plt.xlabel("Time Step")
plt.ylabel("Net Worth ($)")
plt.legend()
plt.show()

# Plot Balance and Equity Curve
plt.figure(figsize=(12, 6))
plt.plot(balances, label="Balance")
equity = [net_worths[i] - balances[i] for i in range(len(balances))]
plt.plot(equity, label="Equity in Shares")
plt.title("Balance and Equity Over Time")
plt.xlabel("Time Step")
plt.ylabel("Amount ($)")
plt.legend()
plt.show()

# Plot Actions Over Time
action_dict = {0: "Hold", 1: "Buy", 2: "Sell"}
action_names = [action_dict[a] for a in actions]

plt.figure(figsize=(12, 6))
plt.plot(prices, label="Price")
buy_signals = [prices[i] if actions[i] == 1 else np.nan for i in range(len(actions))]
sell_signals = [prices[i] if actions[i] == 2 else np.nan for i in range(len(actions))]
plt.scatter(range(len(actions)), buy_signals, marker="^", color="g", label="Buy Signal")
plt.scatter(
    range(len(actions)), sell_signals, marker="v", color="r", label="Sell Signal"
)
plt.title("Trading Actions Over Time")
plt.xlabel("Time Step")
plt.ylabel("Price ($)")
plt.legend()
plt.show()

# Plot Returns Distribution
returns = pd.Series(test_env.returns)
plt.figure(figsize=(10, 5))
plt.hist(returns, bins=50, edgecolor="black")
plt.title("Distribution of Daily Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.show()

# Compare with Buy-and-Hold Strategy
transaction_fee_rate = 0.01
initial_price = prices[0]

initial_investment_after_fee = test_env.initial_net_worth * (1 - transaction_fee_rate)
num_shares = initial_investment_after_fee / initial_price

buy_and_hold_net_worth = []
for price in prices:
    net_worth = num_shares * price
    buy_and_hold_net_worth.append(net_worth)

plt.figure(figsize=(12, 6))
plt.plot(net_worths, label="Agent Net Worth")
plt.plot(buy_and_hold_net_worth, label="Buy and Hold Net Worth")
plt.title("Agent vs Buy and Hold Strategy")
plt.xlabel("Time Step")
plt.ylabel("Net Worth ($)")
plt.legend()
plt.show()
