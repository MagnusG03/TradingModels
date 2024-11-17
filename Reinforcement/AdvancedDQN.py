import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import random
import os

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
    interval='1d'
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

# Download 2-minute data
data_2min = yf.download(
    'CL=F',
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
        return pd.DataFrame()  # Return empty DataFrame if no columns

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

# Handle missing values and ensure data types are correct
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

# Fill NaN values
daily_data.fillna(0, inplace=True)

# Features for the model
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50', 'RSI', 'Volatility_hourly', 'Volatility_2min']

# Scale the features between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(daily_data[features])
scaled_df = pd.DataFrame(scaled_data, columns=features, index=daily_data.index)
daily_data[features] = scaled_df

# Use the daily_data from the LSTM code
data = daily_data.copy()
data.reset_index(inplace=True)

# Update the Trading Environment
class TradingEnv:
    def __init__(self, data, transaction_fee=0.01):
        self.data = data.reset_index(drop=True)
        self.features = features
        self.current_step = 0
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.transaction_fee = transaction_fee
        self.max_steps = len(self.data) - 1
        self.action_space = [0, 1, 2]
        self.state_size = len(self.features) + 2

        # Initialize buy-and-hold strategy
        self.buy_hold_shares = self.initial_balance / self.data.loc[self.current_step, 'Close']
        self.buy_hold_net_worth = self.buy_hold_shares * self.data.loc[self.current_step, 'Close']

    def reset(self):
        # Reset the environment to initial state
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance

        # Reset buy-and-hold strategy
        self.buy_hold_shares = self.initial_balance / self.data.loc[self.current_step, 'Close']
        self.buy_hold_net_worth = self.buy_hold_shares * self.data.loc[self.current_step, 'Close']

        state = self._get_observation()
        return state

    def _get_observation(self):
        # Get the current state
        obs = self.data.loc[self.current_step, self.features].values.astype(np.float32)
        # Normalize balance and include shares held
        obs = np.append(obs, [self.balance / self.initial_balance, self.shares_held / 1000.0])
        return obs

    def step(self, action):
        current_price = self.data.loc[self.current_step, 'Close']
        if current_price <= 1e-8:
            current_price = 1e-8
        done = False

        # Update buy-and-hold net worth
        self.buy_hold_net_worth = self.buy_hold_shares * current_price

        # Execute action
        if action == 1:  # Buy
            shares_to_buy = self.balance // current_price
            if shares_to_buy > 0:
                total_cost = shares_to_buy * current_price * (1 + self.transaction_fee)
                if self.balance >= total_cost:
                    self.balance -= total_cost
                    self.shares_held += shares_to_buy
        elif action == 2:  # Sell
            if self.shares_held > 0:
                total_revenue = self.shares_held * current_price * (1 - self.transaction_fee)
                self.balance += total_revenue
                self.shares_held = 0

        # Update net worth after action
        self.net_worth = self.balance + self.shares_held * current_price

        # Move to the next step
        self.current_step += 1

        # Calculate reward as the difference between agent's net worth and buy-and-hold net worth
        reward = (self.net_worth - self.buy_hold_net_worth) / self.initial_balance

        # Check if the episode is done
        if self.current_step >= self.max_steps:
            done = True

        next_state = self._get_observation()
        return next_state, reward, done

# Add the ReplayBuffer class
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        # Add experience to buffer
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Sample a batch of experiences
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Build the Deep Q-Network model
def build_model(state_size, action_size):
    model = keras.Sequential([
        layers.Dense(64, input_dim=state_size, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer()
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = build_model(state_size, action_size)
        self.target_model = build_model(state_size, action_size)
        self.update_target_model()
        self.batch_size = 32

    def update_target_model(self):
        # Update target network weights
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        # Decide action based on current state
        if np.random.rand() <= self.epsilon:
            return random.choice([0, 1, 2])
        act_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def replay(self):
        try:
            # Train the model using experiences from the replay buffer
            if len(self.memory) < self.batch_size:
                return

            minibatch = self.memory.sample(self.batch_size)
            states = np.array([e[0] for e in minibatch])
            actions = np.array([e[1] for e in minibatch])
            rewards = np.array([e[2] for e in minibatch])
            next_states = np.array([e[3] for e in minibatch])
            dones = np.array([e[4] for e in minibatch])

            # Predict Q-values for current states
            target = self.model.predict(states, verbose=0)
            # Predict Q-values for next states using target network
            target_next = self.target_model.predict(next_states, verbose=0)

            for i in range(self.batch_size):
                if dones[i]:
                    target[i][actions[i]] = rewards[i]
                else:
                    # Q-learning update rule
                    target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])

            # Train the model
            self.model.fit(states, target, epochs=1, verbose=0)
        except Exception as e:
            print(f"An error occurred during replay: {e}")

# Path to save and load the model
model_path = './TrainedModels/AdvancedDQN_CrudeOil.keras'

# Create training and evaluation environments
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
eval_data = data[train_size:]

train_env = TradingEnv(train_data)
eval_env = TradingEnv(eval_data)

state_size = train_env.state_size
action_size = len(train_env.action_space)

# Check if model exists
model_exists = os.path.exists(model_path)

# Load or initialize the agent
if model_exists:
    print("Loading saved model...")
    try:
        agent = DQNAgent(state_size, action_size)
        agent.model = keras.models.load_model(model_path)
        agent.update_target_model()
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
else:
    print("No saved model found. Initializing a new agent.")
    agent = DQNAgent(state_size, action_size)

# Validate data integrity
print(f"Total data length: {len(data)}")
print(f"Training data length: {len(train_data)}")
print(f"Evaluation data length: {len(eval_data)}")

# Select only numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns

# Check for NaN or Inf values in numeric data
if data[numeric_cols].isnull().values.any():
    print("Data contains NaN values.")
if np.isinf(data[numeric_cols].values).any():
    print("Data contains infinite values.")

# Only train if the model does not exist
if not model_exists:
    # Training the agent
    num_episodes = 500
    update_target_frequency = 5
    reward_threshold = 0

    best_reward = -float('inf')
    patience = 20  # Number of episodes to wait before early stopping
    episodes_without_improvement = 0

    try:
        for e in range(num_episodes):
            state = train_env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, done = train_env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                agent.replay()

            # Update target network
            if e % update_target_frequency == 0:
                agent.update_target_model()

            # Decay exploration rate epsilon
            if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay

            print(f"Episode {e + 1}/{num_episodes}, Total Reward: {total_reward:.4f}, Epsilon: {agent.epsilon:.4f}")

            # Check for improvement
            if total_reward > best_reward:
                best_reward = total_reward
                episodes_without_improvement = 0
                # Save the model if total reward has improved
                agent.model.save(model_path)
                print(f"Model saved successfully with reward: {total_reward:.4f}")
            else:
                episodes_without_improvement += 1

            # Early stopping
            if episodes_without_improvement >= patience:
                print(f"No improvement for {patience} episodes. Early stopping.")
                break
    except Exception as e:
        print(f"An error occurred during training: {e}")

# Test the trained agent
state = eval_env.reset()
done = False
net_worths = []
buy_hold_net_worths = []

while not done:
    action = agent.act(state)
    next_state, reward, done = eval_env.step(action)
    state = next_state
    net_worths.append(eval_env.net_worth)
    buy_hold_net_worths.append(eval_env.buy_hold_net_worth)

# Plot portfolio value over time
plt.figure(figsize=(12, 6))
plt.plot(net_worths, label='Agent Portfolio Value')
plt.plot(buy_hold_net_worths, label='Buy and Hold Portfolio Value')
plt.title('Portfolio Value Comparison')
plt.xlabel('Time Steps')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.show()

# Print final results
print(f'Final Agent Net Worth: ${net_worths[-1]:.2f}')
print(f'Final Buy and Hold Net Worth: ${buy_hold_net_worths[-1]:.2f}')
