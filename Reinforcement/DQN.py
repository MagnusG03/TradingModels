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
from tqdm import tqdm

# Set end and start dates for data retrieval
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=50)

# Download historical prices
gold_data = yf.download(
    "GC=F",
    start=start_date.strftime("%Y-%m-%d"),
    end=end_date.strftime("%Y-%m-%d"),
    interval="2m",
)
gold_data.reset_index(inplace=True)

# Flatten MultiIndex columns
if isinstance(gold_data.columns, pd.MultiIndex):
    gold_data.columns = [
        "_".join(col).strip() if col[1] else col[0] for col in gold_data.columns.values
    ]
else:
    gold_data.columns = [col.strip() for col in gold_data.columns.values]

print("Columns after flattening:", gold_data.columns)

gold_data.rename(
    columns={
        "Datetime_": "Datetime",
        "Adj Close_GC=F": "Adj Close",
        "Close_GC=F": "Close",
        "High_GC=F": "High",
        "Low_GC=F": "Low",
        "Open_GC=F": "Open",
        "Volume_GC=F": "Volume",
    },
    inplace=True,
)

# Verify that 'Close' column exists
if "Close" not in gold_data.columns:
    print("Error: 'Close' column not found in data")
    print("Available columns:", gold_data.columns)
    exit()

# Handle missing values and ensure data types are correct
gold_data.ffill(inplace=True)
gold_data["Close"] = gold_data["Close"].astype(float)

# Calculate moving averages and RSI
gold_data["MA10"] = gold_data["Close"].rolling(window=10).mean()
gold_data["MA50"] = gold_data["Close"].rolling(window=50).mean()

window_length = 14
delta = gold_data["Close"].diff()
gain = delta.where(delta > 0, 0).rolling(window=window_length).mean()
loss = -delta.where(delta < 0, 0).rolling(window=window_length).mean()
rs = gain / loss
gold_data["RSI"] = 100 - (100 / (1 + rs))

# Fill NaN values
gold_data.fillna(0, inplace=True)

# Features for the model
features = ["Open", "High", "Low", "Close", "Volume", "MA10", "MA50", "RSI"]

# Scale the features between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(gold_data[features])
scaled_df = pd.DataFrame(scaled_data, columns=features, index=gold_data.index)
gold_data[features] = scaled_df


# Custom Trading Environment
class TradingEnv:
    def __init__(self, data, transaction_fee=0.01):
        self.data = data.reset_index(drop=True)
        self.features = features
        self.current_step = 0
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.transaction_fee = transaction_fee
        self.max_steps = len(self.data) - 1
        self.action_space = [0, 1, 2]  # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.state_size = len(self.features) + 2

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        state = self._get_observation()
        return state

    def _get_observation(self):
        obs = self.data.loc[self.current_step, self.features].values.astype(np.float32)
        obs = np.append(obs, [self.balance / self.initial_balance, self.shares_held])
        return obs

    def step(self, action):
        current_price = self.data.loc[self.current_step, "Close"]
        if current_price <= 1e-8:
            current_price = 1e-8
        done = False

        # Execute action
        if action == 1:  # Buy
            max_shares_can_buy = self.balance / (
                current_price * (1 + self.transaction_fee)
            )
            shares_to_buy = max_shares_can_buy
            if shares_to_buy > 0:
                total_cost = shares_to_buy * current_price * (1 + self.transaction_fee)
                self.balance -= total_cost
                self.shares_held += shares_to_buy
        elif action == 2:  # Sell
            if self.shares_held > 0:
                total_revenue = (
                    self.shares_held * current_price * (1 - self.transaction_fee)
                )
                self.balance += total_revenue
                self.shares_held = 0.0

        self.current_step += 1

        # Update net worth
        self.net_worth = self.balance + self.shares_held * current_price

        # Calculate reward as a percentage change in net worth
        reward = (self.net_worth - self.initial_balance) / self.initial_balance

        # Check if the episode is done
        if self.current_step >= self.max_steps:
            done = True

        next_state = self._get_observation()
        return next_state, reward, done


# Replay Buffer for experience replay
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# Build the Deep Q-Network model
def build_model(state_size, action_size):
    model = keras.Sequential(
        [
            layers.Dense(64, input_dim=state_size, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(action_size, activation="linear"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer()
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        self.model = build_model(state_size, action_size)
        self.target_model = build_model(state_size, action_size)
        self.update_target_model()
        self.batch_size = 64

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice([0, 1, 2])
        act_values = self.model.predict(state[np.newaxis], verbose=0)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def replay(self):
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
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(
                    target_next[i]
                )

        # Train the model
        self.model.fit(states, target, epochs=1, verbose=0)


# Path to save and load the model
model_path = "./TrainedModels/DQN_Gold.keras"

# Create training and evaluation environments
train_size = int(len(gold_data) * 0.8)
train_data = gold_data[:train_size]
eval_data = gold_data[train_size:]

train_env = TradingEnv(train_data)
eval_env = TradingEnv(eval_data)

state_size = train_env.state_size
action_size = len(train_env.action_space)

# Print dataset sizes
print(f"Total data points: {len(gold_data)}")
print(f"Training data points: {len(train_data)}")
print(f"Evaluation data points: {len(eval_data)}")

# Load or initialize the agent
model_loaded = False
if os.path.exists(model_path):
    print("Loading saved model...")
    agent = DQNAgent(state_size, action_size)
    agent.model = keras.models.load_model(model_path)
    agent.update_target_model()
    agent.epsilon = agent.epsilon_min
    model_loaded = True
else:
    print("No saved model found. Initializing a new agent.")
    agent = DQNAgent(state_size, action_size)

if not model_loaded:
    # Training the agent
    num_episodes = 10  # Reduced number of episodes for quick testing
    update_target_frequency = 5

    best_reward = -float("inf")

    for e in range(num_episodes):
        state = train_env.reset()
        total_reward = 0
        done = False
        step = 0  # Initialize step counter

        with tqdm(
            total=train_env.max_steps,
            desc=f"Episode {e + 1}/{num_episodes}",
            unit="step",
        ) as pbar:
            while not done:
                action = agent.act(state)
                next_state, reward, done = train_env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                step += 1

                agent.replay()

                # Update progress bar
                pbar.update(1)

        if e % update_target_frequency == 0:
            agent.update_target_model()

        # Decay epsilon at the end of each episode
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            agent.epsilon = max(agent.epsilon_min, agent.epsilon)

        print(
            f"Episode {e + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}"
        )

        if total_reward > best_reward:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            agent.model.save(model_path)
            best_reward = total_reward
            print(f"Model saved successfully with reward: {total_reward:.2f}")

    agent.epsilon = agent.epsilon_min

# Test the trained agent
state = eval_env.reset()
done = False
net_worths = []

while not done:
    action = agent.act(state)
    next_state, reward, done = eval_env.step(action)
    state = next_state
    net_worths.append(eval_env.net_worth)

# Buy-and-hold strategy for comparison
buy_hold_net_worths = []
eval_env.reset()
eval_env.balance = 0
initial_price = eval_env.data.loc[0, "Close"]
eval_env.shares_held = (
    eval_env.initial_balance * (1 - eval_env.transaction_fee)
) / initial_price

for step in range(len(eval_env.data)):
    current_price = eval_env.data.loc[step, "Close"]
    net_worth = eval_env.shares_held * current_price
    buy_hold_net_worths.append(net_worth)

# Plot portfolio value over time
plt.figure(figsize=(12, 6))
plt.plot(net_worths, label="Agent Portfolio Value")
plt.plot(buy_hold_net_worths, label="Buy and Hold Portfolio Value")
plt.title("Portfolio Value Comparison")
plt.xlabel("Time Steps")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.show()

# Print final results
print(f"Final Agent Net Worth: ${net_worths[-1]:.2f}")
print(f"Final Buy and Hold Net Worth: ${buy_hold_net_worths[-1]:.2f}")
