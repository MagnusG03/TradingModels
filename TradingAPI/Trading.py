import os
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import pickle
import json
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import oandapyV20
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.pricing as pricing
from oandapyV20.exceptions import V20Error

# Suppress TensorFlow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# OANDA API credentials (replace with your own)
OANDA_API_TOKEN = '6ce0ac89b1280a778f6ac042d371886d-47fe0057cb059a6bde90da0cf91be53f'
OANDA_ACCOUNT_ID = '101-004-30344862-001'

# Initialize OANDA API client
client = oandapyV20.API(access_token=OANDA_API_TOKEN)

# List of commodities
symbols = ['NG=F', 'CL=F', 'GC=F', 'SI=F', 'HG=F']

# Mapping between Yahoo Finance symbols and OANDA instruments
symbol_to_instrument = {
    'NG=F': 'NATGAS_USD',
    'CL=F': 'WTICO_USD',
    'GC=F': 'XAU_USD',
    'SI=F': 'XAG_USD',
    'HG=F': 'XCU_USD',  # Corrected instrument code for Copper
}

save_dir = 'TrainedModels/TradingAPI'
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Function to get account balance
def get_account_balance(client, account_id):
    r = accounts.AccountDetails(accountID=account_id)
    try:
        response = client.request(r)
        balance = float(response['account']['balance'])
        return balance
    except V20Error as e:
        print(f"Error fetching account balance: {e}")
        return None

# Function to get current price from OANDA
def get_current_price(client, account_id, instrument):
    params = {
        "instruments": instrument
    }
    r = pricing.PricingInfo(accountID=account_id, params=params)
    try:
        rv = client.request(r)
        # Extract the last bid and ask price
        prices = rv['prices'][0]
        bid = float(prices['bids'][0]['price'])
        ask = float(prices['asks'][0]['price'])
        mid_price = (bid + ask) / 2
        return mid_price
    except V20Error as e:
        print(f"Error fetching current price for {instrument}: {e}")
        return None

# Main trading loop
try:
    while True:
        for symbol in symbols:
            print(f"\nProcessing {symbol}...")
            instrument = symbol_to_instrument.get(symbol)
            if not instrument:
                print(f"No OANDA instrument mapping for symbol {symbol}. Skipping.")
                continue

            # Set the date ranges
            end_date = datetime.datetime.today()
            start_date_daily = end_date - datetime.timedelta(days=365 * 10)  # 10 years
            start_date_hourly = end_date - datetime.timedelta(days=700)      # 700 days
            start_date_2min = end_date - datetime.timedelta(days=50)         # 50 days

            # Download daily data
            daily_data = yf.download(
                symbol,
                start=start_date_daily.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            if daily_data.empty:
                print(f"No daily data available for {symbol}. Skipping.")
                continue

            # Flatten MultiIndex columns if present in daily_data
            if isinstance(daily_data.columns, pd.MultiIndex):
                daily_data.columns = daily_data.columns.get_level_values(0)
            daily_data.reset_index(inplace=True)
            daily_data.set_index('Date', inplace=True)

            # Download hourly data
            hourly_data = yf.download(
                symbol,
                start=start_date_hourly.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1h'
            )
            if hourly_data.empty:
                print(f"No hourly data available for {symbol}. Cannot calculate 'Volatility_hourly'.")
                hourly_data = None
            else:
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
                symbol,
                start=start_date_2min.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='2m'
            )
            if data_2min.empty:
                print(f"No 2-minute data available for {symbol}. Cannot calculate 'Volatility_2min'.")
                data_2min = None
            else:
                if isinstance(data_2min.columns, pd.MultiIndex):
                    data_2min.columns = data_2min.columns.get_level_values(0)
                data_2min.reset_index(inplace=True)
                if 'Datetime' in data_2min.columns:
                    data_2min.set_index('Datetime', inplace=True)
                elif 'Date' in data_2min.columns:
                    data_2min.set_index('Date', inplace=True)
                else:
                    print("Datetime column not found in data_2min.")

            # Calculate volatility from higher-frequency data
            def calculate_volatility(high_freq_data):
                if high_freq_data is None or high_freq_data.empty:
                    return pd.DataFrame(columns=['Date', 'Volatility'])
                if 'Close' not in high_freq_data.columns:
                    print("No 'Close' column available for volatility calculation.")
                    return pd.DataFrame(columns=['Date', 'Volatility'])
                returns = high_freq_data['Close'].pct_change()
                volatility = returns.resample('D').std()
                volatility = volatility.reset_index()
                volatility.columns = ['Date', 'Volatility']
                return volatility

            # Calculate volatility for hourly and 2-minute data
            volatility_hourly = calculate_volatility(hourly_data)
            volatility_hourly.rename(columns={'Volatility': 'Volatility_hourly'}, inplace=True)
            volatility_2min = calculate_volatility(data_2min)
            volatility_2min.rename(columns={'Volatility': 'Volatility_2min'}, inplace=True)

            # Merge volatility features into daily_data
            daily_data = daily_data.reset_index()
            daily_data = daily_data.merge(volatility_hourly, on='Date', how='left')
            daily_data = daily_data.merge(volatility_2min, on='Date', how='left')
            daily_data.set_index('Date', inplace=True)

            # Fill missing volatility values
            daily_data['Volatility_hourly'] = daily_data['Volatility_hourly'].ffill()
            daily_data['Volatility_2min'] = daily_data['Volatility_2min'].ffill()

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
                print(f"Missing features in daily_data for {symbol}: {missing_features}")
                features = [feat for feat in features if feat in daily_data.columns]

            # Save the feature list for later use
            feature_filename = os.path.join(save_dir, f'features_{symbol}.json')
            with open(feature_filename, 'w') as f:
                json.dump(features, f)

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(daily_data[features])
            scaled_df = pd.DataFrame(scaled_data, columns=features, index=daily_data.index)
            daily_data[features] = scaled_df

            # Save the scaler
            scaler_filename = os.path.join(save_dir, f'scaler_{symbol}.pkl')
            with open(scaler_filename, 'wb') as f:
                pickle.dump(scaler, f)

            X = daily_data[features]
            y = daily_data['Close'].shift(-1)
            X = X[:-1]
            y = y[:-1]

            def create_sequences(X, y, time_steps=10):
                Xs, ys, indices = [], [], []
                for i in range(len(X) - time_steps):
                    Xs.append(X.iloc[i:(i + time_steps)].values)
                    ys.append(y.iloc[i + time_steps])
                    indices.append(y.index[i + time_steps])
                return np.array(Xs), np.array(ys), indices

            time_steps = 10
            X_seq, y_seq, indices_seq = create_sequences(X, y, time_steps)

            # Check if we have enough data
            if len(X_seq) == 0:
                print(f"Not enough data to create sequences for {symbol}. Skipping.")
                continue

            # Split the data
            split = int(0.8 * len(X_seq))
            X_train, X_test = X_seq[:split], X_seq[split:]
            y_train, y_test = y_seq[:split], y_seq[split:]
            indices_train, indices_test = indices_seq[:split], indices_seq[split:]

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
            model.fit(
                X_train, y_train,
                epochs=500,
                batch_size=32,
                validation_split=0.1,
                callbacks=[early_stop],
                verbose=0  # Suppress training output for clarity
            )

            # Save the model
            model_filename = os.path.join(save_dir, f'model_{symbol.replace("=F", "")}.h5')
            model.save(model_filename)
            print(f"Model for {symbol} retrained and saved as {model_filename}")

            # Use the last 'time_steps' data points for prediction
            X_input = scaled_df[-time_steps:].values.reshape(1, time_steps, -1)

            # Make prediction
            prediction = model.predict(X_input)

            # Inverse transform the prediction
            # Prepare data for inverse scaling
            last_scaled_data = scaled_df[-1:].copy()
            col_idx = last_scaled_data.columns.get_loc('Close')
            last_scaled_data.iloc[0, col_idx] = prediction[0][0]
            prediction_inv = scaler.inverse_transform(last_scaled_data)[0][col_idx]

            # Get the current price from OANDA
            current_price = get_current_price(client, OANDA_ACCOUNT_ID, instrument)
            if current_price is None:
                print(f"Could not retrieve current price for {symbol}. Skipping.")
                continue

            # Generate signal
            if prediction_inv > current_price * 1.035:
                signal = 'BUY'
            elif prediction_inv < current_price * 0.80:
                signal = 'SELL'
            else:
                signal = 'HOLD'

            print(f"{symbol} prediction: {prediction_inv:.2f}, current price: {current_price:.2f}, signal: {signal}")

            if signal in ['BUY', 'SELL']:
                # Get account balance
                balance = get_account_balance(client, OANDA_ACCOUNT_ID)
                if balance is None:
                    print(f"Could not retrieve account balance. Skipping {symbol}.")
                    continue

                # Calculate required margin or cost for the trade
                buy_amount = 1000  # Fixed amount to buy or sell
                units = int(np.floor(buy_amount / current_price))
                if units == 0:
                    units = 1
                units = units if signal == 'BUY' else -units  # Positive for buy, negative for sell
                required_margin = abs(units) * current_price

                # Check if balance is sufficient
                if balance < required_margin:
                    print(f"Not enough balance to execute trade for {symbol}. Available balance: {balance}, required: {required_margin}")
                    continue

                # Place order via OANDA API
                data = {
                    "order": {
                        "instrument": instrument,
                        "units": str(units),
                        "type": "MARKET",
                        "positionFill": "DEFAULT"
                    }
                }
                r = orders.OrderCreate(OANDA_ACCOUNT_ID, data=data)
                try:
                    rv = client.request(r)
                    print(f"Order placed for {symbol}: {rv}")
                except V20Error as e:
                    print(f"Error placing order for {symbol}: {e}")
            else:
                print(f"No action for {symbol}.")

except KeyboardInterrupt:
    print("Script interrupted by user. Exiting.")
