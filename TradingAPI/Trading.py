import os
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import pickle
import json
import time
from tensorflow.keras.models import load_model
import oandapyV20
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.pricing as pricing  # Import pricing endpoint
from oandapyV20.exceptions import V20Error

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

interval_seconds = 3600  # 3600 seconds = 1 hour

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

try:
    while True:
        for symbol in symbols:
            print(f"\nProcessing {symbol}...")
            instrument = symbol_to_instrument.get(symbol)
            if not instrument:
                print(f"No OANDA instrument mapping for symbol {symbol}. Skipping.")
                continue

            # Load the saved model
            model_filename = os.path.join(save_dir, f'model_{symbol.replace("=F", "")}.h5')
            if not os.path.exists(model_filename):
                print(f"Model file {model_filename} not found. Skipping {symbol}.")
                continue
            model = load_model(model_filename)

            # Load the scaler
            scaler_filename = os.path.join(save_dir, f'scaler_{symbol}.pkl')
            if not os.path.exists(scaler_filename):
                print(f"Scaler file {scaler_filename} not found. Skipping {symbol}.")
                continue
            with open(scaler_filename, 'rb') as f:
                scaler = pickle.load(f)

            # Load the feature list
            feature_filename = os.path.join(save_dir, f'features_{symbol}.json')
            if not os.path.exists(feature_filename):
                print(f"Feature file {feature_filename} not found. Skipping {symbol}.")
                continue
            with open(feature_filename, 'r') as f:
                features = json.load(f)

            # Get the latest data
            end_date = datetime.datetime.today()
            start_date = end_date - datetime.timedelta(days=365)  # Last 1 year
            daily_data = yf.download(
                symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            if daily_data.empty:
                print(f"No daily data available for {symbol}. Skipping.")
                continue

            # Flatten MultiIndex columns if present
            if isinstance(daily_data.columns, pd.MultiIndex):
                daily_data.columns = daily_data.columns.get_level_values(0)
            daily_data.reset_index(inplace=True)
            daily_data.set_index('Date', inplace=True)

            # Download hourly data
            start_date_hourly = end_date - datetime.timedelta(days=719)  # Max 720 days
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
            start_date_2min = end_date - datetime.timedelta(days=59)  # Max 60 days
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

            # Ensure the data has the necessary features
            # Calculate moving averages and RSI
            daily_data['MA10'] = daily_data['Close'].rolling(window=10).mean()
            daily_data['MA50'] = daily_data['Close'].rolling(window=50).mean()

            window_length = 14
            delta = daily_data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window_length).mean()
            loss = -delta.where(delta < 0, 0).rolling(window_length).mean()
            rs = gain / loss
            daily_data['RSI'] = 100 - (100 / (1 + rs))

            daily_data.fillna(0, inplace=True)

            # Check for missing features
            missing_features = [feat for feat in features if feat not in daily_data.columns]
            if missing_features:
                print(f"Missing features in daily_data for {symbol}: {missing_features}")
                continue

            # Ensure that feature columns are in the correct order
            daily_data = daily_data[features]

            # Scale the data using the saved scaler
            scaled_data = scaler.transform(daily_data)
            scaled_df = pd.DataFrame(scaled_data, columns=features, index=daily_data.index)

            # Create sequences (use the same time_steps as in training)
            time_steps = 10
            if len(scaled_df) < time_steps:
                print(f"Not enough data to create sequences for {symbol}. Skipping.")
                continue
            X = scaled_df[-time_steps:].values.reshape(1, time_steps, -1)

            # Make prediction
            prediction = model.predict(X)

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
                units = 10 if signal == 'BUY' else -10  # Positive for buy, negative for sell
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

        print(f"Waiting for {interval_seconds} seconds before the next check...")
        time.sleep(interval_seconds)

except KeyboardInterrupt:
    print("Script interrupted by user. Exiting.")
