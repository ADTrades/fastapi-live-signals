import pandas as pd
import numpy as np
import joblib
import requests
import tensorflow as tf
from fastapi import FastAPI, WebSocket
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import asyncio
import yfinance as yf

app = FastAPI()

# Load trained models
xgb_model = joblib.load("xgb_model.pkl")
lgbm_model = joblib.load("lgbm_model.pkl")
rf_model = joblib.load("rf_model.pkl")
lstm_model = tf.keras.models.load_model("lstm_model.keras")

# Load the scaler
# scaler = joblib.load("scaler.pkl")

# Define stock symbol and API URL
STOCK_SYMBOL = "ICICIBANK.NS"
API_URL = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={STOCK_SYMBOL}"

# WebSocket connections
clients = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            await send_live_signal()
            await asyncio.sleep(60)  # Fetch every 60 seconds
    except Exception as e:
        print(f"Client disconnected: {e}")
        clients.remove(websocket)

def get_live_stock_data(stock):
    try:
        df = yf.download(stock, interval="1h", period="30d")
        print(f"Downloaded Data: \n{df.tail()}")  # Debug print

        if df.empty:
            return None  # No data available

        # Reset MultiIndex column names (flatten them)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        latest_data = df.iloc[-1]  # Get the last row

        # Convert the latest row to a dictionary
        latest_data_dict = latest_data.to_dict()

        return {
            "STOCK_SYMBOL": stock,
            "latest_data": latest_data
        }

    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return None

def preprocess_data(live_data, prev_data):
    """ Apply feature engineering and scaling to live data. """
    df = pd.DataFrame([live_data])
    
    # Add previous close prices
    df["Close_1"] = prev_data["Close"]
    df["Close_2"] = prev_data["Close_1"]

    # Bollinger Bands
    window = 20
    prev_data["SMA_20"] = prev_data["Close"].rolling(window=window).mean().iloc[-1]
    prev_data["STD_20"] = prev_data["Close"].rolling(window=window).std().iloc[-1]
    df["Upper_Band"] = prev_data["SMA_20"] + (prev_data["STD_20"] * 2)
    df["Lower_Band"] = prev_data["SMA_20"] - (prev_data["STD_20"] * 2)

    # Stochastic Oscillator
    df["14-high"] = max(prev_data["High"].tail(14))
    df["14-low"] = min(prev_data["Low"].tail(14))
    df["%K"] = (df["Close"] - df["14-low"]) / (df["14-high"] - df["14-low"]) * 100
    df["%D"] = prev_data["%K"].rolling(3).mean().iloc[-1]

    # OBV
    df["OBV"] = prev_data["OBV"].iloc[-1] + np.sign(df["Close"].iloc[0] - prev_data["Close"].iloc[-1]) * df["Volume"].iloc[0]

    # Select features
    features = ['Close', 'High', 'Low', 'Open', 'Volume', 'Close_1', 'Close_2', 'Upper_Band', 'Lower_Band', '%K', '%D', 'OBV']
    df = df[features]

    # Apply scaling
    df_scaled = scaler.transform(df)
    
    return df_scaled, df

async def send_live_signal():
    """ Predict price change and send buy/sell signals to clients. """
    live_data = get_live_stock_data(STOCK_SYMBOL)  # Pass stock symbol

    # Check if fetching data failed
    if isinstance(live_data, dict) and "error" in live_data:
        print(f"Error fetching stock data: {live_data['error']}")
        return

    # Load historical data to compute rolling features
    historical_df = pd.read_csv("C:/Users/Admin/Intraday/Data/Merged_ICICIBANK.NS.csv", parse_dates=["Date"])
    historical_df.set_index("Date", inplace=True)
    historical_df.fillna(method="bfill", inplace=True)

    # Ensure timestamps are unique
    historical_df = historical_df[~historical_df.index.duplicated(keep='last')]

    # Preprocess live data
    X_live, df_live = preprocess_data(live_data, historical_df)

    # Make predictions
    pred_xgb = xgb_model.predict(X_live)[0]
    pred_lgbm = lgbm_model.predict(X_live)[0]
    pred_rf = rf_model.predict(X_live)[0]

    # LSTM Prediction
    X_live_lstm = np.reshape(X_live, (X_live.shape[0], X_live.shape[1], 1))
    pred_lstm = lstm_model.predict(X_live_lstm).flatten()[0]

    # Average Prediction
    predicted_close = np.mean([pred_xgb, pred_lgbm, pred_rf, pred_lstm])
    price_change = predicted_close - live_data["Close"]

    # Determine Buy/Sell Signal
    if price_change >= 50:
        signal = "BUY"
    elif price_change <= -50:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Format response
    response = {
        "Date": live_data["Date"].strftime("%Y-%m-%d %H:%M:%S"),
        "Current_Close": live_data["Close"],
        "Predicted_Close": predicted_close,
        "Price_Change": price_change,
        "Signal": signal
    }

    print(response)  # Log in terminal

    # Send signal to all connected WebSocket clients
    for client in clients:
        await client.send_json(response)

@app.get("/")
def read_root():
    return {"message": "Live Stock Signal API Running!"}

@app.get("/predict")
def get_prediction(stock: str):
    live_df = get_live_stock_data(stock)
    
    if live_df is None:
        return {"error": "Failed to fetch live stock data"}
    
    # If it's a dict, wrap in DataFrame
    if isinstance(live_df, dict):
        live_df = pd.DataFrame([live_df])
    
    if live_df.empty:
        return {"error": "Live data is empty"}
    
    # Ensure timestamps are unique
    live_df = live_df[~live_df.index.duplicated(keep='last')]

    return live_df

    # Load historical data
    historical_df = pd.read_csv("C:/Users/Admin/Intraday/Data/Merged_ICICIBANK.NS.csv", parse_dates=["Date"])
    print(historical_df.index.duplicated().sum())  # Count duplicate indexes
    #print(live_data.index.duplicated().sum())
    historical_df.set_index("Date", inplace=True)
    historical_df.bfill(inplace=True)

    # Preprocess live data
    # Ensure timestamps are unique
    live_df = live_df[~live_df.index.duplicated(keep='last')]
    X_live, df_live = preprocess_data(live_data, historical_df)

    # Make predictions
    pred_xgb = xgb_model.predict(X_live)[0]
    pred_lgbm = lgbm_model.predict(X_live)[0]
    pred_rf = rf_model.predict(X_live)[0]

    # LSTM Prediction
    X_live_lstm = np.reshape(X_live, (X_live.shape[0], X_live.shape[1], 1))
    pred_lstm = lstm_model.predict(X_live_lstm).flatten()[0]

    # Average Prediction
    predicted_close = np.mean([pred_xgb, pred_lgbm, pred_rf, pred_lstm])
    price_change = predicted_close - live_data["Close"]

    # Determine Buy/Sell Signal
    if price_change >= 50:
        signal = "BUY"
    elif price_change <= -50:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Return response
    return {
        "Date": live_data["Date"].strftime("%Y-%m-%d %H:%M:%S"),
        "Current_Close": live_data["Close"],
        "Predicted_Close": predicted_close,
        "Price_Change": price_change,
        "Signal": signal
    }

