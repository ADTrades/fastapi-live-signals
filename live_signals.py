import pandas as pd
import numpy as np
import joblib
import requests
import tensorflow as tf
from fastapi import FastAPI, WebSocket
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import asyncio

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

def get_live_stock_data():
    """ Fetch live stock data from Yahoo Finance. """
    try:
        response = requests.get(API_URL)
        data = response.json()
        quote = data["quoteResponse"]["result"][0]
        
        return {
            "Date": datetime.now(),
            "Close": quote["regularMarketPrice"],
            "High": quote["regularMarketDayHigh"],
            "Low": quote["regularMarketDayLow"],
            "Open": quote["regularMarketOpen"],
            "Volume": quote["regularMarketVolume"],
        }
    except Exception as e:
        print("Error fetching live data:", e)
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
    live_data = get_live_stock_data()
    if not live_data:
        return

    # Load historical data to compute rolling features
    historical_df = pd.read_csv("C:/Users/Admin/Intraday/Data/Merged_ICICIBANK.NS.csv", parse_dates=["Date"])
    historical_df.set_index("Date", inplace=True)
    historical_df.fillna(method="bfill", inplace=True)

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
def get_prediction():
    """ Fetch live data, make predictions, and return the signal. """
    live_data = get_live_stock_data()
    if not live_data:
        return {"error": "Failed to fetch live stock data"}

    # Load historical data
    historical_df = pd.read_csv("C:/Users/Admin/Intraday/Data/Merged_ICICIBANK.NS.csv", parse_dates=["Date"])
    historical_df.set_index("Date", inplace=True)
    historical_df.fillna(method="bfill", inplace=True)

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

    # Return response
    return {
        "Date": live_data["Date"].strftime("%Y-%m-%d %H:%M:%S"),
        "Current_Close": live_data["Close"],
        "Predicted_Close": predicted_close,
        "Price_Change": price_change,
        "Signal": signal
    }

