import datetime
import os
import time

import numpy as np
import pandas as pd
import pytz
import requests
import tensorflow as tf
from joblib import load
from requests.exceptions import RequestException
from ta import add_all_ta_features

symbol = "BTCUSDT"
interval = "1h"
price_delta = 1.01
num_intervals = 12
timezone = pytz.timezone('Europe/Warsaw')

url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={1000}"
data = []

end_time = int(time.time() * 1000)
start_time = int(datetime.datetime(2021, 2, 28).timestamp() * 1000)

while start_time < end_time:
    try:
        response = requests.get(url + f"&startTime={start_time}&endTime={end_time}")
        response.raise_for_status()
        klines = response.json()
        data += klines
        start_time = klines[-1][6] + 1
    except RequestException as e:
        print("Error occurred:", e)
        time.sleep(60)

df = pd.DataFrame(data,
                  columns=["Open time", "Open", "High", "Low", "Close", "Volume", "Close time", "Quote asset volume",
                           "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"])

df["Open time"] = pd.to_datetime(df["Open time"], unit='ms').dt.tz_localize(pytz.utc).dt.tz_convert(timezone)
df["Close time"] = pd.to_datetime(df["Close time"], unit='ms').dt.tz_localize(pytz.utc).dt.tz_convert(timezone)

df = df.drop(df.index[-1])
entry_price = pd.to_numeric(df.iloc[-1]['Close'])

df.to_csv('prediction_data.csv', index=False)

print("\nData has been successfully downloaded\n")

last_datetime = pd.to_datetime(df.iloc[-1]['Close time'])
last_datetime = last_datetime.replace(microsecond=0, tzinfo=None)
valid_until = last_datetime + pd.Timedelta(interval)
valid_until = valid_until.replace(microsecond=0, tzinfo=None)

interval_units = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks", "M": "months", "y": "years"}
interval_unit = interval[-1]
interval_num = int(interval[:-1])
total_delta = num_intervals * datetime.timedelta(**{interval_units[interval_unit]: interval_num})

df = pd.read_csv('prediction_data.csv')

df = df.sort_values(by="Open time")
df.reset_index(drop=True, inplace=True)

np.seterr(divide='ignore', invalid='ignore')
df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

df.drop(
    ['High', 'Open', 'Close', 'Low', 'Volume', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
     'Taker buy quote asset volume', 'Ignore', 'Open time', 'Close time'],
    inplace=True, axis=1)

df.dropna(inplace=True)

x_pred = df.iloc[df.shape[0] - 1:, :]

pca = load('prediction_model/pca.joblib')
scaler = load('prediction_model/scaler.joblib')

x_pred_scaled = scaler.transform(x_pred)
x_pred_PCA = pca.transform(x_pred_scaled)

url_price = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
response = requests.get(url_price).json()
price = float(response["price"])

if os.path.exists('prediction_model'):
    models = []
    for file_name in os.listdir('prediction_model'):
        if file_name.endswith('.h5'):
            model_path = os.path.join('prediction_model', file_name)
            model = tf.keras.models.load_model(model_path)
            models.append(model)

    for model in models:
        y_pred = model.predict(x_pred_PCA)
        print(f"Prediction: {y_pred[0].item():.4f}")

else:
    print("\nError: Model directory not found")

print(
    f"\nPrediction is valid within the time frame: {last_datetime:%Y-%m-%d %H:%M:%S} - {valid_until:%Y-%m-%d %H:%M:%S}")
print(
    f"Sell within time frame:                    {last_datetime:%Y-%m-%d %H:%M:%S} - {(last_datetime + total_delta):%Y-%m-%d %H:%M:%S}")
print(f"Current BTC/USDT price: {price}")
print(f"Target price (sell): {entry_price * price_delta}")
