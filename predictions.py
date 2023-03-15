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
price_delta = 1.015
num_intervals = 12
timezone = pytz.timezone('Europe/Warsaw')

url = f"https://data.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={1000}"
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

print("\nData has been successfully downloaded\n")

df = pd.DataFrame(data,
                  columns=["Open time", "Open", "High", "Low", "Close", "Volume", "Close time", "Quote asset volume",
                           "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"])

df["Open time"] = pd.to_datetime(df["Open time"], unit='ms').dt.tz_localize(pytz.utc).dt.tz_convert(timezone)
df["Close time"] = pd.to_datetime(df["Close time"], unit='ms').dt.tz_localize(pytz.utc).dt.tz_convert(timezone)

df = df.drop(df.index[-1])
entry_price = pd.to_numeric(df.iloc[-1]['Close'])

df.to_csv('prediction_data.csv', index=False)

last_datetime = pd.to_datetime(df.iloc[-1]['Close time']) + datetime.timedelta(seconds=1)
last_datetime = last_datetime.replace(microsecond=0, tzinfo=None)

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

pca = load('IA-NN-models/pca.joblib')
scaler = load('IA-NN-models/scaler.joblib')

x_pred_scaled = scaler.transform(x_pred)
x_pred_PCA = pca.transform(x_pred_scaled)



if os.path.exists('IA-NN-models'):
    for file_name in os.listdir('IA-NN-models'):
        if file_name.endswith('.h5'):
            model_path = os.path.join('IA-NN-models', file_name)
            model = tf.keras.models.load_model(model_path)
            y_pred = model.predict(x_pred_PCA)
            print(f"Prediction from {file_name}: {y_pred[0].item():.4f}")
else:
    print("\nError: Model directory not found")


print(
    f"\nPrediction is valid at: {last_datetime:%Y-%m-%d %H:%M:%S}")
print(
    f"Sell within time frame: {last_datetime:%Y-%m-%d %H:%M:%S} - {(last_datetime + total_delta):%Y-%m-%d %H:%M:%S}")

url_price_USDT = "https://data.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
response = requests.get(url_price_USDT).json()
price_USDT = float(response["price"])

url_price_BUSD = "https://data.binance.com/api/v3/ticker/price?symbol=BTCBUSD"
response = requests.get(url_price_BUSD).json()
price_BUSD = float(response["price"])

print(f"\nTarget price USDT: {round(price_USDT * price_delta, 2)}")
print(f"Target price BUSD: {round(price_BUSD * price_delta, 2)}")
