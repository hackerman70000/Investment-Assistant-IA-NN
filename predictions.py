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
limit = 1000
tz = pytz.timezone('Europe/Warsaw')
end_time = int(time.time() * 1000)
start_time = int(datetime.datetime(2021, 2, 28).timestamp() * 1000)

url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

data = []

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

df["Open time"] = pd.to_datetime(df["Open time"], unit='ms').dt.tz_localize(pytz.utc).dt.tz_convert(tz)
df["Close time"] = pd.to_datetime(df["Close time"], unit='ms').dt.tz_localize(pytz.utc).dt.tz_convert(tz)

df = df.drop(df.index[-1])

df.to_csv('prediction_data.csv', index=False)

print("\nData has been successfully downloaded\n")

last_datetime = pd.to_datetime(df.iloc[-1]['Close time'])
last_datetime = last_datetime.replace(microsecond=0, tzinfo=None)
valid_until = last_datetime + pd.Timedelta(interval)
valid_until = valid_until.replace(microsecond=0, tzinfo=None)

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

if os.path.exists('prediction_model/prediction_model.h5'):
    model = tf.keras.models.load_model('prediction_model/prediction_model.h5')

    y_pred = model.predict(x_pred_PCA)
    print("Prediction is valid since:", last_datetime.strftime('%Y-%m-%d %H:%M:%S'))
    print("Prediction is valid until:", valid_until.strftime('%Y-%m-%d %H:%M:%S'))
    print("Prediction: {:.4f}".format(y_pred[0].item()))

else:
    print("Error: Model file not found")
