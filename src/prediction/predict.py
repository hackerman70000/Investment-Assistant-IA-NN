import datetime
import os

import numpy as np
import pandas as pd
import pytz
import requests
import tensorflow as tf
from joblib import load
from ta import add_all_ta_features

from market_data_scraper import market_data_scraper


class PredictionModel:
    def __init__(self, symbol, interval, price_delta, num_intervals, timezone, limit, directory, filename):
        self.symbol = symbol
        self.interval = interval
        self.price_delta = price_delta
        self.num_intervals = num_intervals
        self.timezone = timezone
        self.limit = limit
        self.directory = directory
        self.filename = filename
        self.path = f"{directory}/{filename}_{interval}.csv"
        self.scraper = market_data_scraper(symbol, interval, limit, directory, filename)
        self.entry_price = None
        self.last_datetime = None
        self.df = None
        self.total_delta = None

    def run_prediction(self):
        self.download_market_data()
        self.preprocess_data()
        self.load_models()
        self.predict()
        self.print_results()
        self.get_target_prices()

    def download_market_data(self):
        self.df = self.scraper.download_market_data()
        self.df = self.df.drop(self.df.index[-1])
        self.entry_price = pd.to_numeric(self.df.iloc[-1]['Close'])
        self.scraper.save_market_data_to_csv(self.df)
        self.last_datetime = pd.to_datetime(self.df.iloc[-1]['Close time']) + datetime.timedelta(seconds=1)
        self.last_datetime = self.last_datetime.replace(microsecond=0, tzinfo=None)

    def preprocess_data(self):
        interval_units = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks", "M": "months", "y": "years"}
        interval_unit = self.interval[-1]
        interval_num = int(self.interval[:-1])
        self.total_delta = self.num_intervals * datetime.timedelta(**{interval_units[interval_unit]: interval_num})

        self.df = pd.read_csv(self.path)
        self.df = self.df.sort_values(by="Open time")
        self.df.reset_index(drop=True, inplace=True)
        np.seterr(divide='ignore', invalid='ignore')
        self.df = add_all_ta_features(self.df, open="Open", high="High", low="Low", close="Close", volume="Volume",
                                      fillna=True)
        self.df.drop(['High', 'Open', 'Close', 'Low', 'Volume', 'Quote asset volume', 'Number of trades',
                      'Taker buy base asset volume',
                      'Taker buy quote asset volume', 'Ignore', 'Open time', 'Close time'], inplace=True, axis=1)
        self.df.dropna(inplace=True)

        self.x_pred = self.df.iloc[self.df.shape[0] - 1:, :]

    def load_models(self):
        pca = load('IA-NN-models/pca.joblib')
        scaler = load('IA-NN-models/scaler.joblib')
        self.x_pred_scaled = scaler.transform(self.x_pred)
        self.x_pred_PCA = pca.transform(self.x_pred_scaled)

    def predict(self):
        if os.path.exists('IA-NN-models'):
            for file_name in os.listdir('IA-NN-models'):
                if file_name.endswith('.h5'):
                    model_path = os.path.join('IA-NN-models', file_name)
                    model = tf.keras.models.load_model(model_path)
                    y_pred = model.predict(self.x_pred_PCA)
                    print(f"\nPrediction from {file_name}: {y_pred[0].item():.4f}")
        else:
            print("\nError: Model directory not found")

    def print_results(self):
        print(f"\nPrediction is valid at: {self.last_datetime:%Y-%m-%d %H:%M:%S}")
        print(f"Sell until:             {(self.last_datetime + self.total_delta):%Y-%m-%d %H:%M:%S}")

    def get_target_prices(self):
        url_price_USDT = "https://data.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        response = requests.get(url_price_USDT).json()
        price_USDT = float(response["price"])

        url_price_BUSD = "https://data.binance.com/api/v3/ticker/price?symbol=BTCBUSD"
        response = requests.get(url_price_BUSD).json()
        price_BUSD = float(response["price"])

        print(f"\nTarget price USDT: {round(price_USDT * self.price_delta, 2)}")
        print(f"Target price BUSD: {round(price_BUSD * self.price_delta, 2)}")


if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = "1h"
    price_delta = 1.015
    num_intervals = 12
    timezone = pytz.timezone('Europe/Warsaw')
    limit = 1000
    directory = "prediction_data"
    filename = "prediction_data"

    model = PredictionModel(symbol, interval, price_delta, num_intervals, timezone, limit, directory, filename)
    model.run_prediction()
