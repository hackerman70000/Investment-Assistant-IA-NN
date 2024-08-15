import datetime
import os
import sys
import time
from typing import Any, Dict

import joblib
import pandas as pd
import pytz
import requests
import tensorflow as tf
from requests.exceptions import RequestException
from ta import add_all_ta_features

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class PredictionModel:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.symbol = config["symbol"]
        self.interval = config["interval"]
        self.price_delta = config["price_delta"]
        self.num_intervals = config["num_intervals"]
        self.timezone = pytz.timezone(config["timezone"])
        self.limit = config["limit"]
        self.directory = config["directory"]
        self.filename = config["filename"]
        self.path = (
            f"{self.directory}/{self.filename}_{self.symbol}_{self.interval}.csv"
        )
        self.entry_price = None
        self.last_datetime = None
        self.df = None
        self.total_delta = None
        self.x_pred = None
        self.x_pred_scaled = None
        self.x_pred_PCA = None
        self.model_dir = f"models/prediction/model_{self.symbol}_{self.interval}_latest"

    def run_prediction(self):
        self.download_market_data()
        self.preprocess_data()
        self.load_models()
        predictions = self.predict()
        self.print_results(predictions)
        self.get_target_prices()

    def download_market_data(self):
        end_time = int(time.time() * 1000)
        start_time = int(
            datetime.datetime(2023, 2, 28, tzinfo=pytz.utc)
            .astimezone(self.timezone)
            .timestamp()
            * 1000
        )
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={self.symbol}&interval={self.interval}&limit={self.limit}"
        data = []
        while start_time < end_time:
            try:
                response = requests.get(
                    url + f"&startTime={start_time}&endTime={end_time}"
                )
                response.raise_for_status()
                klines = response.json()
                data += klines
                start_time = klines[-1][6] + 1
            except RequestException as e:
                print("Error occurred:", e)
                url = f"https://data-api.binance.vision/api/v3/klines?symbol={self.symbol}&interval={self.interval}&limit={self.limit}"
                time.sleep(60)

        self.df = pd.DataFrame(
            data,
            columns=[
                "Open time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Close time",
                "Quote asset volume",
                "Number of trades",
                "Taker buy base asset volume",
                "Taker buy quote asset volume",
                "Ignore",
            ],
        )
        self.df["Open time"] = (
            pd.to_datetime(self.df["Open time"], unit="ms")
            .dt.tz_localize(pytz.utc)
            .dt.tz_convert(self.timezone)
        )
        self.df["Close time"] = (
            pd.to_datetime(self.df["Close time"], unit="ms")
            .dt.tz_localize(pytz.utc)
            .dt.tz_convert(self.timezone)
        )

        self.df = self.df.iloc[:-1]  # Drop the last row
        self.entry_price = float(self.df.iloc[-1]["Close"])
        self.save_market_data_to_csv()
        self.last_datetime = pd.to_datetime(
            self.df.iloc[-1]["Close time"]
        ) + datetime.timedelta(seconds=1)
        self.last_datetime = self.last_datetime.replace(microsecond=0, tzinfo=None)

    def save_market_data_to_csv(self):
        try:
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            self.df.to_csv(self.path, index=False)
            print(f"Data has been successfully saved to {self.path}")
        except Exception as e:
            print(f"Error saving data to {self.path}: {e}")

    def preprocess_data(self):
        interval_units = {
            "m": "minutes",
            "h": "hours",
            "d": "days",
            "w": "weeks",
            "M": "months",
            "y": "years",
        }
        interval_unit = self.interval[-1]
        interval_num = int(self.interval[:-1])
        self.total_delta = self.num_intervals * datetime.timedelta(
            **{interval_units[interval_unit]: interval_num}
        )

        self.df = pd.read_csv(self.path)
        self.df.sort_values(by="Open time", inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        self.df = add_all_ta_features(
            self.df,
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume",
            fillna=True,
        )

        columns_to_drop = [
            "High",
            "Open",
            "Close",
            "Low",
            "Volume",
            "Quote asset volume",
            "Number of trades",
            "Taker buy base asset volume",
            "Taker buy quote asset volume",
            "Ignore",
            "Open time",
            "Close time",
        ]
        self.df = self.df.drop(columns=columns_to_drop).dropna()

        self.x_pred = self.df.iloc[-1:, :]

    def load_models(self):
        artifacts_dir = os.path.join(self.model_dir, "artifacts")
        pca = joblib.load(os.path.join(artifacts_dir, "pca.joblib"))
        scaler = joblib.load(os.path.join(artifacts_dir, "scaler.joblib"))
        self.x_pred_scaled = scaler.transform(self.x_pred)
        self.x_pred_PCA = pca.transform(self.x_pred_scaled)

    def predict(self):
        predictions = {}
        model_path = os.path.join(self.model_dir, "artifacts", "NN_model.keras")
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            y_pred = model.predict(self.x_pred_PCA)
            predictions["NN_model"] = y_pred[0].item()
        else:
            print(f"\nError: Model not found at {model_path}")
        return predictions

    def print_results(self, predictions: Dict[str, float]):
        for model_name, prediction in predictions.items():
            print(f"\nPrediction from {model_name}: {prediction:.4f}")
        print(f"\nPrediction is valid at: {self.last_datetime:%Y-%m-%d %H:%M:%S}")
        print(
            f"Sell until:             {(self.last_datetime + self.total_delta):%Y-%m-%d %H:%M:%S}"
        )

    def get_target_prices(self):
        def get_price(symbol):
            url = f"https://data-api.binance.vision/api/v3/ticker/price?symbol={symbol}"
            response = requests.get(url).json()
            return float(response["price"])

        price_USDT = get_price(f"{self.symbol}")
        price_BUSD = get_price(f"{self.symbol[:-4]}BUSD")

        print(f"\nTarget price USDT: {round(price_USDT * self.price_delta, 2)}")
        print(f"Target price BUSD: {round(price_BUSD * self.price_delta, 2)}")


if __name__ == "__main__":
    config = {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "price_delta": 1.015,
        "num_intervals": 12,
        "timezone": "Europe/Warsaw",
        "limit": 1000,
        "directory": "data/raw",
        "filename": "market_data",
    }
    model = PredictionModel(config)
    model.run_prediction()
