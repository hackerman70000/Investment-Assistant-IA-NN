import datetime
import json
import logging
import os
from typing import Any, Dict

import joblib
import pandas as pd
import pytz
import requests
import tensorflow as tf
from requests.exceptions import RequestException
from sklearn.exceptions import NotFittedError
from ta import add_all_ta_features


def setup_logging(log_file: str = "logs/predict.log"):
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


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
        self.model_dir = f"models/prediction/model_{self.symbol}_{self.interval}_latest"
        self.feature_order = self._load_feature_order(config)

        self.entry_price = None
        self.last_datetime = None
        self.df = None
        self.total_delta = None
        self.x_pred = None
        self.x_pred_scaled = None
        self.x_pred_PCA = None

    def run_prediction(self):
        try:
            self.download_market_data()
            self.preprocess_data()
            self.load_models()
            predictions = self.predict()
            self.print_results(predictions)
            self.get_target_prices()
        except Exception as e:
            logging.error(f"An error occurred during prediction: {e}", exc_info=True)

    def download_market_data(self):
        end_time = int(datetime.datetime.now(self.timezone).timestamp() * 1000)
        start_time = int(
            datetime.datetime(2023, 2, 28, tzinfo=pytz.utc)
            .astimezone(self.timezone)
            .timestamp()
            * 1000
        )
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={self.symbol}&interval={self.interval}&limit={self.limit}"

        data = self._fetch_data(url, start_time, end_time)
        self._process_downloaded_data(data)

    def _fetch_data(self, url: str, start_time: int, end_time: int) -> list:
        data = []
        while start_time < end_time:
            try:
                response = requests.get(
                    f"{url}&startTime={start_time}&endTime={end_time}"
                )
                response.raise_for_status()
                klines = response.json()
                data.extend(klines)
                start_time = klines[-1][6] + 1
            except RequestException as e:
                logging.error(f"Error occurred while fetching data: {e}")
                raise

        return data

    def _process_downloaded_data(self, data: list):
        columns = [
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
        ]
        self.df = pd.DataFrame(data, columns=columns)

        for col in ["Open time", "Close time"]:
            self.df[col] = (
                pd.to_datetime(self.df[col], unit="ms")
                .dt.tz_localize(pytz.utc)
                .dt.tz_convert(self.timezone)
            )

        self.df = self.df.iloc[:-1]
        self.entry_price = float(self.df.iloc[-1]["Close"])
        self.save_market_data_to_csv()
        self.last_datetime = self.df.iloc[-1]["Close time"] + pd.Timedelta(seconds=1)
        self.last_datetime = self.last_datetime.replace(microsecond=0, tzinfo=None)

    def _load_feature_order(self, config: Dict[str, Any]):
        artifacts_dir = os.path.join(self.model_dir, "artifacts")
        json_path = os.path.join(artifacts_dir, "feature_order.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        return data["columns"]

    def save_market_data_to_csv(self):
        os.makedirs(self.directory, exist_ok=True)
        try:
            self.df.to_csv(self.path, index=False)
            logging.info(f"Data has been successfully saved to {self.path}")
        except Exception as e:
            logging.error(f"Error saving data to {self.path}: {e}")
            raise

    def preprocess_data(self):
        interval_units = {
            "m": "minutes",
            "h": "hours",
            "d": "days",
            "w": "weeks",
            "M": "months",
            "y": "years",
        }
        interval_num, interval_unit = int(self.interval[:-1]), self.interval[-1]
        self.total_delta = self.num_intervals * pd.Timedelta(
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

        columns_to_keep = set(self.df.columns) - {
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
        }
        self.df = self.df[list(columns_to_keep)].dropna()

        self.x_pred = self.df.iloc[-1:, :]
        logging.info(f"Preprocessed data shape: {self.x_pred.shape}")
        logging.info(f"Preprocessed data columns: {self.x_pred.columns.tolist()}")

    def load_models(self):
        artifacts_dir = os.path.join(self.model_dir, "artifacts")
        try:
            pca = joblib.load(os.path.join(artifacts_dir, "pca.joblib"))
            scaler = joblib.load(os.path.join(artifacts_dir, "scaler.joblib"))
            self.x_pred = self.x_pred[self.feature_order]

            self.x_pred_scaled = scaler.transform(self.x_pred)
            self.x_pred_PCA = pca.transform(self.x_pred_scaled)

            logging.info("Models loaded successfully")
        except (FileNotFoundError, NotFittedError) as e:
            logging.error(f"Error loading models: {e}")
            raise

    def predict(self):
        model_path = os.path.join(self.model_dir, "artifacts", "NN_model.keras")
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            y_pred = model.predict(self.x_pred_PCA)
            logging.info(f"Prediction made: {y_pred[0].item()}")
            return {"NN_model": y_pred[0].item()}
        else:
            logging.error(f"Model not found at {model_path}")
            return {}

    def print_results(self, predictions: Dict[str, float]):
        for model_name, prediction in predictions.items():
            logging.info(f"Prediction from {model_name}: {prediction:.4f}")
        logging.info(f"Prediction is valid at: {self.last_datetime:%Y-%m-%d %H:%M:%S}")
        logging.info(
            f"Sell until: {(self.last_datetime + self.total_delta):%Y-%m-%d %H:%M:%S}"
        )

    def get_target_prices(self):
        def get_price(symbol):
            url = f"https://data-api.binance.vision/api/v3/ticker/price?symbol={symbol}"
            response = requests.get(url).json()
            return float(response["price"])

        price_USDT = get_price("BTCUSDT")
        logging.info(f"Target price USDT: {round(price_USDT * self.price_delta, 2)}")


if __name__ == "__main__":
    setup_logging()
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
