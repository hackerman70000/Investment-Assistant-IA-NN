import glob
import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features


class Preprocessor:
    def __init__(
        self,
        symbol: str,
        interval: str,
        price_delta: float,
        num_intervals: int,
        prediction_klines_pct: float,
    ):
        self.symbol = symbol
        self.interval = interval
        self.price_delta = price_delta
        self.num_intervals = num_intervals
        self.prediction_klines_pct = prediction_klines_pct
        self.processed_data_path = f"data/processed/{self.symbol}_{self.interval}.npy"

    def preprocess_market_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        logging.info("Starting data preprocessing")
        os.makedirs("data/processed", exist_ok=True)

        if os.path.exists(self.processed_data_path):
            logging.info(
                f"Preprocessed data found at {self.processed_data_path}. Loading data..."
            )
            data = np.load(self.processed_data_path, allow_pickle=True).item()
            return (data["x_train"], data["y_train"], data["x_dev"], data["y_dev"])

        df = self._load_csv_data()
        df = self._add_technical_features(df)
        df = self._create_target_variable(df)
        df = self._drop_unnecessary_columns(df)

        x, y, x_dev, y_dev = self._split_data(df)

        self._print_nan_occurrences(x, y, x_dev, y_dev)

        logging.info(f"Saving preprocessed data to {self.processed_data_path}")
        np.save(
            self.processed_data_path,
            {"x_train": x, "y_train": y, "x_dev": x_dev, "y_dev": y_dev},
        )

        return x, y, x_dev, y_dev

    def split_train_test_data(
        self, x: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        logging.info("Splitting data into train and test sets")
        return train_test_split(x, y, test_size=0.1, shuffle=True)

    def scale_data(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame, x_dev: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logging.info("Scaling data")
        scaler = MinMaxScaler(feature_range=(0, 1))
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        x_dev_scaled = scaler.transform(x_dev)

        return x_train_scaled, x_test_scaled, x_dev_scaled, scaler

    def apply_pca(
        self,
        x_train_scaled: np.ndarray,
        x_test_scaled: np.ndarray,
        x_dev_scaled: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
        logging.info("Applying PCA")
        pca = PCA(n_components=24)
        x_train_PCA = pca.fit_transform(x_train_scaled)
        x_test_PCA = pca.transform(x_test_scaled)
        x_dev_PCA = pca.transform(x_dev_scaled)

        return x_train_PCA, x_test_PCA, x_dev_PCA, pca

    def _load_csv_data(self) -> pd.DataFrame:
        file_pattern = f"data/raw/{self.symbol}_{self.interval}_*.csv"
        csv_files = glob.glob(file_pattern)

        if not csv_files:
            logging.error(f"No CSV file found matching the pattern: {file_pattern}")
            raise FileNotFoundError(
                f"No CSV file found matching the pattern: {file_pattern}"
            )

        selected_file = max(csv_files, key=lambda f: self._get_date_range(f))
        logging.info(f"Selected file: {selected_file}")

        logging.info("Loading selected CSV file")
        df = pd.read_csv(selected_file)
        return df.sort_values(by="Open time").reset_index(drop=True)

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Adding technical analysis features")
        return add_all_ta_features(
            df,
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume",
            fillna=True,
        )

    def _create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Creating target variable")
        conditions = [
            (df["Close"] * self.price_delta <= df["High"].shift(-i))
            for i in range(1, self.num_intervals + 1)
        ]
        df["target"] = np.where(np.any(conditions, axis=0), 1, 0)
        logging.info("Counts of target values:")
        logging.info(df["target"].value_counts())
        return df

    def _drop_unnecessary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
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
        return df.drop(columns=columns_to_drop).dropna()

    def _split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        number_of_prediction_klines = int(len(df) * self.prediction_klines_pct)
        df_train = df.iloc[:-number_of_prediction_klines]
        df_dev = df.iloc[number_of_prediction_klines : -self.num_intervals]

        x = df_train.drop(columns=["target"])
        y = df_train["target"]
        x_dev = df_dev.drop(columns=["target"])
        y_dev = df_dev["target"]

        return x, y, x_dev, y_dev

    @staticmethod
    def _get_date_range(filename: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
        parts = filename.split("_")
        if len(parts) < 4:
            logging.warning(f"Unexpected filename format: {filename}")
            return pd.Timestamp.min, pd.Timestamp.min

        start_date_str, end_date_str = parts[-2], parts[-1].replace(".csv", "")

        try:
            start_date = pd.to_datetime(start_date_str)
            end_date = pd.to_datetime(end_date_str)
            return start_date, end_date
        except ValueError as e:
            logging.warning(f"Error parsing dates from filename {filename}: {str(e)}")
            return pd.Timestamp.min, pd.Timestamp.min

    @staticmethod
    def _print_nan_occurrences(*dataframes):
        logging.info("NaNs occurrences:")
        for name, df in zip(["x", "y", "x_dev", "y_dev"], dataframes):
            n_missing = df.isnull().sum().sum()
            logging.info(
                f"{name}: {'No' if n_missing == 0 else n_missing} missing values"
            )
