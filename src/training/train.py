import glob
import json
import logging
import os
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from architecture import build_model
from preprocess import Preprocessor
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features


def setup_logging(log_file: str = "logs/train.log"):
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def load_config(config_path: str = "src/training/config.json"):
    with open(config_path, "r") as file:
        config = json.load(file)
    return config


class Trainer:
    def __init__(
        self,
        symbol: str,
        interval: str,
        price_delta: float,
        num_intervals: int,
        prediction_klines_pct: float,
        epochs: int,
        batch_size: int,
        patience: int,
    ):
        self.symbol = symbol
        self.interval = interval
        self.price_delta = price_delta
        self.num_intervals = num_intervals
        self.prediction_klines_pct = prediction_klines_pct
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_dir = (
            f"data/models/model_{self.symbol}_{self.interval}_{self.timestamp}"
        )

        self.preprocessor = Preprocessor(
            symbol, interval, price_delta, num_intervals, prediction_klines_pct
        )

        logging.info(
            f"Initializing model trainer for {self.symbol} with interval {self.interval}"
        )

    def preprocess_market_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        logging.info("Starting data preprocessing")
        os.makedirs("data/models", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)

        if os.path.exists(self.processed_data_path):
            logging.info(
                f"Preprocessed data found at {self.processed_data_path}. Loading data..."
            )
            data = np.load(self.processed_data_path, allow_pickle=True).item()
            return (data["x_train"], data["y_train"], data["x_dev"], data["y_dev"])

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

        df = df.sort_values(by="Open time").reset_index(drop=True)

        logging.info("Adding technical analysis features")
        df = add_all_ta_features(
            df,
            open="Open",
            high="High",
            low="Low",
            close="Close",
            volume="Volume",
            fillna=True,
        )

        logging.info("Creating target variable")
        conditions = [
            (df["Close"] * self.price_delta <= df["High"].shift(-i))
            for i in range(1, self.num_intervals + 1)
        ]
        df["target"] = np.where(np.any(conditions, axis=0), 1, 0)

        logging.info("Counts of target values:")
        logging.info(df["target"].value_counts())

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
        df = df.drop(columns=columns_to_drop).dropna()

        number_of_prediction_klines = int(len(df) * self.prediction_klines_pct)
        df_train = df.iloc[:-number_of_prediction_klines]
        df_dev = df.iloc[number_of_prediction_klines : -self.num_intervals]

        x = df_train.drop(columns=["target"])
        y = df_train["target"]
        x_dev = df_dev.drop(columns=["target"])
        y_dev = df_dev["target"]

        self._print_nan_occurrences(x, y, x_dev, y_dev)

        logging.info(f"Saving preprocessed data to {self.processed_data_path}")
        np.save(
            self.processed_data_path,
            {"x_train": x, "y_train": y, "x_dev": x_dev, "y_dev": y_dev},
        )

        return x, y, x_dev, y_dev

    @staticmethod
    def _get_date_range(filename: str) -> Tuple[datetime, datetime]:
        parts = filename.split("_")
        if len(parts) < 4:
            logging.warning(f"Unexpected filename format: {filename}")
            return datetime.min, datetime.min

        start_date_str, end_date_str = parts[-2], parts[-1].replace(".csv", "")

        date_format = "%Y-%m-%d"

        try:
            start_date = datetime.strptime(start_date_str, date_format)
            end_date = datetime.strptime(end_date_str, date_format)
            return start_date, end_date
        except ValueError as e:
            logging.warning(f"Error parsing dates from filename {filename}: {str(e)}")
            return datetime.min, datetime.min

    @staticmethod
    def _print_nan_occurrences(*dataframes):
        logging.info("NaNs occurrences:")
        for name, df in zip(["x", "y", "x_dev", "y_dev"], dataframes):
            n_missing = df.isnull().sum().sum()
            logging.info(
                f"{name}: {'No' if n_missing == 0 else n_missing} missing values"
            )

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

        os.makedirs(self.model_dir, exist_ok=True)
        np.save(os.path.join(self.model_dir, "scaler_scale.npy"), scaler.scale_)
        np.save(os.path.join(self.model_dir, "scaler_min.npy"), scaler.min_)

        return x_train_scaled, x_test_scaled, x_dev_scaled

    def apply_pca(
        self,
        x_train_scaled: np.ndarray,
        x_test_scaled: np.ndarray,
        x_dev_scaled: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logging.info("Applying PCA")
        pca = PCA(n_components=24)
        x_train_PCA = pca.fit_transform(x_train_scaled)
        x_test_PCA = pca.transform(x_test_scaled)
        x_dev_PCA = pca.transform(x_dev_scaled)

        np.save(os.path.join(self.model_dir, "pca_components.npy"), pca.components_)
        np.save(os.path.join(self.model_dir, "pca_mean.npy"), pca.mean_)

        return x_train_PCA, x_test_PCA, x_dev_PCA

    def build_model(self, input_shape: int) -> tf.keras.Model:
        return build_model(input_shape)

    def train_model(
        self, model: tf.keras.Model, x_train_PCA: np.ndarray, y_train: pd.Series
    ) -> tf.keras.callbacks.History:
        logging.info("Training the model")
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience
        )

        history = model.fit(
            x_train_PCA,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            shuffle=True,
            callbacks=[early_stopping],
        )
        logging.info("Model training completed")

        return history

    def evaluate_model(
        self, model: tf.keras.Model, x_test_PCA: np.ndarray, y_test: pd.Series
    ) -> Tuple[float, float, float]:
        logging.info("Evaluating the model")
        loss, precision, recall = model.evaluate(x_test_PCA, y_test, batch_size=32)
        f1 = 2 * (precision * recall) / (precision + recall)
        logging.info(
            f"Test loss: {loss:.4f}, Test precision: {precision:.4f}, Test recall: {recall:.4f}, Test F1 score: {f1:.4f}"
        )

        os.makedirs(self.model_dir, exist_ok=True)
        model.save(os.path.join(self.model_dir, "NN_model.keras"))
        logging.info(f"Model saved to {os.path.join(self.model_dir, 'NN_model.keras')}")

        return loss, precision, recall

    def save_results(
        self,
        model: tf.keras.Model,
        x_dev_PCA: np.ndarray,
        y_dev: pd.Series,
        test_metrics: Tuple[float, float, float],
    ):
        logging.info(f"Saving results to directory: {self.model_dir}")
        os.makedirs(self.model_dir, exist_ok=True)

        y_pred = model.predict(x_dev_PCA).flatten()
        y_pred_binary = (y_pred > 0.5).astype(int)

        cm = confusion_matrix(y_dev, y_pred_binary)
        precision = precision_score(y_dev, y_pred_binary)
        recall = recall_score(y_dev, y_pred_binary)
        f1 = f1_score(y_dev, y_pred_binary)
        accuracy = accuracy_score(y_dev, y_pred_binary)

        true_positives = cm[1][1]
        total_positives = np.sum(cm[1])
        pct_taken_opportunities = (
            true_positives / total_positives if total_positives > 0 else 0
        )

        self._write_results_to_file(
            test_metrics,
            cm,
            precision,
            recall,
            f1,
            accuracy,
            pct_taken_opportunities,
            model,
        )
        logging.info("Results written to results.txt")

    def _write_results_to_file(
        self,
        test_metrics: Tuple[float, float, float],
        cm: np.ndarray,
        precision: float,
        recall: float,
        f1: float,
        accuracy: float,
        pct_taken_opportunities: float,
        model: tf.keras.Model,
    ):
        with open(
            os.path.join(self.model_dir, "results.txt"), "w", encoding="utf-8"
        ) as f:
            f.write("Market Data Analysis Results\n")
            f.write("============================\n\n")

            f.write("Model Configuration\n")
            f.write("-------------------\n")
            f.write(f"Symbol: {self.symbol}\n")
            f.write(f"Interval: {self.interval}\n")
            f.write(f"Price delta: {self.price_delta}\n")
            f.write(f"Number of intervals: {self.num_intervals}\n")
            f.write(
                f"Prediction klines percentage: {self.prediction_klines_pct:.2%}\n\n"
            )

            f.write("Test Set Metrics\n")
            f.write("-----------------\n")
            f.write(f"Loss: {test_metrics[0]:.4f}\n")
            f.write(f"Precision: {test_metrics[1]:.4f}\n")
            f.write(f"Recall: {test_metrics[2]:.4f}\n")
            f.write(
                f"F1 Score: {2 * (test_metrics[1] * test_metrics[2]) / (test_metrics[1] + test_metrics[2]):.4f}\n\n"
            )

            f.write("Validation Set Metrics\n")
            f.write("-----------------------\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(
                f"Percentage of Taken Opportunities: {pct_taken_opportunities:.2%}\n\n"
            )

            f.write("Confusion Matrix\n")
            f.write("-----------------\n")
            f.write("    Predicted\n")
            f.write("    0    1\n")
            f.write(f"0   {cm[0][0]:<4} {cm[0][1]:<4}\n")
            f.write(f"1   {cm[1][0]:<4} {cm[1][1]:<4}\n\n")

            f.write("Model Summary\n")
            f.write("--------------\n")
            model.summary(print_fn=lambda x: f.write(x + "\n"))

    def _save_preprocessing_artifacts(self, scaler: MinMaxScaler, pca: PCA):
        os.makedirs(self.model_dir, exist_ok=True)
        np.save(os.path.join(self.model_dir, "scaler_scale.npy"), scaler.scale_)
        np.save(os.path.join(self.model_dir, "scaler_min.npy"), scaler.min_)
        np.save(os.path.join(self.model_dir, "pca_components.npy"), pca.components_)
        np.save(os.path.join(self.model_dir, "pca_mean.npy"), pca.mean_)

    def run(self):
        x, y, x_dev, y_dev = self.preprocessor.preprocess_market_data()
        x_train, x_test, y_train, y_test = self.preprocessor.split_train_test_data(x, y)
        x_train_scaled, x_test_scaled, x_dev_scaled, scaler = (
            self.preprocessor.scale_data(x_train, x_test, x_dev)
        )
        x_train_PCA, x_test_PCA, x_dev_PCA, pca = self.preprocessor.apply_pca(
            x_train_scaled, x_test_scaled, x_dev_scaled
        )

        self._save_preprocessing_artifacts(scaler, pca)

        model = self.build_model(x_train_PCA.shape[1])
        logging.info("Starting the training process")
        self.train_model(model, x_train_PCA, y_train)
        test_metrics = self.evaluate_model(model, x_test_PCA, y_test)
        self.save_results(model, x_dev_PCA, y_dev, test_metrics)
        logging.info("Training process completed")


if __name__ == "__main__":
    config = load_config()
    setup_logging(config.get("log_file", "logs/train.log"))

    logging.info("Starting model trainer")
    try:
        trainer = Trainer(
            symbol=config["symbol"],
            interval=config["interval"],
            price_delta=config["price_delta"],
            num_intervals=config["num_intervals"],
            prediction_klines_pct=config["prediction_klines_pct"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            patience=config["patience"],
        )
        trainer.run()
        logging.info("Model trainer process completed successfully")
    except Exception as e:
        logging.exception(f"An error occurred during training: {e}")
    logging.info("Model trainer process ended")
