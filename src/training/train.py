import json
import logging
import os
import shutil
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from architecture import build_model
from preprocess import Preprocessor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


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
        config_path: str = "src/training/config.json",
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
            f"models/training/model_{self.symbol}_{self.interval}_{self.timestamp}"
        )
        self.config_path = config_path

        self.preprocessor = Preprocessor(
            symbol,
            interval,
            price_delta,
            num_intervals,
            prediction_klines_pct,
            self.model_dir,
        )

        logging.info(
            f"Initializing model trainer for {self.symbol} with interval {self.interval}"
        )

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

        artifacts_dir = self._create_artifacts_subdir("artifacts")
        model.save(os.path.join(artifacts_dir, "NN_model.keras"))
        logging.info(f"Model saved to {os.path.join(artifacts_dir, 'NN_model.keras')}")

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

    def _copy_config_file(self):
        artifacts_dir = self._create_artifacts_subdir("artifacts")
        dest_path = os.path.join(artifacts_dir, "config.json")
        try:
            shutil.copy2(self.config_path, dest_path)
            logging.info(f"Configuration file copied to {dest_path}")
        except FileNotFoundError:
            logging.error(f"Configuration file not found at {self.config_path}")
        except PermissionError:
            logging.error(
                f"Permission denied when copying configuration file to {dest_path}"
            )
        except Exception as e:
            logging.error(
                f"An error occurred while copying the configuration file: {str(e)}"
            )

    def _create_artifacts_subdir(self, subdir_name: str):
        subdir_path = os.path.join(self.model_dir, subdir_name)
        os.makedirs(subdir_path, exist_ok=True)
        return subdir_path

    def _save_model_summary(self, model: tf.keras.Model):
        os.makedirs(self.model_dir, exist_ok=True)
        summary_path = os.path.join(self.model_dir, "architecture.txt")

        with open(summary_path, "w", encoding="utf-8") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))

        logging.info(f"Model summary saved to {summary_path}")

    def run(self):
        x, y, x_dev, y_dev = self.preprocessor.preprocess_market_data()
        x_train, x_test, y_train, y_test = self.preprocessor.split_train_test_data(x, y)
        x_train_scaled, x_test_scaled, x_dev_scaled, _ = self.preprocessor.scale_data(
            x_train, x_test, x_dev
        )
        x_train_PCA, x_test_PCA, x_dev_PCA, _ = self.preprocessor.apply_pca(
            x_train_scaled, x_test_scaled, x_dev_scaled
        )

        model = self.build_model(x_train_PCA.shape[1])
        logging.info("Starting the training process")
        self.train_model(model, x_train_PCA, y_train)
        test_metrics = self.evaluate_model(model, x_test_PCA, y_test)
        self.save_results(model, x_dev_PCA, y_dev, test_metrics)
        self._copy_config_file()
        self._save_model_summary(model)
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
