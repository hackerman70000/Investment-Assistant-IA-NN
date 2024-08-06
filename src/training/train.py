import os
from datetime import datetime
import glob
import io

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features


class Training:
    def __init__(self, symbol, interval, price_delta, num_intervals):
        self.symbol = symbol
        self.interval = interval
        self.price_delta = price_delta
        self.num_intervals = num_intervals
        self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.dir_name = f'data/models/model_{self.timestamp}'

    def preprocess_market_data(self):
        if not os.path.exists('data/models'):
            os.makedirs('data/models')

        os.makedirs(self.dir_name)

        file_pattern = f'data/raw/{self.symbol}_{self.interval}_*.csv'
        csv_files = glob.glob(file_pattern)
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found matching the pattern: {file_pattern}")
        
        df_list = []
        for file in csv_files:
            df = pd.read_csv(file)
            df_list.append(df)
        
        df = pd.concat(df_list, ignore_index=True)

        df = df.sort_values(by="Open time")
        df.reset_index(drop=True, inplace=True)

        np.seterr(divide='ignore', invalid='ignore')
        df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

        conditions = [
            (df['Close'] * self.price_delta <= df['High'].shift(-i))
            for i in range(1, self.num_intervals + 1)
        ]
        conditions_array = np.array(conditions)
        df['exp'] = np.where(conditions_array.any(axis=0), 1, 0)

        print('\nCounts of expected values :')
        print(df['exp'].value_counts())

        df.drop(
            ['High', 'Open', 'Close', 'Low', 'Volume', 'Quote asset volume', 'Number of trades',
             'Taker buy base asset volume',
             'Taker buy quote asset volume', 'Ignore', 'Open time', 'Close time'],
            inplace=True, axis=1)

        df.dropna(inplace=True)

        number_of_prediction_klines = 1000
        df_train = df.iloc[:df.shape[0] - number_of_prediction_klines, :]
        df_dev = df.iloc[df.shape[0] - number_of_prediction_klines:, :]

        df_dev = df_dev.drop(df_dev.index[-self.num_intervals:])

        df_train.reset_index(drop=True, inplace=True)
        df_dev.reset_index(drop=True, inplace=True)

        x = df_train.iloc[:, :len(df.columns) - 1]
        y = df_train['exp']

        x_dev = df_dev.iloc[:, :len(df_dev.columns) - 1]
        y_dev = df_dev['exp']

        print('\nNaNs occurences:')
        data_frames = {'x': x, 'y': y, 'x_dev': x_dev, 'y_dev': y_dev}
        for name, df in data_frames.items():
            n_missing = df.isnull().sum().sum()
            if n_missing > 0:
                print(f"{name}: {n_missing} missing values")
            else:
                print(f"{name}: No missing values")
        print('')

        return x, y, x_dev, y_dev

    def split_train_test_data(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)
        return x_train, x_test, y_train, y_test

    def scale_data(self, x_train, x_test, x_dev):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(x_train)
        joblib.dump(scaler, os.path.join(self.dir_name, 'scaler.joblib'))

        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        x_dev_scaled = scaler.transform(x_dev)

        return x_train_scaled, x_test_scaled, x_dev_scaled

    def apply_pca(self, x_train_scaled, x_test_scaled, x_dev_scaled):
        pca = PCA(n_components=24)
        pca.fit(x_train_scaled)
        joblib.dump(pca, os.path.join(self.dir_name, 'pca.joblib'))

        x_train_PCA = pca.transform(x_train_scaled)
        x_test_PCA = pca.transform(x_test_scaled)
        x_dev_PCA = pca.transform(x_dev_scaled)

        return x_train_PCA, x_test_PCA, x_dev_PCA

    def build_model(self, x_train_PCA, y_train):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )

        early_stopping = EarlyStopping(monitor='val_loss', patience=5)

        model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=[tf.keras.metrics.Precision()])

        model.fit(x_train_PCA, y_train, batch_size=1, epochs=2, validation_split=0.2, validation_data=None,
                  shuffle=True,
                  callbacks=[early_stopping])
        print("")

        return model

    def evaluate_model(self, model, x_test_PCA, y_test):
        evaluation = model.evaluate(x_test_PCA, y_test, batch_size=1)
        print("\ntest loss, test acc:", evaluation)
        print("")

        model.save(os.path.join(self.dir_name, f'NN_{round(evaluation[1] * 100, 1)}%.h5'))

        return evaluation

    def save_results(self, model, x_dev_PCA, y_dev, evaluation):
        predictions = pd.DataFrame(model.predict(x_dev_PCA), columns=['0-1'])
        predictions['target'] = y_dev
        predictions['predictions'] = np.where(predictions['0-1'] > 0.5, 1, 0)

        pd.set_option('display.max_rows', None)
        print(predictions)

        predictions.to_csv(os.path.join(self.dir_name, 'predictions.csv'), index=False)

        m = tf.keras.metrics.Precision()
        m.update_state(predictions['target'], predictions['predictions'])

        precision = round(m.result().numpy(), 1)
        num_good_decision = len(predictions[(predictions['predictions'] == 1) & (predictions['target'] == 1)])
        num_opportunities = len(predictions[(predictions['target'] == 1)])
        pct_good_decision = num_good_decision / num_opportunities

        print(f"\nVal precision: {str(round(precision * 100, 1))}%")
        print(f"Test Precision: {str(round(evaluation[1] * 100, 1))}%")
        print(f"Percentage of taken opportunities: {str(round(pct_good_decision * 100, 1))}%")

        with open(os.path.join(self.dir_name, 'results.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Symbol: {self.symbol}\n")
            f.write(f"Interval: {self.interval}\n")
            f.write(f"Price delta: {self.price_delta}\n")
            f.write(f"Num intervals: {self.num_intervals}\n\n")
            f.write(f"Val precision: {str(round(precision * 100, 1))}%\n")
            f.write(f"Test Precision: {str(round(evaluation[1] * 100, 1))}%\n")
            f.write(f"Percentage of taken opportunities: {str(round(pct_good_decision * 100, 1))}%\n\n")
            
            # Capture model summary in a string
            summary_string = io.StringIO()
            model.summary(print_fn=lambda x: summary_string.write(x + '\n'))
            f.write(summary_string.getvalue())

        final_dir_name = f'data/models/model_P-{round(precision * 100, 2)}%_O-{round(pct_good_decision * 100, 2)}%'

        try:
            os.rename(self.dir_name, final_dir_name)
        except FileNotFoundError:
            print(f"Directory '{self.dir_name}' not found")
        except FileExistsError:
            print(f"Directory '{final_dir_name}' already exists")
            os.rename(self.dir_name, f'{final_dir_name}_{self.timestamp}')
        except OSError as e:
            print(f"Error: {e}")


    def run(self):
        x, y, x_dev, y_dev = self.preprocess_market_data()
        x_train, x_test, y_train, y_test = self.split_train_test_data(x, y)
        x_train_scaled, x_test_scaled, x_dev_scaled = self.scale_data(x_train, x_test, x_dev)
        x_train_PCA, x_test_PCA, x_dev_PCA = self.apply_pca(x_train_scaled, x_test_scaled, x_dev_scaled)
        model = self.build_model(x_train_PCA, y_train)
        evaluation = self.evaluate_model(model, x_test_PCA, y_test)
        self.save_results(model, x_dev_PCA, y_dev, evaluation)


if __name__ == '__main__':
    Training(symbol="BTCUSDT", interval="1d", price_delta=1.05, num_intervals=12).run()