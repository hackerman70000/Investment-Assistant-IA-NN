import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
interval = "1h"
dir_name = 'saved_models/model_' + timestamp

if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

os.makedirs(dir_name)

df = pd.read_csv(f'market_data_{interval}.csv')

df = df.sort_values(by="Open time")
df.reset_index(drop=True, inplace=True)

price_delta = 1.012

np.seterr(divide='ignore', invalid='ignore')
df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

num_intervals = 12
conditions = [
    (df['Close'] * price_delta <= df['High'].shift(-i))
    for i in range(1, num_intervals + 1)
]
conditions_array = np.array(conditions)
df['exp'] = np.where(conditions_array.any(axis=0), 1, 0)

print('\nCounts of expected values :')
print(df['exp'].value_counts())

df.drop(
    ['High', 'Open', 'Close', 'Low', 'Volume', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
     'Taker buy quote asset volume', 'Ignore', 'Open time', 'Close time'],
    inplace=True, axis=1)

df.dropna(inplace=True)

number_of_prediction_klines = 1000
df_train = df.iloc[:df.shape[0] - number_of_prediction_klines, :]
df_dev = df.iloc[df.shape[0] - number_of_prediction_klines:, :]

df_dev = df_dev.drop(df_dev.index[-num_intervals:])

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(x_train)
joblib.dump(scaler, dir_name + '/scaler.joblib')

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_dev_scaled = scaler.transform(x_dev)

pca = PCA(n_components=24)
pca.fit(x_train_scaled)
joblib.dump(pca, dir_name + '/pca.joblib')

x_train_PCA = pca.transform(x_train_scaled)
x_test_PCA = pca.transform(x_test_scaled)
x_dev_PCA = pca.transform(x_dev_scaled)

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

model.fit(x_train_PCA, y_train, batch_size=1, epochs=2, validation_split=0.2, validation_data=None, shuffle=True,
          callbacks=[early_stopping])
print("")

evaluation = model.evaluate(x_test_PCA, y_test, batch_size=1)
print("\ntest loss, test acc:", evaluation)
print("")

model.save(dir_name + '/NN_' + str(round(evaluation[1] * 100, 1)) + '%' + '.h5')

new_dir_name = 'saved_models/model_' + str(round(evaluation[1] * 100, 1)) + '%_' + timestamp

try:
    os.rename(dir_name, new_dir_name)
except FileNotFoundError:
    print(f"Directory '{dir_name}' not found")
except FileExistsError:
    print(f"Directory '{new_dir_name}' already exists")
except OSError as e:
    print(f"Error: {e}")

predictions = pd.DataFrame(model.predict(x_dev_PCA), columns=['0-1'])
predictions['target'] = y_dev
predictions['predictions'] = np.where(predictions['0-1'] > 0.5, 1, 0)

pd.set_option('display.max_rows', None)
print(predictions)

predictions.to_csv(new_dir_name + '/predictions.csv', index=False)

m = tf.keras.metrics.Precision()
m.update_state(predictions['target'], predictions['predictions'])

precision = round(m.result().numpy(), 1)
num_good_decision = len(predictions[(predictions['predictions'] == 1) & (predictions['target'] == 1)])
num_opportunities = len(predictions[(predictions['target'] == 1)])
pct_good_decision = num_good_decision / num_opportunities

print(f"\nVal precision: {str(round(precision * 100, 1))}%")
print(f"Test Precision: {str(round(evaluation[1] * 100, 1))}%")
print(f"Percentage of taken opportunities: {str(round(pct_good_decision * 100, 1))}%")

with open(new_dir_name + '/results.txt', 'w') as f:
    f.write(f"Interval: {interval}\n")
    f.write(f"Price delta: {price_delta}\n")
    f.write(f"Num intervals: {num_intervals}\n")
    f.write(f"PCA n_components: {pca.n_components}\n\n")
    f.write(f"Val precision: {str(round(precision * 100, 1))}%\n")
    f.write(f"Test Precision: {str(round(evaluation[1] * 100, 1))}%\n")
    f.write(f"Percentage of taken opportunities: {str(round(pct_good_decision * 100, 1))}%\n\n")
    model.summary(print_fn=lambda x: f.write(x + '\n'))

f.close()

final_dir_name = 'saved_models/model_P-' + str(round(precision * 100, 2)) + '%_' + 'O-' + str(
    round(pct_good_decision * 100, 2)) + '%'

try:
    os.rename(new_dir_name, final_dir_name)
except FileNotFoundError:
    print(f"Directory '{new_dir_name}' not found")
except FileExistsError:
    print(f"Directory '{final_dir_name}' already exists")
    os.rename(new_dir_name, final_dir_name + '_' + timestamp)
except OSError as e:
    print(f"Error: {e}")
