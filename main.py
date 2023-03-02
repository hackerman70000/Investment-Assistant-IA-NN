import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
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

price_delta = 1.01

np.seterr(divide='ignore', invalid='ignore')
df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

df['exp'] = np.where((df['Close'] * price_delta <= df['High'].shift(-1))
                     | (df['Close'] * price_delta <= df['High'].shift(-2))
                     | (df['Close'] * price_delta <= df['High'].shift(-3))
                     | (df['Close'] * price_delta <= df['High'].shift(-4))
                     | (df['Close'] * price_delta <= df['High'].shift(-5))
                     | (df['Close'] * price_delta <= df['High'].shift(-6))
                     | (df['Close'] * price_delta <= df['High'].shift(-7)),
                     1, 0)

print('\nCounts of expected values :')
print(df['exp'].value_counts())

df.drop(
    ['High', 'Open', 'Close', 'Low', 'Volume', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
     'Taker buy quote asset volume', 'Ignore', 'Open time', 'Close time'],
    inplace=True, axis=1)

df.dropna(inplace=True)

number_of_prediction_klines = 100
df_train = df.iloc[:df.shape[0] - number_of_prediction_klines, :]
df_dev = df.iloc[df.shape[0] - number_of_prediction_klines:, :]

df_train.reset_index(drop=True, inplace=True)
df_dev.reset_index(drop=True, inplace=True)

x = df_train.iloc[:, :len(df.columns) - 1]
y = df_train['exp']

x_dev = df_dev.iloc[:, :len(df_dev.columns) - 1]
y_dev = df_dev['exp']

print('\nNaNs occurences:')
print(x.isnull().any().any())
print(y.isnull().any().any())
print(x_dev.isnull().any().any())
print(y_dev.isnull().any().any())
print(' ')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(x_train)
joblib.dump(scaler, dir_name + '/scaler.joblib')

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_dev_scaled = scaler.transform(x_dev)

pca = PCA(n_components=16)
pca.fit(x_train_scaled)
joblib.dump(pca, dir_name + '/pca.joblib')

x_train_PCA = pca.transform(x_train_scaled)
x_test_PCA = pca.transform(x_test_scaled)
x_dev_PCA = pca.transform(x_dev_scaled)

model = tf.keras.Sequential(
    [
        # tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(loss='binary_crossentropy', optimizer="Adam", metrics=[tf.keras.metrics.Precision()])

model.fit(x_train_PCA, y_train, batch_size=1, epochs=1, validation_split=0.1, validation_data=None, shuffle=True)
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

m = tf.keras.metrics.Precision()
m.update_state(predictions['target'], predictions['predictions'])

precision = round(m.result().numpy(), 1)
num_decisions = predictions['predictions'].sum()
num_good_decision = len(predictions[(predictions['predictions'] == 1) & (predictions['target'] == 1)])
num_bad_decision = len(predictions[(predictions['predictions'] == 1) & (predictions['target'] == 0)])

print("\nPrecision : " + str(precision))
print("Number of decision taken : " + str(num_decisions))
print("Number of good decision taken : " + str(num_good_decision))
print("Number of bad decision taken : " + str(num_bad_decision))

with open(new_dir_name + '/results.txt', 'w') as f:
    f.write(f"Interval: {interval}\n")
    f.write(f"PCA n_components: {pca.n_components}\n")
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    f.write(f"Precision: {precision}")
    f.write(f"Number of decisions taken: {num_decisions}")
    f.write(f"Number of good decisions taken: {num_good_decision}")
    f.write(f"Number of bad decisions taken: {num_bad_decision}")


f.close()
