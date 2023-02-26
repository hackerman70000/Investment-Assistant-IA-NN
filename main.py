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

if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

os.makedirs('saved_models/model_' + timestamp)

df = pd.read_csv('market_data.csv')

df = df.sort_values(by="Open time")
df.reset_index(drop=True, inplace=True)

price_delta = 1.01

np.seterr(divide='ignore', invalid='ignore')
df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

df['exp'] = np.where(  (df['Close'] * price_delta <= df['High'].shift(-1))
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
df_pred = df.iloc[df.shape[0] - number_of_prediction_klines:, :]

df_train.reset_index(drop=True, inplace=True)
df_pred.reset_index(drop=True, inplace=True)

x = df_train.iloc[:, :len(df.columns) - 1]
y = df_train['exp']

x_pred = df_pred.iloc[:, :len(df_pred.columns) - 1]
y_pred = df_pred['exp']

print('\nNaNs occurences:')
print(x.isnull().any().any())
print(y.isnull().any().any())
print(x_pred.isnull().any().any())
print(y_pred.isnull().any().any())
print(' ')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(x_train)
joblib.dump(scaler, 'saved_models/model_' + timestamp + '/scaler.joblib')

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_pred_scaled = scaler.transform(x_pred)

pca = PCA(n_components=16)
pca.fit(x_train_scaled)
joblib.dump(pca, 'saved_models/model_' + timestamp + '/pca.joblib')

x_train_PCA = pca.transform(x_train_scaled)
x_test_PCA = pca.transform(x_test_scaled)
x_pred_PCA = pca.transform(x_pred_scaled)

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

model.save('saved_models/model_' + timestamp + '/NN_' + str(round(evaluation[1] * 100, 1)) + '%_' + timestamp + '.h5')

predictions = pd.DataFrame(model.predict(x_pred_PCA), columns=['0-1'])
predictions['target'] = y_pred
predictions['predictions'] = np.where(predictions['0-1'] > 0.5, 1, 0)

pd.set_option('display.max_rows', None)
print(predictions)

m = tf.keras.metrics.Precision()
m.update_state(predictions['target'], predictions['predictions'])

precision = round(m.result().numpy(), 1)
num_decisions = predictions['predictions'].sum()
pct_good_decisions = str(round(num_decisions / len(predictions['predictions']), 1))

print("\nprecision : " + str(precision))
print("\nNumber of decision taken : " + str(num_decisions))
print("\nPercentage of good decision taken : " + pct_good_decisions)

with open('saved_models/model_' + timestamp + '/results.txt', 'w') as f:
    f.write(f"Precision: {precision}\n")
    f.write(f"Number of decisions taken: {num_decisions}\n")
    f.write(f"Percentage of good decisions taken: {pct_good_decisions}\n")

f.close()
