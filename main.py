import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features

df = pd.read_csv('market_data.csv')

df = df.sort_values(by="Open time")
df.reset_index(drop=True, inplace=True)

price_delta = 1.03

df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

df['exp'] = np.where((df['Close'] * price_delta <= df['High'].shift(-1))
                     | (df['Close'] * price_delta <= df['High'].shift(-2))
                     | (df['Close'] * price_delta <= df['High'].shift(-3))
                     | (df['Close'] * price_delta <= df['High'].shift(-4))
                     | (df['Close'] * price_delta <= df['High'].shift(-5))
                     , 1, 0)

print(df['exp'].value_counts())
df.drop(
    ['High', 'Open', 'Close', 'Low', 'Volume', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
     'Taker buy quote asset volume', 'Ignore', 'Open time', 'Close time'],
    inplace=True, axis=1)

df.dropna(inplace=True)

number_of_klines = 100
df_train = df.iloc[:df.shape[0] - number_of_klines, :]
df_pred = df.iloc[df.shape[0] - number_of_klines:, :]

df_train.reset_index(drop=True, inplace=True)
df_pred.reset_index(drop=True, inplace=True)

x = df_train.iloc[:, :len(df.columns) - 1]
y = df_train['exp']

x_pred = df_pred.iloc[:, :len(df_pred.columns) - 1]
y_pred = df_pred['exp']

print(x.info())

print(x.isnull().any().any())
print(y.isnull().any().any())

print(x_pred.isnull().any().any())
print(y_pred.isnull().any().any())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(x_train)

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_pred_scaled = scaler.transform(x_pred)

pca = PCA(n_components=16)
pca.fit(x_train_scaled)

x_train_PCA = pca.transform(x_train_scaled)
x_test_PCA = pca.transform(x_test_scaled)
x_pred_PCA = pca.transform(x_pred_scaled)

scaler.fit(x_train_PCA)

x_train_scaled_PCA = scaler.transform(x_train_PCA)
x_test_scaled_PCA = scaler.transform(x_test_PCA)
x_pred_scaled_PCA = scaler.transform(x_pred_PCA)

print(x_train_scaled_PCA)
