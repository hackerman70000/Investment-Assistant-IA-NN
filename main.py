import pandas as pd
import ta

# Wczytanie danych z pliku CSV
df = pd.read_csv('market_data.csv')

df = df.sort_values(by="Open time")
df.reset_index(drop=True, inplace=True)

nan_count = df.isnull().sum().sum()
print('NaNs: ', nan_count)
df.dropna(inplace=True)

print(df.head())
print(df.info())

print(df.shape[0])
