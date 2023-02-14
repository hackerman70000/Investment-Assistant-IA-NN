import requests
import pandas as pd
import time
import datetime

symbol = "BTCUSDT"
interval = "30m"
limit = 1000

# Pobierz aktualny czas
end_time = int(time.time() * 1000)

# Ustaw początkowy czas na 2021-02-28
start_time = int(datetime.datetime(2021, 2, 28).timestamp() * 1000)

# API endpoint
url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

# Pobierz dane
data = []
while start_time < end_time:
    # Żądanie API z ograniczeniem 1000 wierszy i interwałem 30 minut
    response = requests.get(url + f"&startTime={start_time}&endTime={end_time}")
    klines = response.json()
    data += klines

    # Ustaw nowy początkowy czas
    start_time = klines[-1][6] + 1

# Konwertuj dane na DataFrame Pandas
df = pd.DataFrame(data, columns=["Open time", "Open", "High", "Low", "Close", "Volume", "Close time", "Quote asset volume", "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"])

# Konwertuj czasy na typ datetime
df["Open time"] = pd.to_datetime(df["Open time"], unit='ms')
df["Close time"] = pd.to_datetime(df["Close time"], unit='ms')

# Zapisanie danych do pliku CSV
df.to_csv('market_data.csv', index=False)


