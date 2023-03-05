import datetime
import time
import pandas as pd
import pytz
import requests
from requests.exceptions import RequestException

symbol = "BTCUSDT"
interval = "1h"
limit = 1000

timezone = pytz.timezone('Europe/Warsaw')

end_time = int(time.time() * 1000)
start_time = int(datetime.datetime(2021, 2, 28, tzinfo=pytz.utc).astimezone(timezone).timestamp() * 1000)

url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

data = []

while start_time < end_time:
    try:
        response = requests.get(url + f"&startTime={start_time}&endTime={end_time}")
        response.raise_for_status()
        klines = response.json()
        data += klines
        start_time = klines[-1][6] + 1
    except RequestException as e:
        print("Error occurred:", e)
        time.sleep(60)

df = pd.DataFrame(data,
                  columns=["Open time", "Open", "High", "Low", "Close", "Volume", "Close time", "Quote asset volume",
                           "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"])

df["Open time"] = pd.to_datetime(df["Open time"], unit='ms').dt.tz_localize(pytz.utc).dt.tz_convert(timezone)
df["Close time"] = pd.to_datetime(df["Close time"], unit='ms').dt.tz_localize(pytz.utc).dt.tz_convert(timezone)

try:
    df.to_csv(f'market_data_{interval}.csv', index=False)
    print(f"Data has been successfully saved to market_data_{interval}.csv")
except Exception as e:
    print(f"Error saving data to market_data_{interval}.csv: {e}")

