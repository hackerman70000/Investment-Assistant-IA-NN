import datetime
import os
import time

import pandas as pd
import pytz
import requests
from requests.exceptions import RequestException


class market_data_scraper:

    def __init__(self, symbol, interval, limit, directory, filename):
        self.symbol = symbol
        self.interval = interval
        self.limit = limit
        self.timezone = pytz.timezone('Europe/Warsaw')
        self.directory = directory
        self.filename = filename

    def download_market_data(self):
        end_time = int(time.time() * 1000)
        start_time = int(datetime.datetime(2021, 2, 28, tzinfo=pytz.utc).astimezone(self.timezone).timestamp() * 1000)
        url = f"https://data.binance.com/api/v3/klines?symbol={self.symbol}&interval={self.interval}&limit={self.limit}"
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
                url = f"https://data.binance.com/api/v3/klines?symbol={self.symbol}&interval={self.interval}&limit={self.limit}"
                time.sleep(60)

        df = pd.DataFrame(data,
                          columns=["Open time", "Open", "High", "Low", "Close", "Volume", "Close time",
                                   "Quote asset volume", "Number of trades", "Taker buy base asset volume",
                                   "Taker buy quote asset volume", "Ignore"])

        df["Open time"] = pd.to_datetime(df["Open time"], unit='ms').dt.tz_localize(pytz.utc).dt.tz_convert(
            self.timezone)
        df["Close time"] = pd.to_datetime(df["Close time"], unit='ms').dt.tz_localize(pytz.utc).dt.tz_convert(
            self.timezone)

        return df

    def save_market_data_to_csv(self, df):
        path = f"{self.directory}/{self.directory}_{self.interval}.csv"

        try:
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
            df.to_csv(path, index=False)
            print(f"Data has been successfully saved to {path}")
        except Exception as e:
            print(f"Error saving data to {path}: {e}")


if __name__ == '__main__':
    symbol = "BTCUSDT"
    interval = "1h"
    limit = 1000
    directory = "training_data"
    filename = f"market_data_{interval}.csv"

    data_frame = market_data_scraper(symbol, interval, limit, directory, filename).download_market_data()
    market_data_scraper(symbol, interval, limit, directory, filename).save_market_data_to_csv(data_frame)
