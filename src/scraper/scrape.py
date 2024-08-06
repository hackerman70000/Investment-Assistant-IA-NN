import os
import hashlib
from typing import List, Optional
import pandas as pd
import requests
from zipfile import ZipFile
from datetime import datetime, timedelta
from requests.exceptions import RequestException
from tqdm import tqdm
import logging
import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Scraper:
    DAILY_INTERVALS: frozenset = frozenset(["1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"])
    MONTHLY_INTERVALS: frozenset = frozenset(["1d", "3d", "1w", "1mo"])
    EARLIEST_DATE: datetime = datetime(2017, 8, 17)
    BASE_URL: str = "https://data.binance.vision/data/spot/monthly/klines"
    DAILY_BASE_URL: str = "https://data.binance.vision/data/spot/daily/klines"

    def __init__(self, symbol: str = "BTCUSDT", interval: str = "1d", start_date: datetime = EARLIEST_DATE, end_date: Optional[datetime] = None, directory: str = "data/raw", verify_checksum: bool = False):
        self.symbol: str = symbol.upper()
        self.interval: str = interval
        self.start_date: datetime = start_date
        self.end_date: datetime = end_date or datetime.now()
        self.directory: str = directory
        self.verify_checksum: bool = verify_checksum

        self.data: pd.DataFrame = pd.DataFrame()
        self.validate_inputs()

    def validate_inputs(self) -> None:
        if self.interval not in self.DAILY_INTERVALS.union(self.MONTHLY_INTERVALS):
            raise ValueError(f"Invalid interval: {self.interval}")
        if self.start_date < self.EARLIEST_DATE:
            raise ValueError(f"Invalid start date: {self.start_date}. Data is only available from {self.EARLIEST_DATE}")
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")

    def generate_urls(self) -> List[str]:
        urls: List[str] = []
        current_month: datetime = self.start_date.replace(day=1)
        end_month: datetime = min(self.end_date, datetime.now()).replace(day=1) + timedelta(days=31)
        end_month = end_month.replace(day=1)

        while current_month < end_month:
            year, month = current_month.year, current_month.month
            url: str = f"{self.BASE_URL}/{self.symbol}/{self.interval}/{self.symbol}-{self.interval}-{year}-{month:02d}.zip"
            
            if self.check_url_exists(url):
                urls.append(url)
            elif self.interval in self.DAILY_INTERVALS:
                last_day: datetime = (current_month + timedelta(days=32)).replace(day=1) - timedelta(days=1)
                urls.extend(self.generate_daily_urls(current_month, last_day))
            
            current_month = (current_month + timedelta(days=32)).replace(day=1)

        return urls

    def generate_daily_urls(self, start_date: datetime, end_date: datetime) -> List[str]:
        return [
            f"{self.DAILY_BASE_URL}/{self.symbol}/{self.interval}/{self.symbol}-{self.interval}-{date.strftime('%Y-%m-%d')}.zip"
            for date in (start_date + timedelta(n) for n in range((end_date - start_date).days + 1))
            if date.date() < datetime.now().date()
        ]

    @staticmethod
    def check_url_exists(url: str) -> bool:
        try:
            return requests.head(url, allow_redirects=True).status_code == 200
        except RequestException:
            return False

    @staticmethod
    def download_file(url: str) -> Optional[bytes]:
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.content
        except RequestException as e:
            logging.error(f"Error fetching data from Binance API: {e}")
            return None

    @staticmethod
    def save_file(content: bytes, file_path: str) -> None:
        try:
            with open(file_path, 'wb') as file:
                file.write(content)
        except IOError as e:
            logging.error(f"Error saving file to {file_path}: {e}")

    @staticmethod
    def calculate_sha256(file_path: str) -> Optional[str]:
        sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as file:
                for chunk in iter(lambda: file.read(8192), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except IOError as e:
            logging.error(f"Error calculating checksum for {file_path}: {e}")
            return None

    def verify_checksum_file(self, zip_file_path: str, checksum_file_path: str) -> bool:
        try:
            with open(checksum_file_path, 'r') as file:
                expected_checksum = file.read().split()[0]
            return expected_checksum == self.calculate_sha256(zip_file_path)
        except IOError as e:
            logging.error(f"Error verifying checksum: {e}")
            return False

    def process_zip_file(self, zip_file_path: str) -> pd.DataFrame:
        with ZipFile(zip_file_path) as thezip:
            with thezip.open(thezip.namelist()[0]) as thefile:
                df = pd.read_csv(thefile, header=None, names=[
                    "Open time", "Open", "High", "Low", "Close", "Volume", "Close time",
                    "Quote asset volume", "Number of trades", "Taker buy base asset volume",
                    "Taker buy quote asset volume", "Ignore"
                ])
        
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
        
        # Filter the dataframe to include only rows from the start_date onwards
        df = df[df['Open time'] >= self.start_date]
        
        return df

    def download_and_process_url(self, url: str) -> pd.DataFrame:
        zip_content = self.download_file(url)
        if zip_content is None:
            return pd.DataFrame()

        zip_file_path = os.path.join(self.directory, os.path.basename(url))
        self.save_file(zip_content, zip_file_path)

        if self.verify_checksum:
            checksum_url = url + '.CHECKSUM'
            checksum_content = self.download_file(checksum_url)
            if checksum_content:
                checksum_file_path = os.path.join(self.directory, os.path.basename(checksum_url))
                self.save_file(checksum_content, checksum_file_path)
                if not self.verify_checksum_file(zip_file_path, checksum_file_path):
                    os.remove(zip_file_path)
                    os.remove(checksum_file_path)
                    return pd.DataFrame()
                os.remove(checksum_file_path)

        df = self.process_zip_file(zip_file_path)
        os.remove(zip_file_path)
        return df

    def run(self) -> None:
        urls = self.generate_urls()
        os.makedirs(self.directory, exist_ok=True)

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(self.download_and_process_url, url): url for url in urls}
            for future in tqdm(as_completed(future_to_url), total=len(urls), desc="Processing files", unit="file"):
                df = future.result()
                if not df.empty:
                    self.data = pd.concat([self.data, df], ignore_index=True)

        if not self.data.empty:
            self.data = self.data[self.data['Open time'] <= self.end_date]
            self.data.sort_values('Open time', inplace=True)
            self.save_market_data_to_csv()
        else:
            logging.warning("No data was downloaded.")

    def save_market_data_to_csv(self) -> None:
        if self.data.empty:
            logging.warning("No data to save.")
            return

        start_date = self.data['Open time'].min().strftime('%Y-%m-%d')
        end_date = self.data['Open time'].max().strftime('%Y-%m-%d')
        filename = f"{self.symbol}_{self.interval}_{start_date}_{end_date}.csv"
        path = os.path.join(self.directory, filename)
        try:
            os.makedirs(self.directory, exist_ok=True)  # Ensure the directory exists
            self.data.to_csv(path, index=False)
            logging.info(f"Data has been successfully saved to {path}")
        except IOError as e:
            logging.error(f"Error saving data to {path}: {e}")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape market data from Binance")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--interval", type=str, default="1w", help="Data interval")
    parser.add_argument("--start_date", type=lambda s: datetime.strptime(s, "%Y-%m-%d"), default=Scraper.EARLIEST_DATE, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=lambda s: datetime.strptime(s, "%Y-%m-%d"), default=datetime.now(), help="End date in YYYY-MM-DD format")
    parser.add_argument("--directory", type=str, default="data/raw", help="Directory to save data")
    parser.add_argument("--verify_checksum", action="store_true", help="Verify checksum of downloaded files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    scraper = Scraper(
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        directory=args.directory,
        verify_checksum=args.verify_checksum
    )
    scraper.run()
