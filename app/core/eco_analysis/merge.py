import yfinance as yf
import pandas as pd
from fredapi import Fred
import requests
from pymongo import MongoClient
from pandas.api.types import CategoricalDtype

class FearGreedIndexCollector:
    def __init__(self, start_date="2023-07-01"):
        self.start_date = pd.to_datetime(start_date)
        self.df_minutely = self.get_fear_greed_index()

    def get_fear_greed_index(self):
        url = "https://api.alternative.me/fng/?limit=1000&format=json"
        response = requests.get(url)
        data = response.json()

        df = pd.DataFrame(data['data'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['value'] = pd.to_numeric(df['value'])

        categories = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
        cat_type = CategoricalDtype(categories=categories, ordered=True)
        df['value_classification'] = df['value_classification'].astype(cat_type)
        df['value_classification'] = df['value_classification'].cat.codes

        df = df[df['timestamp'] >= self.start_date]
        df = df.drop(columns=['time_until_update'], errors='ignore')

        df_minutely = pd.DataFrame()
        for date in pd.date_range(start=self.start_date, end=pd.Timestamp.now(), freq='D').date:
            daily_data = df[df['timestamp'].dt.date == date]
            if not daily_data.empty:
                time_range = pd.date_range(start=f"{date} 00:00", end=f"{date} 23:59", freq='T')
                daily_minutely = pd.DataFrame({
                    'timestamp': time_range,
                    'value': daily_data['value'].values[0],
                    'value_classification': daily_data['value_classification'].values[0]
                })
            else:
                time_range = pd.date_range(start=f"{date} 00:00", end=f"{date} 23:59", freq='T')
                daily_minutely = pd.DataFrame({
                    'timestamp': time_range,
                    'value': [None] * len(time_range),
                    'value_classification': [None] * len(time_range)
                })

            df_minutely = pd.concat([df_minutely, daily_minutely])

        df_minutely.set_index('timestamp', inplace=True)
        return df_minutely

class CryptoAndMarketDataCollector:
    def __init__(self, crypto_symbol="BTCUSDT", start_date="2024-07-01", fred_api_key=None):
        if fred_api_key is None:
            raise ValueError("FRED API key must be provided")
        self.crypto_symbol = crypto_symbol
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.Timestamp.now()
        self.fred_api_key = fred_api_key
        self.bitcoin_data = self.get_binance_minute_data(self.crypto_symbol, self.start_date, self.end_date)
        self.fear_greed_collector = FearGreedIndexCollector(start_date=start_date)
        self.initial_data_merge()

    def get_binance_minute_data(self, symbol, start_date, end_date):
        base_url = "https://api.binance.com/api/v3/klines"
        interval = "1m"
        limit = 1000
        all_data = []

        start_timestamp = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)

        while start_timestamp < end_timestamp:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
                "startTime": start_timestamp,
                "endTime": end_timestamp
            }
            response = requests.get(base_url, params=params)
            data = response.json()

            if not data:
                print("더 이상 데이터가 없습니다.")
                break

            all_data.extend(data)

            last_timestamp = data[-1][0]
            start_timestamp = last_timestamp + 1

            print(f"Collected data up to: {pd.to_datetime(last_timestamp, unit='ms')}")

        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                   'Close time', 'Quote asset volume', 'Number of trades',
                   'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
        df = pd.DataFrame(all_data, columns=columns)

        df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]

        df['timestamp'] = pd.to_datetime(df['Open time'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    def initial_data_merge(self):
        yahoo_indicators = {
            '^GSPC': 'S&P 500 Index',
            '^IXIC': 'NASDAQ Composite',
            'GC=F': 'Gold Futures',
            '^VIX': 'CBOE Volatility Index',
            '^TNX': '10-Year Treasury Yield',
            'CL=F': 'Crude Oil Prices',
            'EURUSD=X': 'EUR/USD Exchange Rate',
            'BZ=F': 'Brent Crude Oil Prices',
            '000001.SS': 'Shanghai Composite Index',
            '^N225': 'Nikkei 225',
            '^FTSE': 'FTSE 100 Index',
            '^STOXX50E': 'Euro Stoxx 50 Index',
            'HG=F': 'Copper Futures',
            'JPY=X': 'USD/JPY Exchange Rate'
        }
        fred_indicators = {
            'M1SL': 'M1 Money Stock',
            'CPILFESL': 'Consumer Price Index',
            'PAYEMS': 'Total Nonfarm Payroll',
            'INDPRO': 'Industrial Production Index',
            'UNRATE': 'Unemployment Rate',
            'BUSLOANS': 'Commercial and Industrial Loans',
            'M2SL': 'M2 Money Stock',
            'DEXJPUS': 'US Dollar to Japanese Yen Exchange Rate',
            'DJIA': 'Dow Jones Industrial Average',
            'FEDFUNDS': 'Federal Funds Rate',
            'VIXCLS': 'CBOE Volatility Index',
            'GS10': '10-Year Treasury Constant Maturity Rate',
            'DCOILWTICO': 'Crude Oil Prices: West Texas Intermediate (WTI)'
        }

        for ticker in yahoo_indicators:
            try:
                data = yf.download(ticker, start=self.start_date, end=self.end_date, interval='1h')
                if not data.empty:
                    data.index = data.index.tz_localize(None)
                    data = data[['Adj Close']].rename(columns={'Adj Close': ticker})
                    self.bitcoin_data = self.bitcoin_data.merge(data, left_index=True, right_index=True, how='left')
            except Exception as e:
                print(f"Failed to retrieve data for {ticker}: {e}")

        fred = Fred(api_key=self.fred_api_key)
        for series_id in fred_indicators:
            try:
                data = fred.get_series(series_id, observation_start=self.start_date, observation_end=self.end_date)
                if not data.empty:
                    df = pd.DataFrame(data, columns=[series_id])
                    df.index = pd.to_datetime(df.index)
                    df.index = df.index.tz_localize(None)
                    df_resampled = df.resample('1h').asfreq().ffill()
                    self.bitcoin_data = self.bitcoin_data.merge(df_resampled, left_index=True, right_index=True, how='left')
            except Exception as e:
                print(f"Failed to retrieve FRED data for {series_id}: {e}")

        self.bitcoin_data = self.bitcoin_data.merge(self.fear_greed_collector.df_minutely, left_index=True, right_index=True, how='left')
        
        self.bitcoin_data = self.bitcoin_data.bfill().ffill()

    def update_data(self):
        last_timestamp = self.bitcoin_data.index[-1]
        new_end_date = pd.Timestamp.now()

        new_crypto_data = self.get_binance_minute_data(self.crypto_symbol, last_timestamp, new_end_date)

        self.fear_greed_collector.df_minutely = self.fear_greed_collector.get_fear_greed_index()
        new_fear_greed_data = self.fear_greed_collector.df_minutely[self.fear_greed_collector.df_minutely.index > last_timestamp]

        if not new_crypto_data.empty:
            self.bitcoin_data = pd.concat([self.bitcoin_data, new_crypto_data])
        if not new_fear_greed_data.empty:
            self.bitcoin_data.update(new_fear_greed_data)
        
        self.bitcoin_data = self.bitcoin_data.bfill().ffill()
        print(f"Data updated up to: {self.bitcoin_data.index[-1]}")

        print("Latest Bitcoin Data (1 minute):")
        print(self.bitcoin_data.iloc[-1])

    def save_to_csv(self):
        output_file = "combined_data.csv"
        self.bitcoin_data.to_csv(output_file)
        print(f"Data has been saved to {output_file}")

    def save_to_mongo(self, mongo_uri="mongodb://localhost:27017/", db_name="economic", collection_name="eco"):
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]

        records = self.bitcoin_data.reset_index().to_dict('records')
        if records:
            collection.insert_many(records)
            print(f"Data has been successfully saved to MongoDB collection '{collection_name}'.")
        else:
            print("No data to save to MongoDB.")
