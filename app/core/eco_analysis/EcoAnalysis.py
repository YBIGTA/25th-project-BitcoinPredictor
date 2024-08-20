import yfinance as yf
import pandas as pd
from fredapi import Fred
import requests
import schedule
import time

class CryptoAndMarketDataCollector:
    def __init__(self, crypto_symbol="BTCUSDT", start_date="2024-07-01", fred_api_key=None):
        if fred_api_key is None:
            raise ValueError("FRED API key must be provided")
        self.crypto_symbol = crypto_symbol
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.Timestamp.now()
        self.fred_api_key = fred_api_key
        self.bitcoin_data = self.get_binance_minute_data(self.crypto_symbol, self.start_date, self.end_date)
        self.initial_data_merge()

    # Binance API로부터 1분 단위 비트코인 데이터를 가져오는 함수
    def get_binance_minute_data(self, symbol, start_date, end_date):
        base_url = "https://api.binance.com/api/v3/klines"
        interval = "1m"  # 1분 간격
        limit = 1000  # 한 번에 요청할 수 있는 최대 데이터 수
        all_data = []

        start_timestamp = int(start_date.timestamp() * 1000)  # milliseconds
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

            # 수집한 데이터 추가
            all_data.extend(data)

            # 수집된 데이터의 가장 최신 timestamp 가져오기
            last_timestamp = data[-1][0]
            start_timestamp = last_timestamp + 1  # 다음 데이터 수집 시작점

            # 진행 상황 출력
            print(f"Collected data up to: {pd.to_datetime(last_timestamp, unit='ms')}")

        # 데이터프레임으로 변환
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                   'Close time', 'Quote asset volume', 'Number of trades',
                   'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
        df = pd.DataFrame(all_data, columns=columns)

        # 필요한 컬럼만 선택
        df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # 타임스탬프를 datetime으로 변환
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df.set_index('Open time', inplace=True)

        return df

    # Yahoo Finance와 FRED 데이터 초기 수집 및 병합
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
        failed_tickers = []

        for ticker, description in yahoo_indicators.items():
            try:
                data = yf.download(ticker, start=self.start_date, end=self.end_date, interval='1h')
                if not data.empty:
                    data.index = data.index.tz_localize(None)
                    data = data[['Adj Close']].rename(columns={'Adj Close': ticker})
                    self.bitcoin_data = self.bitcoin_data.merge(data, left_index=True, right_index=True, how='left')
                else:
                    failed_tickers.append(ticker)
                    print(f"No data available for {ticker}")
            except Exception as e:
                failed_tickers.append(ticker)
                print(f"Failed to retrieve data for {ticker}: {e}")

        fred = Fred(api_key=self.fred_api_key)
        for series_id, description in fred_indicators.items():
            data = fred.get_series(series_id, observation_start=self.start_date, observation_end=self.end_date)
            if not data.empty:
                df = pd.DataFrame(data, columns=[series_id])
                df.index = pd.to_datetime(df.index)
                df.index = df.index.tz_localize(None)
                df_resampled = df.resample('1h').asfreq().ffill()
                self.bitcoin_data = self.bitcoin_data.merge(df_resampled, left_index=True, right_index=True, how='left')
            else:
                print(f"No data available for {series_id}")

        self.bitcoin_data = self.bitcoin_data.bfill().ffill()

    # 주기적으로 데이터를 업데이트하는 함수
    def update_data(self):
        last_timestamp = self.bitcoin_data.index[-1]
        new_end_date = pd.Timestamp.now()

        # 새로 추가된 데이터만 가져옴
        new_data = self.get_binance_minute_data(self.crypto_symbol, last_timestamp, new_end_date)

        if not new_data.empty:
            # 중복 제거: last_timestamp 이후의 데이터만 추가
            new_data = new_data[new_data.index > last_timestamp]
            if not new_data.empty:
                self.bitcoin_data = pd.concat([self.bitcoin_data, new_data])
                self.bitcoin_data = self.bitcoin_data.bfill().ffill()
                print(f"Data updated up to: {self.bitcoin_data.index[-1]}")
            else:
                print("No new data to append.")
        else:
            print("No new data available.")

        # DataFrame을 CSV 파일로 저장
        output_file = "combined.csv"
        self.bitcoin_data.to_csv(output_file)
        print(f"Data has been saved to {output_file}")

    # 1분마다 데이터를 업데이트하도록 스케줄 설정
    def start_scheduled_updates(self):
        schedule.every(1).minutes.do(self.update_data)
        while True:
            schedule.run_pending()
            time.sleep(1)

# 객체 초기화 및 데이터 수집 시작
collector = CryptoAndMarketDataCollector(crypto_symbol="BTCUSDT", start_date="2024-07-01", fred_api_key="your_fred_api_key")
collector.start_scheduled_updates()
