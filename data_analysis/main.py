from data_analysis.eco_analysis.merge import CryptoAndMarketDataCollector
from dotenv import load_dotenv
import os
import schedule
import time

dotenv_path = os.path.join('..', '.env')
load_dotenv(dotenv_path)

fred_api_key = os.getenv('FRED_API_KEY')
def main():
    start_date = '2023-07-01'
    collector = CryptoAndMarketDataCollector(crypto_symbol="BTCUSDT", start_date=start_date, fred_api_key=fred_api_key)
    collector.update_and_save()
    
    schedule.every(1).minutes.do(collector.update_and_save, collector=collector)

    
    # Keep the script running and performing the scheduled tasks
    while True:
        schedule.run_pending()
        time.sleep(1)
main()