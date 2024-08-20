from merge import CryptoAndMarketDataCollector
import schedule
import time

def update_and_save(collector):
    collector.update_data()  # Update the data
    collector.save_to_csv()  # Save the data to CSV
    collector.save_to_mongo()  # Save the new data to MongoDB

if __name__ == "__main__":
    start_date = input("Enter the start date (YYYY-MM-DD): ")

    collector = CryptoAndMarketDataCollector(crypto_symbol="BTCUSDT", start_date=start_date, fred_api_key="7fe1959a1692174b4b7acdd3b9eb8260")

    # Update and save the data initially
    update_and_save(collector)

    # Schedule to run the update_and_save function every minute
    schedule.every(1).minutes.do(update_and_save, collector=collector)

    # Keep the script running and performing the scheduled tasks
    while True:
        schedule.run_pending()
        time.sleep(1)
