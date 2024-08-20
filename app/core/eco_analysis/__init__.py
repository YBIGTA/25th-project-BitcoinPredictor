import schedule
import time
import pymongo
import pandas as pd
from datetime import datetime
from fear_class import FearGreedIndexCollector  # fear_class.py 파일에서 FearGreedIndexCollector 클래스 가져오기

# MongoDB 설정
client = pymongo.MongoClient("mongodb://localhost:27017/")  # MongoDB 서버 연결
db = client['crypto_database']  # 사용할 데이터베이스 선택
collection = db['fear_greed_index']  # 사용할 컬렉션 선택

# 1분마다 Fear and Greed Index 데이터를 수집하고 MongoDB에 저장하는 함수
def fetch_and_store_fear_greed_index():
    collector = FearGreedIndexCollector()
    df_minutely = collector.get_fear_greed_index()

    # MongoDB에 데이터 저장
    records = df_minutely.reset_index().to_dict('records')
    if records:
        collection.insert_many(records)
        print(f"{datetime.now()} - Fear and Greed Index data stored in MongoDB")
    else:
        print(f"{datetime.now()} - No data to store")

# 스케줄링 설정
schedule.every(1).minutes.do(fetch_and_store_fear_greed_index)

# 스케줄 실행
while True:
    schedule.run_pending()
    time.sleep(1)
