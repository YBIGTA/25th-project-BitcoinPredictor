import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv
import logging
import numpy as np

# .env 파일에서 환경 변수를 로드
load_dotenv('/root/25th-project-BitcoinPredictor/.env')

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

json_file_path = '/home/25th-project-BitcoinPredictor/app/data/prediction_price_data.json'
csv_file_path = '/home/25th-project-BitcoinPredictor/app/data/combined_data_no_timezone.csv'

@router.get("/predict")
async def predict(start_date: str = '2023-08-22 00:00', end_date: str = '2024-08-22 11:59', timeframe: str = 'minute'):
    try:
        logger.info("파일 읽기 시작")

        # JSON 형식의 predict 데이터 읽기
        predict = pd.read_json(json_file_path)
        predict['Date'] = pd.to_datetime(predict['date'])

        # CSV 형식의 true_value 데이터 읽기
        true_value = pd.read_csv(csv_file_path)
        true_value['Date'] = pd.to_datetime(true_value['timestamp'])
        
        logger.info(f"predict 데이터 프레임 헤드: {predict.head()}")
        logger.info(f"true_value 데이터 프레임 헤드: {true_value.head()}")

        # 날짜 범위 설정
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # 날짜 범위 필터링
        true_value = true_value[(true_value['Date'] >= start_date) & (true_value['Date'] <= end_date)]

        # 데이터가 일 단위로 되어 있는지 확인
        if timeframe == 'day':
            true_value = true_value.set_index('Date').resample('D').ffill().reset_index()  # 일 단위로 데이터 재구성

        # # 이동평균선 계산 (20일 및 60일 기준)
        # true_value['MA20'] = true_value['Close'].rolling(window=20).mean()  # 20일 이동평균
        # true_value['MA60'] = true_value['Close'].rolling(window=60).mean()  # 60일 이동평균
        
        # logger.info(f"새로 생성된 MA: {true_value['MA20']}")

        # # RSI 계산 (14일 기준)
        # delta = true_value['Close'].diff(1)
        # gain = delta.where(delta > 0, 0)
        # loss = -delta.where(delta < 0, 0)

        # avg_gain = gain.rolling(window=14).mean()
        # avg_loss = loss.rolling(window=14).mean()

        # rs = avg_gain / avg_loss
        # true_value['RSI'] = 100 - (100 / (1 + rs))

        # 기간별 그룹화
        if timeframe == 'week':
            true_value = true_value.resample('W-MON', on='Date').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).reset_index()
        elif timeframe == 'minute':
            true_value = true_value.resample('T', on='Date').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).reset_index()
        else:  # default to 'day'
            true_value = true_value.resample('D', on='Date').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).reset_index()

        true_value['Date'] = true_value['Date'].astype(str)
        predict['Date'] = predict['Date'].astype(str)
        
        # 데이터프레임을 딕셔너리 리스트로 변환
        true_value_data = true_value.to_dict(orient="records")
        predict_data = predict[['Date', 'price']].to_dict(orient="records")  # 열을 딕셔너리로 변환

        return JSONResponse(content={"true_value": true_value_data, "predict": predict_data})

    except Exception as e:
        logger.error(f"오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))
