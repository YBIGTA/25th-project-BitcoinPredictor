import pandas as pd
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from dotenv import load_dotenv
import logging

# .env 파일에서 환경 변수를 로드
load_dotenv('/home/25th-project-BitcoinPredictor/.env')

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# 환경 변수를 가져옴
template_url = os.getenv("TEMPLATE_URL")
csv_file_path = "/root/25th-project-BitcoinPredictor/app/data/combined_data.csv"
templates = Jinja2Templates(directory="templates")

@router.get("/predict")
async def predict(start_date: str = '2023-07-01', end_date: str = '2024-7-31', timeframe: str = 'week'):
    try:
        logger.info("CSV 파일 읽기 시작")
        df = pd.read_csv(csv_file_path)

        # 필요한 컬럼 선택 및 날짜 변환
        df['Date'] = pd.to_datetime(df['timestamp'])
        
        logger.info("날짜 변환 완료")

        # 날짜 범위 필터링 (선택된 날짜 범위가 있을 경우)
        if start_date and end_date:
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        else:
            df = df[df['Date'] >= (pd.to_datetime("now") - pd.DateOffset(years=1))]

        logger.info(f"필터링 후 데이터 프레임 헤드: {df.head()}")

        # 이동평균선 계산
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        logger.info(f"새로 생성된 MA: {df['MA20']}")

        # RSI 계산 (14일 기준)
        delta = df['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 기간별 그룹화
        if timeframe == 'week':
            df = df.resample('W-MON', on='Date').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum',
                'MA20': 'last',
                'MA60': 'last',
                'RSI': 'last'
            }).reset_index()
        elif timeframe == 'minute':
            df = df.resample('T', on='Date').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum',
                'MA20': 'last',
                'MA60': 'last',
                'RSI': 'last'
            }).reset_index()
        else:  # default to 'day'
            df = df.resample('D', on='Date').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum',
                'MA20': 'last',
                'MA60': 'last',
                'RSI': 'last'
            }).reset_index()

        logger.info(f"그룹화 후 데이터 프레임 헤드: {df.head()}")

        df['Date'] = df['Date'].astype(str)

        # 데이터프레임을 딕셔너리 리스트로 변환
        data = df.to_dict(orient="records")

        return JSONResponse(content={"data": data})

    except Exception as e:
        logger.error(f"오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_class=HTMLResponse)
async def get_chart(request: Request):
    return templates.TemplateResponse("visualization.html", {"request": request})
