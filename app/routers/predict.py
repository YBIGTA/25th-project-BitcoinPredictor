from fastapi import APIRouter, Request
import pandas as pd
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드
load_dotenv('/home/25th-project-BitcoinPredictor/.env')

router = APIRouter()

# 환경 변수를 가져옴
template_url = os.getenv("TEMPLATE_URL")
csv_file_path = "/home/25th-project-BitcoinPredictor/app/data/combined_data.csv"
templates = Jinja2Templates(directory="templates")

@router.get("/predict")
async def predict(start_date: str = '2023-07-01', end_date: str = '2024-7-31', timeframe: str = 'week'):
    try:
        # CSV 파일 읽어오기
        df = pd.read_csv(csv_file_path)

        # 필요한 컬럼 선택 및 날짜 변환
        df['Date'] = pd.to_datetime(df['timestamp'])
        
        # 날짜 범위 필터링 (선택된 날짜 범위가 있을 경우)
        if start_date and end_date:
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        else:
            # 기본적으로 최근 1년간 데이터를 필터링
            df = df[df['Date'] >= (pd.to_datetime("now") - pd.DateOffset(years=1))]

        # 기간별 그룹화
        if timeframe == 'week':
            df = df.resample('W-MON', on='Date').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).reset_index()
        elif timeframe == 'minute':
            df = df.resample('T', on='Date').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).reset_index()
        else:  # default to 'day'
            df = df.resample('D', on='Date').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).reset_index()

        df['Date'] = df['Date'].astype(str)

        # 데이터프레임을 딕셔너리 리스트로 변환
        data = df.to_dict(orient="records")

        return JSONResponse(content={"data": data})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_class=HTMLResponse)
async def get_chart(request: Request):
    return templates.TemplateResponse("visualization.html", {"request": request})