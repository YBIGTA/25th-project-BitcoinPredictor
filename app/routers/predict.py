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
csv_file_path = os.getenv("CSV_FILE_PATH")
templates = Jinja2Templates(directory="templates")

@router.get("/predict")
async def predict():
    try:
        # CSV 파일 읽어오기
        df = pd.read_csv("/home/25th-project-BitcoinPredictor/app/data/combined_data.csv")

        # 필요한 컬럼 선택
        df['Date'] = pd.to_datetime(df['timestamp'])

        # 일 단위로 그룹화
        daily_data = df.resample('D', on='Date').agg({
            'Open': 'first',     # 일 단위 첫 번째 값 (시가)
            'High': 'max',       # 일 단위 최대값 (최고가)
            'Low': 'min',        # 일 단위 최소값 (최저가)
            'Close': 'last',     # 일 단위 마지막 값 (종가)
            'Volume': 'sum'      # 일 단위 총합 (거래량)
        }).reset_index()

        daily_data['Date'] = daily_data['Date'].astype(str)

        # 데이터프레임을 딕셔너리 리스트로 변환
        data = daily_data.to_dict(orient="records")

        return JSONResponse(content={"data": data})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_class=HTMLResponse)
async def get_chart(request: Request):
    return templates.TemplateResponse("visualization.html", {"request": request})
