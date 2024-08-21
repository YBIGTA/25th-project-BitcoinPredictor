from fastapi import APIRouter, Request
import pandas as pd
import mplfinance as mpf
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

router = APIRouter()

template_url = os.getenv("TEMPLATE_URL")
csv_file_path = os.getenv("CSV_FILE_PATH")
templates = Jinja2Templates(directory=template_url)

@router.get("/predict")
async def predict():
    try:
        # CSV 파일 읽어오기
        df = pd.read_csv(csv_file_path)

    
        # 필요한 컬럼 선택
        df['Date'] = pd.to_datetime(df['Open time'])
        df['Date'] = df['Date'].astype(str)  # Timestamp를 문자열로 변환

        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # JSON으로 변환 
        data = df.to_dict(orient="records")
        return JSONResponse(content={"data": data})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_class=HTMLResponse)
async def get_chart(request: Request):
    return templates.TemplateResponse("visualization.html", {"request": request})