from fastapi import APIRouter, FastAPI
import pandas as pd

router = APIRouter()

@router.get("/predict")
async def predict():
    # CSV 파일 읽기
    df = pd.read_csv('/root/25th-project-BitcoinPredictor/app/data/combined.csv')
    
    # JSON 형식으로 변환
    data = df.to_dict(orient='records')
    
    return {"data": data}
