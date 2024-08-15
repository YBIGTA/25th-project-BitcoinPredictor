from fastapi import FastAPI, APIRouter, HTTPException
from app.db.mongodb import MongoDB

router = APIRouter()

@router.get("/stats/")
def read_items():
    """
    경제지표 endpoint
    """
    return [{"item_id": "Foo"}, {"item_id": "Bar"}]

@router.get("/predict/")
def visualization():
    try:
        # MongoDB 클라이언트 가져오기 (싱글톤)
        db = MongoDB.get_client()
        
        # 컬렉션에서 데이터를 로드
        collection = db["final_data"]
        data = list(collection.find({}))
        
        # ObjectId를 문자열로 변환
        for item in data:
            item["_id"] = str(item["_id"])
            
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail="Data could not be retrieved")