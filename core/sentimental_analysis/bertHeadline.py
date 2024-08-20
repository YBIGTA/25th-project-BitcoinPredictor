from app.db.mongodb import MongoDB
from typing import Dict, Optional, List
from bson import ObjectId

class bertHeadline:
    """
    Rabbitqueue에 있는것을 받아서 
    bertModel에 넣음
    """
    def __init__(self):
        self.mongodb = MongoDB.get_client()
        self.collection_name = "sentimental_headlines"

    def _load_data(self) -> Optional[List[str]]:
        """
        data load하는 부분
        """
        data = None
        return data

    def _bert_sentimental_analysis(self, data):
        """
        감정분석하는 부분
        """
        result = None
        return result
        
    def _save_data(self, data) -> Optional[List[ObjectId]]:
        """
        헤드라인 or 커뮤니티 데이터를 MongoDB에 저장
        :param data: 헤드라인 데이터 리스트
        """
        if isinstance(data, list):
            collection = self.mongo_db[self.collection_name]  # 'self.mongo_db'는 데이터베이스 객체
            inserted_ids = collection.insert_many(data)
            print(f"Inserted {len(inserted_ids)} documents into MongoDB.")
            return inserted_ids
        else:
            print("Data format is not a list of dictionaries.")
            return None
        
    def anaysis_and_store(self) -> Optional[List[ObjectId]]:
        data = self._load_data()
        result = self._bert_sentimental_analysis(data)
        inserted_ids = self._save_data(result)
        return inserted_ids
