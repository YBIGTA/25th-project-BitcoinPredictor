import json
from app.db.mongodb import MongoDB
from typing import List, Dict, Optional
from bson import ObjectId

class LoadJsonData:
    """
    src 파일에 저장된 headline json파일을 가져와서 
    mongodb에 저장
    """
    def __init__(self,collection_name: str):
        """
        생성자: MongoDB 인스턴스와 컬렉션 이름을 받아서 설정
        """
        self.mongo_db = MongoDB.get_client()  # 싱글톤 패턴을 통해 MongoDB 인스턴스 가져오기
        self.collection_name = collection_name

    def _load_json_file(self, json_file_path: str) -> List[Dict[str, str]]:
        """
        JSON 파일을 로드하여 데이터를 반환
        :param json_file_path: JSON 파일 경로
        :return: 
        documents = [
        {"title": "First Headline", "date": "2023-08-15"},
        {"title": "Second Headline", "date": "2023-08-16"},
        {"title": "Third Headline", "date": "2023-08-17"}
        ]
        헤드라인 데이터 리스트
        """
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def _save_data(self, data: Optional[List[Dict[str, str]]]) -> Optional[List[ObjectId]]:
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

    def process_and_store(self, json_file_path: str):
        """
        JSON 파일을 불러와서 MongoDB에 저장하는 전체 과정 처리
        :param json_file_path: JSON 파일 경로
        """
        data = self._load_json_file(json_file_path)
        inserted_ids = self._save_data(data)
        return inserted_ids
