from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv("../.env")

# 싱글톤으로 세션 유지
class MongoDB:
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            mongo_uri = os.getenv("MONGODB_URI")
            db_name = os.getenv("MONGODB_DATABASE")
            cls._client = MongoClient(mongo_uri)[db_name]
        return cls._client

# Usage
# db = MongoDB.get_client()