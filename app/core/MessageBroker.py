from app.db.mongodb import MongoDB
from typing import Optional, List
from bson import ObjectId

class MessageBroker:
    def __init__(self, comm_inserted_ids:Optional[List[ObjectId]], headline_inserted_ids:Optional[List[ObjectId]]):
        self.mongodb = MongoDB.get_client()
        self.comm_inserted_ids = comm_inserted_ids
        self.headline_inserted_ids = headline_inserted_ids
    
    def _load_data(self):
        """
        크롤링했던 community data와 
        headline data 
        """
        community_docs = [MongoDB.get_client.find_one("community", {"_id": id}) for id in self.comm_inserted_ids]
        headline_docs = [MongoDB.get_client.find_one("headline", {"_id": id}) for id in self.headline_inserted_ids]
    