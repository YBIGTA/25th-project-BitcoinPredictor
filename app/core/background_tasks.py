import threading
import time
from app.core.crawling.LoadJsonData import LoadJsonData
from app.db.mongodb import MongoDB
from app.core.sentimental_analysis.bertCommunity import bertCommunity
from app.core.sentimental_analysis.bertHeadline import bertHeadline
from app.core.MessageBroker import MessageBroker
from app.core.eco_analysis.EcoAnalysis import EcoAnalysis

def background_worker():
    while True:
        print("Background task running...")

        time.sleep(10)  # 백그라운드에서 10초마다 실행되는 작업

def start_background_tasks():
    
    community_data_path = "path/to/community_data"
    headline_data_path = "path/to/headline_data"
    
    # 크롤링한 community data의 mongodb id
    community_data = LoadJsonData("community")
    comm_inserted_ids = community_data.process_and_store(community_data_path)
    
    # 크롤링한 headline data의 mongodb id
    headline_data = LoadJsonData("headline")
    headline_inserted_ids = headline_data.process_and_store(headline_data_path)

    # 경제지표 분석
    ea = EcoAnalysis()

    ## 한시간 주기로 데이터 처리 -> background worker 사용하면 될듯함

    ## rabbitMQ (잘 몰라서 이렇게 넣었음)  <- 경제지표 id 추가?
    mb = MessageBroker(comm_inserted_ids, headline_inserted_ids)

    # 병렬로 처리하는 방법 있을지도?
    comm_bert_model = bertCommunity()
    comm_analysis_ids = comm_bert_model.anaysis_and_store()
    
    head_bert_model = bertHeadline()
    head_analysis_ids_ids = head_bert_model.anaysis_and_store()
    

    # 데이터 결합?
    

    # lstm에 넣음


    # 데이터 추출 후 mongodb에 전달
    # final data collection = db["final_data"]
