import threading
import time
from gpu.crawling.LoadJsonData import LoadJsonData
from gpu.sentimental_analysis.bertCommunity import bertCommunity
from gpu.sentimental_analysis.bertHeadline import bertHeadline
from core.eco_analysis.EcoAnalysis import EcoAnalysis


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

# 병렬로 처리하는 방법 있을지도?
comm_bert_model = bertCommunity()
comm_analysis_ids = comm_bert_model.anaysis_and_store()

head_bert_model = bertHeadline()
head_analysis_ids_ids = head_bert_model.anaysis_and_store()


# 데이터 결합?


# lstm에 넣음


# 데이터 추출 후 mongodb에 전달
# final data collection = db["final_data"]
