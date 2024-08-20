import pandas as pd
import requests

class FearGreedIndexCollector:
    def __init__(self, start_date="2024-07-01"):
        self.start_date = pd.to_datetime(start_date)
        self.df_minutely = self.get_fear_greed_index()

    # Fear and Greed Index 데이터를 가져오는 함수
    def get_fear_greed_index(self):
        url = "https://api.alternative.me/fng/?limit=1000&format=json"
        response = requests.get(url)
        data = response.json()

        # 필요한 데이터를 추출하여 DataFrame으로 변환
        df = pd.DataFrame(data['data'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['value'] = pd.to_numeric(df['value'])
        df['value_classification'] = df['value_classification'].astype('category')

        # 2024년 7월 1일부터 현재까지의 데이터 필터링
        df = df[df['timestamp'] >= self.start_date]

        # time_until_update 열 제거
        df = df.drop(columns=['time_until_update'], errors='ignore')

        # 1분 단위 시계열로 확장
        df_minutely = pd.DataFrame()

        for date in pd.date_range(start=self.start_date, end=pd.Timestamp.now(), freq='D').date:
            daily_data = df[df['timestamp'].dt.date == date]
            if not daily_data.empty:
                # 00:00부터 23:59까지의 1분 단위 시간 생성
                time_range = pd.date_range(start=f"{date} 00:00", end=f"{date} 23:59", freq='T')
                daily_minutely = pd.DataFrame({
                    'Open time': time_range,
                    'value': daily_data['value'].values[0],
                    'value_classification': daily_data['value_classification'].values[0]
                })
            else:
                # 데이터가 없는 경우, 1분 단위로 빈 데이터 생성
                time_range = pd.date_range(start=f"{date} 00:00", end=f"{date} 23:59", freq='T')
                daily_minutely = pd.DataFrame({
                    'Open time': time_range,
                    'value': [None] * len(time_range),
                    'value_classification': [None] * len(time_range)
                })

            # 1분 단위 데이터 추가
            df_minutely = pd.concat([df_minutely, daily_minutely])

        df_minutely.set_index('Open time', inplace=True)
        return df_minutely

# 객체 초기화 및 1분 단위 시계열 데이터 가져오기
collector = FearGreedIndexCollector()
collector.df_minutely = collector.get_fear_greed_index()

# DataFrame을 CSV 파일로 저장
output_file = "fear_greed_index_minutely.csv"
collector.df_minutely.to_csv(output_file)

# 결과 확인
print(collector.df_minutely.head())
print(f"\nData has been saved to {output_file}")