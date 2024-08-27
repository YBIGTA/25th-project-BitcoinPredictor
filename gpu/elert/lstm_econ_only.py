import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pymongo import MongoClient

class EcoLstm:

    def load_and_preprocess_data(self):
        # MongoDB에서 경제 데이터를 로드하고 전처리합니다.
        print("Loading and preprocessing economic data from MongoDB...")
        
        # MongoDB 연결
        client = MongoClient('mongodb://localhost:27017/')
        db = client['trojans']
        collection = db['bitcoin']
        
        # 데이터를 가져오기 (필요한 컬럼만 선택)
        cursor = collection.find({}, {
            "_id": 0,  # _id 필드를 제외
            "timestamp": 1,
            "High": 1,
            "^GSPC": 1,
            "^IXIC": 1,
            "GC=F": 1,
            "^VIX": 1,
            "^TNX": 1,
            "CL=F": 1,
            "EURUSD=X": 1,
            "000001.SS": 1,
            "^N225": 1,
            "^FTSE": 1,
            "^STOXX50E": 1,
            "HG=F": 1,
            "JPY=X": 1,
            "value": 1,
            "value_classification": 1
        })
        
        if cursor: 
            print("몽고디비에서 데이터 추출 성공")
        else: 
            print("안됨")
        # MongoDB 데이터를 DataFrame으로 변환
        economic_df = pd.DataFrame(list(cursor))
        
        # 타임스탬프를 인덱스로 설정하고, 시간대 변환
        economic_df['timestamp'] = pd.to_datetime(economic_df['timestamp'])
        economic_df.set_index('timestamp', inplace=True)
        economic_df.index = economic_df.index.tz_localize('UTC').tz_convert('Asia/Seoul')
        
        # 중복된 타임스탬프 제거 및 정렬
        economic_df = economic_df[~economic_df.index.duplicated(keep='first')]
        economic_df.sort_index(inplace=True)
        
        print(f"Data loaded: {economic_df.shape[0]} rows, {economic_df.shape[1]} columns")
        return economic_df

    def create_sequences(self, data, target_col, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data.iloc[i:(i + seq_length)].values)
            y.append(data.iloc[i + seq_length][target_col])
        return np.array(X), np.array(y)

    def create_model(self, seq_length, num_features):
        # LSTM 모델을 생성합니다.
        main_input = Input(shape=(seq_length, num_features))
        lstm_out = LSTM(50, return_sequences=True)(main_input)
        lstm_out = Dropout(0.3)(lstm_out)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = LSTM(25)(lstm_out)
        lstm_out = Dropout(0.3)(lstm_out)
        lstm_out = BatchNormalization()(lstm_out)
        output = Dense(1)(lstm_out)

        model = Model(inputs=main_input, outputs=output)
        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def train_model(self, model, X_train, y_train, X_test, y_test):
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=64,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
            verbose=1
        )
        return model, history

    def predict_future(self, model, last_sequence, n_minutes, price_scaler):
        predictions = []
        current_sequence = last_sequence.copy()

        for i in range(n_minutes):
            X = current_sequence.reshape((1, current_sequence.shape[0], current_sequence.shape[1]))
            pred = model.predict(X, verbose=0)

            new_datapoint = np.zeros((1, current_sequence.shape[1]))
            new_datapoint[0, 0] = pred[0, 0]
            current_sequence = np.vstack((current_sequence[1:], new_datapoint))

            predictions.append(pred[0, 0])

        predictions = np.array(predictions).reshape(-1, 1)
        predictions = price_scaler.inverse_transform(predictions).flatten()

        return predictions

    def visualize_results(self, original_data, predicted_data, future_predictions, n_minutes):
        last_date = original_data.index[-1]
        future_dates = [last_date + timedelta(minutes=i+1) for i in range(n_minutes)]
        
        plt.figure(figsize=(20, 10))
        plt.plot(original_data.index[-len(predicted_data):], original_data['High'][-len(predicted_data):], label='Actual', linewidth=2)
        plt.plot(original_data.index[-len(predicted_data):], predicted_data, label='Predicted (Training)', color='orange', linewidth=2)
        plt.plot(future_dates, future_predictions, label='Predicted (Future)', color='red', linewidth=2)
        
        plt.title('Bitcoin Price: Actual vs Predicted vs Future', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price (USD)', fontsize=14)
        plt.legend(fontsize=12)
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
        plt.gcf().autofmt_xdate()
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('bitcoin_price_comparison.png', dpi=300)
        plt.close()

    def save_prediction_results(self, future_dates, future_predictions):
        results = [{"date": date.strftime("%Y-%m-%d %H:%M"), "price": float(price)} 
                for date, price in zip(future_dates, future_predictions)]
        
        # MongoDB에 연결하여 prediction 컬렉션에 저장
        client = MongoClient('mongodb://localhost:27017/')
        db = client['trojans']
        collection = db['prediction']
        
        prediction_document = {
            "timestamp": datetime.now().isoformat(),
            "result": results
        }
        
        collection.insert_one(prediction_document)
        print("Prediction results saved to MongoDB in 'prediction' collection.")

    def main(self):
        data = self.load_and_preprocess_data()
        
        target_col = 'High'
        scaler = MinMaxScaler()
        price_scaler = MinMaxScaler()
        
        data['High'] = price_scaler.fit_transform(data[['High']])
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        scaled_numeric_data = pd.DataFrame(scaler.fit_transform(data[numeric_columns]), 
                                        columns=numeric_columns, 
                                        index=data.index)
        
        seq_length = 60
        X, y = self.create_sequences(scaled_numeric_data, target_col, seq_length)

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        num_features = X_train.shape[2]
        model = self.create_model(seq_length, num_features)
        
        model, history = self.train_model(model, X_train, y_train, X_test, y_test)

        model.save('bitcoin_lstm_model_minute_simplified.h5')

        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        all_predictions = np.concatenate([train_predictions, test_predictions])
        all_predictions = price_scaler.inverse_transform(all_predictions).flatten()

        n_minutes = 60 * 24  # 24시간 예측
        last_sequence = X_test[-1]

        future_predictions = self.predict_future(model, last_sequence, n_minutes, price_scaler)

        self.visualize_results(data, all_predictions, future_predictions, n_minutes)

        last_date = data.index[-1]
        future_dates = [last_date + timedelta(minutes=i+1) for i in range(n_minutes)]
        
        self.save_prediction_results(future_dates, future_predictions)

