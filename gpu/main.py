import schedule
import time
from gpu.elert.lstm_econ_only import EcoLstm

def job():
    lstm = EcoLstm()
    lstm.main()

# 1시간에 한 번 main() 함수를 실행하도록 예약
schedule.every(1).hour.do(job)

print("Scheduler started. Press Ctrl+C to stop.")

try:
    while True:
        # 스케줄된 작업이 있는지 확인하고 실행
        schedule.run_pending()
        # 1초 대기 후 다시 체크
        time.sleep(1)
        
except KeyboardInterrupt:
    print("Scheduler stopped.")
