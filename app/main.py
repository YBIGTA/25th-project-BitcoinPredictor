from fastapi import FastAPI
from app.core.background_tasks import start_background_tasks
from app.api.routers import endpoints

app = FastAPI()

# 백그라운드 작업 시작
start_background_tasks()

# 라우터 추가
app.include_router(endpoints.router)
