from fastapi import APIRouter
from routers import predict
from fastapi.templating import Jinja2Templates
from fastapi import Request,FastAPI
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv


load_dotenv("../.env")
app = FastAPI()

# 라우터 추가
app.include_router(predict.router)

# 템플릿 설정
templates = Jinja2Templates(directory="templates")

# 루트 경로("/")에 대한 라우터 정의
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})