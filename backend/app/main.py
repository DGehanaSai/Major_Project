from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth import router as auth_router
from app.predict import router as predict_router
from app.advisory import router as advisory_router
from app.chatbot import router as chatbot_router

app = FastAPI(title="AI Agricultural Advisory System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth")
app.include_router(predict_router, prefix="/predict")
app.include_router(advisory_router, prefix="/advisory")
app.include_router(chatbot_router, prefix="/chat")

@app.get("/")
def root():
    return {"status": "Agri Advisory Backend Running"}
