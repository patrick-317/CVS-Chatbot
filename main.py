from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.routers import recommendation_controller

load_dotenv()

app = FastAPI(title="CVS Honey Combo Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommendation_controller.router)


@app.get("/")
def read_root():
    return {"message": "CVS Honey Combo Chatbot API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
