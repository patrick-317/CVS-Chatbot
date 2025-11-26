from fastapi import FastAPI
from app.routers import recommendation_controller

app = FastAPI(title="CVS Honey Combo Chatbot API")

app.include_router(recommendation_controller.router)

@app.get("/")
def read_root():
    return {"message": "CVS Honey Combo Chatbot API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

