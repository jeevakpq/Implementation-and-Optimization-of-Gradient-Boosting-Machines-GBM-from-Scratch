from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

app = FastAPI(title="GBM API")

class InputData(BaseModel):
    features: list


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def get_prediction(data: InputData):
    prob, pred = predict([data.features])

    return {
        "prediction": pred,
        "probability": prob
    }