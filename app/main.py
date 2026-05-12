# app/main.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.config import MODEL_PATH
from src.logger import get_logger

logger = get_logger("api")

app = FastAPI(
    title="Housing Price Prediction API",
    description="MLOps pipeline — predict California housing prices",
    version="1.0.0"
)

# Load model at startup
try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    logger.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Model load failed: {e}")
    model = None

class PredictRequest(BaseModel):
    longitude:          float
    latitude:           float
    housing_median_age: float
    total_rooms:        float
    total_bedrooms:     Optional[float] = 0.0
    population:         float
    households:         float
    median_income:      float
    ocean_proximity_INLAND:           Optional[float] = 0.0
    ocean_proximity_ISLAND:           Optional[float] = 0.0
    ocean_proximity_NEAR_BAY:         Optional[float] = 0.0
    ocean_proximity_NEAR_OCEAN:       Optional[float] = 0.0

class PredictResponse(BaseModel):
    predicted_house_value: float
    model_version:         str = "LinearRegression-v1"
    currency:              str = "USD"

@app.get("/")
def home():
    return {
        "message": "Housing Price Prediction API is running",
        "docs":     "/docs",
        "health":   "/health"
    }

@app.get("/health")
def health():
    return {
        "status":       "healthy" if model else "model not loaded",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame([data.model_dump()])

        # Align columns with training
        model_columns = model.feature_names_in_
        for col in model_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[model_columns]

        prediction = model.predict(df)[0]
        logger.info(f"Prediction: {prediction:.2f}")

        return PredictResponse(predicted_house_value=round(float(prediction), 2))

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))