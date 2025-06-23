from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as fh:
    CONFIG = yaml.safe_load(fh)

MODEL_PATH = PROJECT_ROOT / CONFIG.get("model", {}).get("linear_regression", {}).get(
    "save_path", "models/linear_regression.pkl"
)
PIPELINE_PATH = PROJECT_ROOT / CONFIG.get("artifacts", {}).get(
    "preprocessing_pipeline", "models/preprocessing_pipeline.pkl"
)

try:
    with MODEL_PATH.open("rb") as fh:
        MODEL = pickle.load(fh)
except FileNotFoundError:
    MODEL = None

try:
    with PIPELINE_PATH.open("rb") as fh:
        PIPELINE = joblib.load(fh)
except FileNotFoundError:
    PIPELINE = None

app = FastAPI()


class PredictionInput(BaseModel):
    danceability: float = Field(..., ge=0.0, le=1.0)
    energy: float = Field(..., ge=0.0, le=1.0)
    liveness: float = Field(..., ge=0.0, le=1.0)
    loudness: float = Field(...)
    speechiness: float = Field(..., ge=0.0, le=1.0)
    tempo: float = Field(..., gt=0.0)
    valence: float = Field(..., ge=0.0, le=1.0)
    duration_ms: float = Field(..., gt=0.0)
    artist_popularity: int = Field(..., ge=0, le=100)
    acousticness: float = Field(..., ge=0.0, le=1.0)
    key: float = Field(..., ge=0.0, le=11.0)

    class Config:
        schema_extra = {
            "example": {
                "danceability": 0.735,
                "energy": 0.578,
                "liveness": 0.0985,
                "loudness": -11.84,
                "speechiness": 0.0596,
                "tempo": 75.0,
                "valence": 0.471,
                "duration_ms": 207959.0,
                "artist_popularity": 63,
                "acousticness": 0.514,
                "key": 1.0
            }
        }


@app.get("/")
def root():
    return {"message": "Welcome to the Spotify track popularity prediction API"}


@app.get("/health")
def health():
    model_status = "loaded" if MODEL is not None else "not_loaded"
    pipeline_status = "loaded" if PIPELINE is not None else "not_loaded"
    return {
        "status": "ok",
        "model_status": model_status,
        "pipeline_status": pipeline_status
    }


@app.post("/predict")
def predict(payload: PredictionInput):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    data = payload.dict()
    df = pd.DataFrame([data])
    
    try:
        if PIPELINE is not None:
            df_processed = PIPELINE.transform(df)
            if hasattr(df_processed, 'toarray'):
                df_processed = df_processed.toarray()
            prediction = MODEL.predict(df_processed)[0]
        else:
            prediction = MODEL.predict(df)[0]
        
        prediction_interval = None
        if hasattr(MODEL, 'get_prediction'):
            try:
                pred_result = MODEL.get_prediction(df_processed if PIPELINE else df)
                prediction_interval = {
                    "lower": float(pred_result.conf_int()[0][0]),
                    "upper": float(pred_result.conf_int()[0][1])
                }
            except:
                pass
        
        return {
            "prediction": float(prediction),
            "prediction_interval": prediction_interval
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch")
def predict_batch(payloads: list[PredictionInput]):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    df = pd.DataFrame([p.dict() for p in payloads])
    
    try:
        if PIPELINE is not None:
            df_processed = PIPELINE.transform(df)
            if hasattr(df_processed, 'toarray'):
                df_processed = df_processed.toarray()
            predictions = MODEL.predict(df_processed)
        else:
            predictions = MODEL.predict(df)
        
        results = []
        for i, pred in enumerate(predictions):
            result = {"prediction": float(pred)}
            results.append(result)
        
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")
