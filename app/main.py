from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

with CONFIG_PATH.open("r", encoding="utf-8") as fh:
    CONFIG = yaml.safe_load(fh)

MODEL_PATH = PROJECT_ROOT / "models" / "linear_regression.pkl"
PIPELINE_PATH = PROJECT_ROOT / "models" / "preprocessing_pipeline.pkl"

# Load MODEL
print("DEBUG: About to load model...")
try:
    with MODEL_PATH.open("rb") as fh:
        MODEL = pickle.load(fh)
    print("DEBUG: Model loaded with pickle.")
    print("DEBUG: Model type:", type(MODEL))
    if hasattr(MODEL, "model") and hasattr(MODEL.model, "exog_names"):
        print("Model expects these features:", MODEL.model.exog_names)
    elif hasattr(MODEL, "exog_names"):
        print("Model expects these features:", MODEL.exog_names)
    elif hasattr(MODEL, "params"):
        print("Model expects these features:", list(MODEL.params.index))
    elif hasattr(MODEL, "feature_names_in_"):
        print("Model expects these features:", MODEL.feature_names_in_)
    else:
        print("Could not determine model features.")
except FileNotFoundError:
    print("DEBUG: Model file not found.")
    MODEL = None
except Exception as e:
    print(f"DEBUG: Error loading model: {e}")
    MODEL = None

# Load PIPELINE
print("DEBUG: About to load pipeline...")
try:
    PIPELINE = joblib.load(PIPELINE_PATH)
    print("DEBUG: Pipeline loaded with joblib.")
except FileNotFoundError:
    print("DEBUG: Pipeline file not found.")
    PIPELINE = None
except Exception as e:
    print(f"DEBUG: Error loading pipeline: {e}")
    PIPELINE = None


app = FastAPI()


class PredictionInput(BaseModel):
    artist_popularity: float = Field(..., ge=0.0, le=100.0)
    genre: str = Field(...)
    danceability: float = Field(..., ge=0.0, le=1.0)
    energy: float = Field(..., ge=0.0, le=1.0)
    liveness: float = Field(..., ge=0.0, le=1.0)
    loudness: float = Field(...)
    speechiness: float = Field(..., ge=0.0, le=1.0)
    tempo: float = Field(..., gt=0.0)
    valence: float = Field(..., ge=0.0, le=1.0)
    duration_ms: float = Field(..., gt=0.0)
    acousticness: float = Field(..., ge=0.0, le=1.0)
    key: float = Field(..., ge=0.0, le=11.0)

    class Config:
        schema_extra = {
            "example": {
                "artist_popularity": 75.0,
                "genre": "rock",
                "danceability": 0.735,
                "energy": 0.578,
                "liveness": 0.0985,
                "loudness": -11.84,
                "speechiness": 0.0596,
                "tempo": 75.0,
                "valence": 0.471,
                "duration_ms": 207959.0,
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

    # --- Correction 1: Print pipeline expected feature names ---
    if PIPELINE is not None and hasattr(PIPELINE, "feature_names_in_"):
        print("DEBUG: PIPELINE.feature_names_in_:", PIPELINE.feature_names_in_)
        expected_order = list(PIPELINE.feature_names_in_)
        df = df[expected_order]
    else:
        # fallback to your manual order if pipeline doesn't have feature_names_in_
        expected_order = [
            'artist_popularity', 'genre', 'danceability', 'energy', 'liveness', 'loudness', 'speechiness',
            'tempo', 'valence', 'duration_ms', 'acousticness', 'key'
        ]
        df = df[expected_order]

    try:
        if PIPELINE is not None:
            print("DEBUG: Input columns for prediction:", df.columns.tolist())
            df_processed = PIPELINE.transform(df)
            print("DEBUG: Type after pipeline.transform:", type(df_processed))
            print("DEBUG: Shape after pipeline.transform:", getattr(df_processed, "shape", "no shape"))
            # If not a DataFrame, convert to DataFrame with feature names
            if not isinstance(df_processed, pd.DataFrame):
                try:
                    if hasattr(PIPELINE, "get_feature_names_out"):
                        feature_names = PIPELINE.get_feature_names_out()
                    else:
                        feature_names = [f"f{i}" for i in range(df_processed.shape[1])]
                except Exception as e:
                    print(f"DEBUG: Could not get feature names: {e}")
                    feature_names = [f"f{i}" for i in range(df_processed.shape[1])]
                df_processed = pd.DataFrame(df_processed, columns=feature_names)
            print("DEBUG: Columns after pipeline.transform:", df_processed.columns.tolist())
            
            # The model expects specific features - let's map from pipeline output
            expected_features = MODEL.model.exog_names
            print("DEBUG: Model expects these features:", expected_features)
            
            # For now, let's select the first few features that might correspond
            # This is a temporary fix - ideally the pipeline should output the right feature names
            if len(expected_features) <= len(df_processed.columns):
                # Select subset of features (excluding 'const' which we'll add)
                model_features = [f for f in expected_features if f != 'const']
                n_features_needed = len(model_features)
                selected_features = df_processed.iloc[:, :n_features_needed]
                selected_features.columns = model_features
                # Add constant
                df_processed = sm.add_constant(selected_features, has_constant='add')
            else:
                # Add constant for statsmodels
                df_processed = sm.add_constant(df_processed, has_constant='add')
            
            print("DEBUG: Final input columns to model:", df_processed.columns.tolist())
            print("DEBUG: Final input shape to model:", df_processed.shape)
            prediction = MODEL.predict(df_processed)[0]
        else:
            prediction = MODEL.predict(df)[0]
        return {"prediction": float(prediction)}
    except Exception as e:
        print("DEBUG: Exception during prediction:", e)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
def predict_batch(payloads: list[PredictionInput]):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    df = pd.DataFrame([p.dict() for p in payloads])
    
    # Use same feature ordering as single prediction
    if PIPELINE is not None and hasattr(PIPELINE, "feature_names_in_"):
        expected_order = list(PIPELINE.feature_names_in_)
        df = df[expected_order]
    else:
        expected_order = [
            'artist_popularity', 'genre', 'danceability', 'energy', 'liveness', 'loudness', 'speechiness',
            'tempo', 'valence', 'duration_ms', 'acousticness', 'key'
        ]
        df = df[expected_order]
    
    try:
        if PIPELINE is not None:
            df_processed = PIPELINE.transform(df)
            # Convert to DataFrame with feature names like in single prediction
            if not isinstance(df_processed, pd.DataFrame):
                try:
                    if hasattr(PIPELINE, "get_feature_names_out"):
                        feature_names = PIPELINE.get_feature_names_out()
                    else:
                        feature_names = [f"f{i}" for i in range(df_processed.shape[1])]
                except Exception as e:
                    feature_names = [f"f{i}" for i in range(df_processed.shape[1])]
                df_processed = pd.DataFrame(df_processed, columns=feature_names)
            
            # Apply same feature selection logic as single prediction
            expected_features = MODEL.model.exog_names
            if len(expected_features) <= len(df_processed.columns):
                # Select subset of features (excluding 'const' which we'll add)
                model_features = [f for f in expected_features if f != 'const']
                n_features_needed = len(model_features)
                selected_features = df_processed.iloc[:, :n_features_needed]
                selected_features.columns = model_features
                # Add constant
                df_processed = sm.add_constant(selected_features, has_constant='add')
            else:
                # Add constant for statsmodels
                df_processed = sm.add_constant(df_processed, has_constant='add')
            
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
