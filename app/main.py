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
    # Metadata fields (for display only, not used in model)
    track_name: str = Field(..., description="Name of the track (for display purposes only)")
    year: int = Field(..., ge=1900, le=2030, description="Year of the track (for display purposes only)")
    
    # Model features (these are actually used for prediction)
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
                "track_name": "How to Save a Life",
                "year": 2007,
                "artist_popularity": 75.0,
                "genre": "rock",
                "danceability": 0.735,
                "energy": 0.578,
                "liveness": 0.0985,
                "loudness": 61.0,
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
    
    # Extract metadata (not used in model)
    track_name = data.pop('track_name')
    year = data.pop('year')
    
    # Create DataFrame with only model features
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
            
            # Get proper feature names from the pipeline
            try:
                feature_names = PIPELINE.named_steps["preprocessor"].get_feature_names_out()
                print("DEBUG: Got feature names from preprocessor:", len(feature_names), "features")
            except Exception as e:
                print(f"DEBUG: Could not get feature names from preprocessor: {e}")
                feature_names = [f"f{i}" for i in range(df_processed.shape[1])]
            
            # Convert to DataFrame with proper feature names
            df_processed = pd.DataFrame(df_processed, columns=feature_names)
            print("DEBUG: Sample of features after pipeline:", df_processed.columns.tolist()[:10])
            
            # Get model's expected features (excluding 'const')
            expected_features = MODEL.model.exog_names
            model_features = [f for f in expected_features if f != 'const']
            print("DEBUG: Model expects these features (excluding const):", model_features)
            
            # Select features by name (this is the fix!)
            missing_features = [f for f in model_features if f not in df_processed.columns]
            if missing_features:
                print(f"DEBUG: Missing features: {missing_features}")
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Select the correct features by name
            selected_features = df_processed[model_features]
            print("DEBUG: Selected feature values:", selected_features.iloc[0].to_dict())
            
            # Add constant for statsmodels
            df_final = sm.add_constant(selected_features, has_constant='add')
            print("DEBUG: Final input columns to model:", df_final.columns.tolist())
            print("DEBUG: Final input shape to model:", df_final.shape)
            
            prediction = MODEL.predict(df_final)[0]
        else:
            prediction = MODEL.predict(df)[0]
        
        # Create user-friendly response message with success buckets
        success_percentage = round(float(prediction), 1)
        
        if success_percentage <= 20:
            message = f"'{track_name}' will be a bust! It has an expected success of {success_percentage}%!"
        elif success_percentage <= 40:
            message = f"'{track_name}' will not be successful! It has an expected success of {success_percentage}%!"
        elif success_percentage <= 60:
            message = f"'{track_name}' will maybe be a hit! It has an expected success of {success_percentage}%!"
        elif success_percentage <= 80:
            message = f"'{track_name}' has a great chance! It has an expected success of {success_percentage}%!"
        else:
            message = f"'{track_name}' will be an absolute HIT! It has an expected success of {success_percentage}%!"
        
        return {"message": message}
    except Exception as e:
        print("DEBUG: Exception during prediction:", e)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
def predict_batch(payloads: list[PredictionInput]):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Extract all data
    all_data = [p.dict() for p in payloads]
    
    # Separate metadata from model features
    metadata = []
    model_data = []
    
    for data in all_data:
        track_name = data.pop('track_name')
        year = data.pop('year')
        metadata.append({'track_name': track_name, 'year': year})
        model_data.append(data)
    
    df = pd.DataFrame(model_data)
    
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
            
            # Get proper feature names from the pipeline
            try:
                feature_names = PIPELINE.named_steps["preprocessor"].get_feature_names_out()
            except Exception as e:
                feature_names = [f"f{i}" for i in range(df_processed.shape[1])]
            
            # Convert to DataFrame with proper feature names
            df_processed = pd.DataFrame(df_processed, columns=feature_names)
            
            # Get model's expected features (excluding 'const')
            expected_features = MODEL.model.exog_names
            model_features = [f for f in expected_features if f != 'const']
            
            # Select features by name (same fix as single prediction)
            missing_features = [f for f in model_features if f not in df_processed.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Select the correct features by name
            selected_features = df_processed[model_features]
            
            # Add constant for statsmodels
            df_final = sm.add_constant(selected_features, has_constant='add')
            
            predictions = MODEL.predict(df_final)
        else:
            predictions = MODEL.predict(df)
        
        results = []
        for i, pred in enumerate(predictions):
            success_percentage = round(float(pred), 1)
            track_name = metadata[i]['track_name']
            year = metadata[i]['year']
            
            # Apply same success buckets as single prediction
            if success_percentage <= 20:
                message = f"'{track_name}' will be a bust! It has an expected success of {success_percentage}%!"
            elif success_percentage <= 40:
                message = f"'{track_name}' will not be successful! It has an expected success of {success_percentage}%!"
            elif success_percentage <= 60:
                message = f"'{track_name}' will maybe be a hit! It has an expected success of {success_percentage}%!"
            elif success_percentage <= 80:
                message = f"'{track_name}' has a great chance! It has an expected success of {success_percentage}%!"
            else:
                message = f"'{track_name}' will be an absolute HIT! It has an expected success of {success_percentage}%!"
            
            result = {"message": message}
            results.append(result)
        
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")
