from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D
)
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import json
import warnings
from sqlalchemy import create_engine
warnings.filterwarnings('ignore')

app = FastAPI(title="TrafficSense Model Transformer", version="2.0.0")

# PostgreSQL connection settings
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "trafficsense")
DB_USER = os.getenv("DB_USER", "trafficsense")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

MODEL_DIR = "/tmp/traffic_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Configuration
SEQUENCE_LENGTH = 30  # Read last 30 minutes
PREDICTION_HORIZONS = [5, 15, 30]  # Predict 5, 15, 30 minutes ahead


class PredictionRequest(BaseModel):
    nama_tol: str
    minutes_ahead: Optional[int] = 5


class MultiHorizonPredictionResponse(BaseModel):
    nama_tol: str
    current_timestamp: str
    current_congestion_index: float
    current_status: str
    predictions: List[Dict]
    has_sufficient_data: bool
    message: Optional[str] = None


class PredictionResponse(BaseModel):
    nama_tol: str
    predicted_congestion_index: float
    predicted_status: str
    predicted_at: str
    confidence: float


class TrainingResponse(BaseModel):
    message: str
    trained_on: int
    models_created: List[str]


def congestion_to_status(index: float) -> str:
    """Convert congestion index to status"""
    index = np.clip(index, 0, 1)
    if index < 0.65:
        return "Low"
    elif index < 0.80:
        return "Medium"
    return "High"


def load_data():
    """Load traffic data from PostgreSQL database"""
    engine = create_engine(DATABASE_URL)
    query = "SELECT * FROM traffic ORDER BY timestamp"
    df = pd.read_sql(query, engine, parse_dates=["timestamp"])
    engine.dispose()
    return df


def create_lag_features(data: np.ndarray, sequence_length: int):
    """Create lag features from time series"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

def build_transformer_model(sequence_length: int):
    inputs = Input(shape=(sequence_length, 1))

    # Transformer Attention
    attention_output = MultiHeadAttention(
        num_heads=4,
        key_dim=16
    )(inputs, inputs)

    # Residual Connection
    x = LayerNormalization(epsilon=1e-6)(
        inputs + attention_output
    )

    # Feed Forward Network
    ffn = Dense(64, activation="relu")(x)
    ffn = Dense(32, activation="relu")(ffn)

    # Residual Connection
    x = LayerNormalization(epsilon=1e-6)(
        x + ffn
    )

    # Pooling
    x = GlobalAveragePooling1D()(x)

    # Output
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


def build_neural_network(input_size: int):
    """Build neural network model"""
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10
    )
    return model


def train_model_for_toll(df_toll: pd.DataFrame, tol_name: str):
    """Train model for specific toll road"""
    if len(df_toll) < SEQUENCE_LENGTH + 10:
        return None, None
    
    # Prepare data
    scaler = MinMaxScaler(feature_range=(0, 1))
    congestion_scaled = scaler.fit_transform(df_toll[['congestion_index']])
    
    X, y = create_lag_features(congestion_scaled.flatten(), SEQUENCE_LENGTH)
    
    if len(X) < 10:
        return None, None
    
    # Reshape for Transformer
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = build_transformer_model(SEQUENCE_LENGTH)

    model.fit(
        X,
        y,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    return model, scaler


def get_latest_sequence(df_toll: pd.DataFrame, sequence_length: int = SEQUENCE_LENGTH):
    """Get latest sequence for prediction"""
    if len(df_toll) < sequence_length:
        return None, None
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    congestion_values = df_toll['congestion_index'].values
    congestion_scaled = scaler.fit_transform(congestion_values.reshape(-1, 1))
    
    latest_seq = congestion_scaled[-sequence_length:].flatten()
    return latest_seq, scaler


def predict_multi_horizon(model, latest_seq: np.ndarray, scaler, horizons: List[int]):
    """Generate multi-horizon predictions using recursive prediction"""
    predictions = []
    current_seq = latest_seq.copy()
    
    for horizon in horizons:
        # Predict step by step to horizon
        temp_seq = current_seq.copy()
        for step in range(horizon):
            # Predict next step
            X_pred = temp_seq.reshape(1, len(temp_seq), 1)
            pred_scaled = model.predict(X_pred)[0]
            # Shift sequence and add new prediction
            temp_seq = np.append(temp_seq[1:], pred_scaled)
        
        # Inverse transform
        pred_value = float(scaler.inverse_transform([[temp_seq[-1]]])[0][0])
        pred_value = np.clip(pred_value, 0, 1)
        predictions.append({
            'horizon_minutes': horizon,
            'predicted_congestion_index': round(pred_value, 4),
            'predicted_status': congestion_to_status(pred_value)
        })
    
    return predictions


@app.get("/health")
def health():
    return {"status": "ok", "service": "model-transformer"}


@app.post("/train", response_model=TrainingResponse)
def train():
    """Train models for all toll roads"""
    try:
        df = load_data()
        trained_tolls = []
        
        for tol_name in df['nama_tol'].unique():
            df_toll = df[df['nama_tol'] == tol_name].reset_index(drop=True)
            model, scaler = train_model_for_toll(df_toll, tol_name)
            
            if model is not None:
                model_path = os.path.join(MODEL_DIR, f"{tol_name}_model.pkl")
                scaler_path = os.path.join(MODEL_DIR, f"{tol_name}_scaler.pkl")
                
                joblib.dump(model, model_path)
                joblib.dump(scaler, scaler_path)
                trained_tolls.append(tol_name)
        
        return TrainingResponse(
            message="Models trained successfully",
            trained_on=len(df),
            models_created=trained_tolls
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-multi-horizon", response_model=MultiHorizonPredictionResponse)
def predict_multi_horizon_endpoint(request: PredictionRequest):
    """Multi-horizon prediction for traffic"""
    try:
        df = load_data()
        
        # Filter by toll road
        df_toll = df[df['nama_tol'] == request.nama_tol]
        if len(df_toll) == 0:
            raise HTTPException(status_code=400, detail=f"Unknown toll road: {request.nama_tol}")
        
        # Check if we have sufficient data
        has_sufficient_data = len(df_toll) >= SEQUENCE_LENGTH
        
        if not has_sufficient_data:
            latest = df_toll.sort_values('timestamp').iloc[-1]
            return MultiHorizonPredictionResponse(
                nama_tol=request.nama_tol,
                current_timestamp=latest['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                current_congestion_index=round(float(latest['congestion_index']), 4),
                current_status=latest['status'],
                predictions=[],
                has_sufficient_data=False,
                message=f"Prediction not available yet. Waiting for sufficient historical data. ({len(df_toll)}/{SEQUENCE_LENGTH})"
            )
        
        # Load or train model
        model_path = os.path.join(MODEL_DIR, f"{request.nama_tol}_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, f"{request.nama_tol}_scaler.pkl")
        
        if not os.path.exists(model_path):
            model, scaler = train_model_for_toll(df_toll.sort_values('timestamp'), request.nama_tol)
            if model is None:
                return MultiHorizonPredictionResponse(
                    nama_tol=request.nama_tol,
                    current_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    current_congestion_index=0,
                    current_status="Low",
                    predictions=[],
                    has_sufficient_data=False,
                    message="Insufficient data to train model"
                )
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
        else:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        
        # Get latest sequence and predict
        df_toll_sorted = df_toll.sort_values('timestamp').reset_index(drop=True)
        latest_seq, latest_scaler = get_latest_sequence(df_toll_sorted, SEQUENCE_LENGTH)
        
        if latest_seq is None:
            raise HTTPException(status_code=400, detail="Cannot prepare sequence")
        
        # Generate predictions
        predictions = predict_multi_horizon(model, latest_seq, latest_scaler, PREDICTION_HORIZONS)
        
        # Get current status
        latest = df_toll_sorted.iloc[-1]
        
        return MultiHorizonPredictionResponse(
            nama_tol=request.nama_tol,
            current_timestamp=latest['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            current_congestion_index=round(float(latest['congestion_index']), 4),
            current_status=latest['status'],
            predictions=predictions,
            has_sufficient_data=True
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Single prediction endpoint (backward compatibility)"""
    try:
        minutes_ahead = request.minutes_ahead or 5
        
        df = load_data()
        df_toll = df[df['nama_tol'] == request.nama_tol]
        
        if len(df_toll) == 0:
            raise HTTPException(status_code=400, detail=f"Unknown toll road: {request.nama_tol}")
        
        # Use multi-horizon prediction
        model_path = os.path.join(MODEL_DIR, f"{request.nama_tol}_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, f"{request.nama_tol}_scaler.pkl")
        
        if not os.path.exists(model_path):
            df_toll_sorted = df_toll.sort_values('timestamp')
            model, scaler = train_model_for_toll(df_toll_sorted, request.nama_tol)
            if model is None:
                raise HTTPException(status_code=400, detail="Insufficient data")
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
        else:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        
        df_toll_sorted = df_toll.sort_values('timestamp').reset_index(drop=True)
        latest_seq, latest_scaler = get_latest_sequence(df_toll_sorted, SEQUENCE_LENGTH)
        
        if latest_seq is None:
            raise HTTPException(status_code=400, detail="Cannot prepare sequence")
        
        # Find closest horizon
        closest_horizon = min(PREDICTION_HORIZONS, key=lambda x: abs(x - minutes_ahead))
        predictions = predict_multi_horizon(model, latest_seq, latest_scaler, [closest_horizon])
        
        pred = predictions[0]
        future_time = datetime.now() + timedelta(minutes=closest_horizon)
        
        return PredictionResponse(
            nama_tol=request.nama_tol,
            predicted_congestion_index=pred['predicted_congestion_index'],
            predicted_status=pred['predicted_status'],
            predicted_at=future_time.strftime("%Y-%m-%d %H:%M:%S"),
            confidence=0.82
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/toll-roads")
def get_toll_roads():
    """Get list of all toll roads"""
    df = load_data()
    return {"toll_roads": sorted(df["nama_tol"].unique().tolist())}


@app.get("/summary")
def get_summary():
    """Get summary statistics"""
    df = load_data()
    summary = []
    for tol, group in df.groupby("nama_tol"):
        latest = group.sort_values("timestamp").iloc[-1]
        summary.append({
            "nama_tol": tol,
            "latest_congestion_index": round(float(latest["congestion_index"]), 4),
            "latest_status": latest["status"],
            "avg_congestion_index": round(float(group["congestion_index"].mean()), 4),
            "total_records": len(group)
        })
    return {"summary": summary}


@app.get("/data-history")
def get_data_history(nama_tol: Optional[str] = None, limit: int = 100):
    """Get historical data"""
    df = load_data()
    if nama_tol:
        df = df[df["nama_tol"] == nama_tol]
    
    df_sorted = df.sort_values("timestamp", ascending=False).head(limit)
    return {
        "data": df_sorted.to_dict(orient="records"),
        "total": len(df)
    }
