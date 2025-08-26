"""
Minimal FastAPI server for Railway deployment
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="ML Toolbox API",
    description="Virtual Primary Care ML API",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory model storage
MODELS = {}
MODEL_DIR = Path("/tmp/models")
MODEL_DIR.mkdir(exist_ok=True)

# Request/Response models with Pydantic v2 config
class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    model_id: str = Field(default="default", description="Model identifier")
    
    model_config = {
        "protected_namespaces": ()
    }

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    environment: str
    models_loaded: int

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "ML Toolbox API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        environment=os.getenv("ENVIRONMENT", "production"),
        models_loaded=len(MODELS)
    )

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "loaded_models": list(MODELS.keys()),
        "model_files": [f.stem for f in MODEL_DIR.glob("*.pkl")]
    }

@app.post("/models/upload")
async def upload_model(file: UploadFile = File(...), model_id: str = "uploaded_model"):
    """Upload a model file"""
    try:
        content = await file.read()
        model_path = MODEL_DIR / f"{model_id}.pkl"
        
        with open(model_path, "wb") as f:
            f.write(content)
        
        MODELS[model_id] = joblib.load(model_path)
        
        return {"status": "success", "message": f"Model '{model_id}' uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make a prediction"""
    model_id = request.model_id
    
    if model_id not in MODELS:
        model_path = MODEL_DIR / f"{model_id}.pkl"
        if model_path.exists():
            MODELS[model_id] = joblib.load(model_path)
        else:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    try:
        model_data = MODELS[model_id]
        if isinstance(model_data, dict):
            model = model_data.get('model', model_data)
        else:
            model = model_data
        
        features_df = pd.DataFrame([request.features])
        prediction = model.predict(features_df)[0]
        
        if hasattr(prediction, 'item'):
            prediction = prediction.item()
        
        return {
            "prediction": prediction,
            "model_id": model_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/simple")
async def train_simple(file: UploadFile = File(...), target: str = "target", model_id: str = "simple_model"):
    """Train a simple model from CSV"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        
        content = await file.read()
        import io
        df = pd.read_csv(io.BytesIO(content))
        
        X = df.drop(columns=[target])
        y = df[target]
        
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.Categorical(X[col]).codes
        
        X = X.fillna(0)
        
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        model.fit(X, y)
        
        model_data = {
            'model': model,
            'features': list(X.columns),
            'target': target
        }
        
        model_path = MODEL_DIR / f"{model_id}.pkl"
        joblib.dump(model_data, model_path)
        MODELS[model_id] = model_data
        
        return {"status": "success", "message": f"Model '{model_id}' trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
