"""
Enhanced FastAPI server with data cleaning and multiple models
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

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Initialize FastAPI app
app = FastAPI(
    title="ML Toolbox API",
    description="Virtual Primary Care ML API with Data Cleaning",
    version="0.2.0"
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

# Data cleaning utilities
class DataCleaner:
    @staticmethod
    def clean_dataframe(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        df_clean = df.copy()
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Handle missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        for col in numeric_cols:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
        
        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                mode = df_clean[col].mode()
                if len(mode) > 0:
                    df_clean[col].fillna(mode[0], inplace=True)
                else:
                    df_clean[col].fillna('unknown', inplace=True)
        
        return df_clean
    
    @staticmethod
    def encode_categoricals(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
        if target_col and target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        return df_encoded

# Request/Response models
class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    model_id: str = Field(default="default", description="Model identifier")
    
    model_config = {"protected_namespaces": ()}

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    environment: str
    models_loaded: int

# API Routes
@app.get("/")
async def root():
    return {
        "name": "ML Toolbox API",
        "version": "0.2.0",
        "status": "running",
        "features": ["data_cleaning", "multiple_models"],
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        environment=os.getenv("ENVIRONMENT", "production"),
        models_loaded=len(MODELS)
    )

@app.get("/models")
async def list_models():
    return {
        "loaded_models": list(MODELS.keys()),
        "model_files": [f.stem for f in MODEL_DIR.glob("*.pkl")]
    }

@app.post("/clean/preview")
async def preview_cleaning(file: UploadFile = File(...)):
    """Preview data cleaning results"""
    try:
        content = await file.read()
        import io
        df_original = pd.read_csv(io.BytesIO(content))
        
        cleaner = DataCleaner()
        df_cleaned = cleaner.clean_dataframe(df_original)
        
        report = {
            "original_shape": list(df_original.shape),
            "cleaned_shape": list(df_cleaned.shape),
            "rows_removed": int(df_original.shape[0] - df_cleaned.shape[0]),
            "duplicates_removed": int(df_original.duplicated().sum()),
            "missing_values_before": {k: int(v) for k, v in df_original.isnull().sum().to_dict().items()},
            "missing_values_after": {k: int(v) for k, v in df_cleaned.isnull().sum().to_dict().items()},
        }
        
        return report
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/models/upload")
async def upload_model(file: UploadFile = File(...), model_id: str = "uploaded_model"):
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
            scaler = model_data.get('scaler')
        else:
            model = model_data
            scaler = None
        
        features_df = pd.DataFrame([request.features])
        
        if scaler:
            features_df = scaler.transform(features_df)
        
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

@app.post("/train/advanced")
async def train_advanced(
    file: UploadFile = File(...),
    target: str = "target",
    model_id: str = "advanced_model",
    model_type: str = "random_forest",
    test_size: float = 0.2,
    clean_data: bool = True,
    scale_features: bool = False
):
    """Train advanced models with data cleaning"""
    try:
        content = await file.read()
        import io
        df = pd.read_csv(io.BytesIO(content))
        
        if target not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target}' not found")
        
        if clean_data:
            cleaner = DataCleaner()
            df = cleaner.clean_dataframe(df, target_col=target)
        
        X = df.drop(columns=[target])
        y = df[target]
        
        cleaner = DataCleaner()
        X = cleaner.encode_categoricals(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        scaler = None
        if scale_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        elif model_type == "logistic_regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
        
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        auc = None
        if len(np.unique(y)) == 2 and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            except:
                pass
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'features': list(X.columns),
            'target': target,
            'model_type': model_type,
            'accuracy': float(accuracy),
            'auc': float(auc) if auc else None,
            'trained_at': datetime.now().isoformat()
        }
        
        model_path = MODEL_DIR / f"{model_id}.pkl"
        joblib.dump(model_data, model_path)
        MODELS[model_id] = model_data
        
        return {
            "status": "success",
            "model_id": model_id,
            "model_type": model_type,
            "accuracy": float(accuracy),
            "auc": float(auc) if auc else None,
            "features": list(X.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models/details/{model_id}")
async def get_model_details(model_id: str):
    """Get detailed information about a model"""
    if model_id not in MODELS:
        model_path = MODEL_DIR / f"{model_id}.pkl"
        if model_path.exists():
            MODELS[model_id] = joblib.load(model_path)
        else:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    model_data = MODELS[model_id]
    
    return {
        "model_id": model_id,
        "model_type": model_data.get('model_type', 'unknown'),
        "features": model_data.get('features', []),
        "target": model_data.get('target'),
        "accuracy": model_data.get('accuracy'),
        "auc": model_data.get('auc'),
        "trained_at": model_data.get('trained_at')
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
