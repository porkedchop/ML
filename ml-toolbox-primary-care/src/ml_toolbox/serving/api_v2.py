"""
Enhanced API with data cleaning and multiple model types
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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Initialize FastAPI app
app = FastAPI(
    title="ML Toolbox API v2",
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

# Model storage
MODELS = {}
MODEL_DIR = Path("/tmp/models")
MODEL_DIR.mkdir(exist_ok=True)

class DataCleaner:
    """Data cleaning utilities"""
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """Clean dataframe with various techniques"""
        df_clean = df.copy()
        
        # Remove duplicate rows
        df_clean = df_clean.drop_duplicates()
        
        # Handle missing values
        # Numeric columns: fill with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        for col in numeric_cols:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
        
        # Categorical columns: fill with mode or 'unknown'
        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            mode = df_clean[col].mode()
            if len(mode) > 0:
                df_clean[col].fillna(mode[0], inplace=True)
            else:
                df_clean[col].fillna('unknown', inplace=True)
        
        # Remove outliers using IQR method for numeric columns
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_clean
    
    @staticmethod
    def encode_categoricals(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """Encode categorical variables"""
        df_encoded = df.copy()
        
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
        if target_col and target_col in categorical_cols:
            categorical_cols.remove(target_col)
        
        for col in categorical_cols:
            # Use label encoding for simplicity
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        return df_encoded

class TrainRequest(BaseModel):
    model_type: str = Field(default="random_forest", description="Model type: random_forest, gradient_boosting, logistic_regression")
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    clean_data: bool = Field(default=True)
    scale_features: bool = Field(default=False)
    
    model_config = {"protected_namespaces": ()}

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "ML Toolbox API v2",
        "version": "0.2.0",
        "features": ["data_cleaning", "multiple_models", "feature_scaling"],
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "production"),
        "models_loaded": len(MODELS)
    }

@app.post("/clean/preview")
async def preview_cleaning(file: UploadFile = File(...)):
    """Preview data cleaning results"""
    try:
        content = await file.read()
        import io
        df_original = pd.read_csv(io.BytesIO(content))
        
        # Clean data
        cleaner = DataCleaner()
        df_cleaned = cleaner.clean_dataframe(df_original)
        
        # Generate cleaning report
        report = {
            "original_shape": df_original.shape,
            "cleaned_shape": df_cleaned.shape,
            "rows_removed": df_original.shape[0] - df_cleaned.shape[0],
            "duplicates_removed": df_original.duplicated().sum(),
            "missing_values_before": df_original.isnull().sum().to_dict(),
            "missing_values_after": df_cleaned.isnull().sum().to_dict(),
            "sample_data": df_cleaned.head().to_dict()
        }
        
        return report
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
        # Read data
        content = await file.read()
        import io
        df = pd.read_csv(io.BytesIO(content))
        
        if target not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target}' not found")
        
        # Clean data if requested
        if clean_data:
            cleaner = DataCleaner()
            df = cleaner.clean_dataframe(df, target_col=target)
        
        # Prepare features and target
        X = df.drop(columns=[target])
        y = df[target]
        
        # Encode categorical features
        cleaner = DataCleaner()
        X = cleaner.encode_categoricals(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) < 10 else None
        )
        
        # Scale features if requested
        scaler = None
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Select model
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        elif model_type == "logistic_regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate AUC if binary classification
        auc = None
        if len(np.unique(y)) == 2 and hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        
        # Save model with metadata
        model_data = {
            'model': model,
            'scaler': scaler,
            'features': list(X.columns),
            'target': target,
            'model_type': model_type,
            'accuracy': accuracy,
            'auc': auc,
            'clean_data': clean_data,
            'scale_features': scale_features,
            'trained_at': datetime.now().isoformat()
        }
        
        model_path = MODEL_DIR / f"{model_id}.pkl"
        joblib.dump(model_data, model_path)
        MODELS[model_id] = model_data
        
        return {
            "status": "success",
            "model_id": model_id,
            "model_type": model_type,
            "accuracy": accuracy,
            "auc": auc,
            "features": list(X.columns),
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/advanced")
async def predict_advanced(
    model_id: str,
    features: Dict[str, Any],
    return_probability: bool = True
):
    """Make predictions with advanced models"""
    if model_id not in MODELS:
        model_path = MODEL_DIR / f"{model_id}.pkl"
        if model_path.exists():
            MODELS[model_id] = joblib.load(model_path)
        else:
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    try:
        model_data = MODELS[model_id]
        model = model_data['model']
        scaler = model_data.get('scaler')
        expected_features = model_data['features']
        
        # Prepare input
        X = pd.DataFrame([features])
        
        # Ensure all expected features are present
        for feat in expected_features:
            if feat not in X.columns:
                X[feat] = 0  # Default value for missing features
        
        X = X[expected_features]  # Ensure correct order
        
        # Scale if necessary
        if scaler:
            X = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Get probability if available
        probability = None
        if return_probability and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[0]
            probability = float(max(probabilities))
        
        # Convert numpy types
        if hasattr(prediction, 'item'):
            prediction = prediction.item()
        
        return {
            "prediction": prediction,
            "probability": probability,
            "model_id": model_id,
            "model_type": model_data.get('model_type'),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        "model_type": model_data.get('model_type'),
        "features": model_data.get('features'),
        "target": model_data.get('target'),
        "accuracy": model_data.get('accuracy'),
        "auc": model_data.get('auc'),
        "clean_data": model_data.get('clean_data'),
        "scale_features": model_data.get('scale_features'),
        "trained_at": model_data.get('trained_at')
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
