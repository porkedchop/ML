"""
Enhanced API with PostgreSQL persistence, authentication, and monitoring
"""
import os
import json
import time
import pickle
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
import re


from fastapi import FastAPI, HTTPException, File, UploadFile, Security, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor
import io

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# Global variables
MODELS = {}
MODEL_DIR = Path("/tmp/models")
MODEL_DIR.mkdir(exist_ok=True)
START_TIME = time.time()
PREDICTION_COUNT = 0

# Configuration
API_KEY = os.getenv("API_KEY", "your-secret-api-key-here")
DATABASE_URL = os.getenv("DATABASE_URL")
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "false").lower() == "true"

# API key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    if not ENABLE_AUTH:
        return True
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

# PostgreSQL Model Storage
class ModelStorage:
    def __init__(self, database_url: str):
        self.database_url = database_url
        if database_url:
            self.init_db()
    
    def get_connection(self):
        return psycopg2.connect(self.database_url)
    
    def init_db(self):
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS models (
                            model_id VARCHAR(255) PRIMARY KEY,
                            model_data BYTEA,
                            model_type VARCHAR(50),
                            accuracy FLOAT,
                            auc FLOAT,
                            features TEXT,
                            target VARCHAR(100),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS predictions (
                            id SERIAL PRIMARY KEY,
                            model_id VARCHAR(255),
                            features TEXT,
                            prediction FLOAT,
                            probability FLOAT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    """)
                    conn.commit()
        except Exception as e:
            print(f"Database init error: {e}")
    
    def save_model(self, model_id: str, model_data: dict):
        if not self.database_url:
            return False
        try:
            serialized = pickle.dumps(model_data)
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO models (model_id, model_data, model_type, accuracy, auc, features, target)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (model_id) DO UPDATE
                        SET model_data = EXCLUDED.model_data,
                            model_type = EXCLUDED.model_type,
                            accuracy = EXCLUDED.accuracy,
                            auc = EXCLUDED.auc,
                            features = EXCLUDED.features,
                            target = EXCLUDED.target,
                            updated_at = CURRENT_TIMESTAMP
                    """, (
                        model_id, 
                        serialized, 
                        model_data.get('model_type'),
                        model_data.get('accuracy'),
                        model_data.get('auc'),
                        json.dumps(model_data.get('features', [])),
                        model_data.get('target')
                    ))
                    conn.commit()
            return True
        except Exception as e:
            print(f"Save model error: {e}")
            return False
    
    def load_model(self, model_id: str) -> Optional[dict]:
        if not self.database_url:
            return None
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT model_data FROM models WHERE model_id = %s", (model_id,))
                    result = cur.fetchone()
                    if result:
                        return pickle.loads(result[0])
        except Exception as e:
            print(f"Load model error: {e}")
        return None
    
    def list_models(self) -> List[Dict]:
        if not self.database_url:
            return []
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT model_id, model_type, accuracy, auc, target, created_at, updated_at
                        FROM models ORDER BY updated_at DESC
                    """)
                    return cur.fetchall()
        except Exception as e:
            print(f"List models error: {e}")
            return []
    
    def log_prediction(self, model_id: str, features: dict, prediction: Any, probability: float = None):
        if not self.database_url:
            return
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO predictions (model_id, features, prediction, probability)
                        VALUES (%s, %s, %s, %s)
                    """, (model_id, json.dumps(features), float(prediction), probability))
                    conn.commit()
        except Exception as e:
            print(f"Log prediction error: {e}")

# Initialize storage
model_storage = ModelStorage(DATABASE_URL) if DATABASE_URL else None

# Lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"Starting API with database: {bool(DATABASE_URL)}")
    if model_storage:
        # Load models from database
        for model_info in model_storage.list_models():
            model_data = model_storage.load_model(model_info['model_id'])
            if model_data:
                MODELS[model_info['model_id']] = model_data
        print(f"Loaded {len(MODELS)} models from database")
    yield
    # Shutdown
    print("Shutting down API")

# Initialize FastAPI app
app = FastAPI(
    title="ML Toolbox API Enhanced",
    description="Production-ready ML API with persistence and monitoring",
    version="0.3.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data cleaning utilities
class DataCleaner:
    @staticmethod
    def clean_dataframe(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        df_clean = df.copy()
        df_clean = df_clean.drop_duplicates()
        
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
        
        # Handle datetime columns - convert to numeric
        datetime_cols = df_encoded.select_dtypes(include=['datetime64']).columns.tolist()
        for col in datetime_cols:
            # Convert datetime to days since earliest date or drop
            df_encoded[col] = pd.to_numeric(df_encoded[col])
        
        # Handle categorical columns
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

# API Routes
@app.get("/")
async def root():
    return {
        "name": "ML Toolbox API Enhanced",
        "version": "0.3.0",
        "features": ["persistence", "authentication", "monitoring"],
        "docs": "/docs",
        "metrics": "/metrics"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "production"),
        "models_loaded": len(MODELS),
        "database": "connected" if DATABASE_URL else "not configured",
        "auth_enabled": ENABLE_AUTH
    }

@app.get("/metrics")
async def get_metrics():
    """System metrics and monitoring"""
    global PREDICTION_COUNT
    
    uptime = time.time() - START_TIME
    
    metrics = {
        "uptime_seconds": uptime,
        "models_count": len(MODELS),
        "models_in_memory": list(MODELS.keys()),
        "total_predictions": PREDICTION_COUNT,
        "avg_predictions_per_hour": (PREDICTION_COUNT / uptime) * 3600 if uptime > 0 else 0,
    }
    
    if model_storage:
        metrics["models_in_database"] = len(model_storage.list_models())
    
    return metrics

@app.get("/models")
async def list_models():
    """List all available models"""
    response = {
        "loaded_models": list(MODELS.keys()),
        "model_files": [f.stem for f in MODEL_DIR.glob("*.pkl")]
    }
    
    if model_storage:
        response["database_models"] = [m['model_id'] for m in model_storage.list_models()]
    
    return response

@app.get("/models/details/{model_id}")
async def get_model_details(model_id: str):
    """Get detailed model information"""
    if model_id not in MODELS:
        if model_storage:
            model_data = model_storage.load_model(model_id)
            if model_data:
                MODELS[model_id] = model_data
            else:
                raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
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

@app.post("/train/advanced", dependencies=[Depends(verify_api_key)])
async def train_advanced(
    file: UploadFile = File(...),
    target: str = "target",
    model_id: str = "advanced_model",
    model_type: str = "random_forest",
    test_size: float = 0.2,
    clean_data: bool = True,
    scale_features: bool = False
):
    """Train models with authentication"""
    try:
        content = await file.read()
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
        
        # Save to memory
        MODELS[model_id] = model_data
        
        # Save to disk
        model_path = MODEL_DIR / f"{model_id}.pkl"
        joblib.dump(model_data, model_path)
        
        # Save to database
        if model_storage:
            model_storage.save_model(model_id, model_data)
        
        return {
            "status": "success",
            "model_id": model_id,
            "model_type": model_type,
            "accuracy": float(accuracy),
            "auc": float(auc) if auc else None,
            "features": list(X.columns),
            "persisted": bool(model_storage)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make predictions"""
    global PREDICTION_COUNT
    model_id = request.model_id
    
    if model_id not in MODELS:
        # Try loading from database
        if model_storage:
            model_data = model_storage.load_model(model_id)
            if model_data:
                MODELS[model_id] = model_data
            else:
                # Try loading from disk
                model_path = MODEL_DIR / f"{model_id}.pkl"
                if model_path.exists():
                    MODELS[model_id] = joblib.load(model_path)
                else:
                    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        else:
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
        
        probability = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_df)[0]
            probability = float(max(probabilities))
        
        if hasattr(prediction, 'item'):
            prediction = prediction.item()
        
        # Log prediction
        PREDICTION_COUNT += 1
        if model_storage:
            model_storage.log_prediction(model_id, request.features, prediction, probability)
        
        return {
            "prediction": prediction,
            "probability": probability,
            "model_id": model_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/compare", dependencies=[Depends(verify_api_key)])
async def compare_models(
    file: UploadFile = File(...),
    model_ids: str = Query(..., description="Comma-separated model IDs"),
    target: str = "outcome"
):
    """Compare multiple models performance"""
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        if target not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target}' not found")
        
        X = df.drop(columns=[target])
        y = df[target]
        
        # Encode categoricals
        cleaner = DataCleaner()
        X = cleaner.encode_categoricals(X)
        
        model_id_list = [m.strip() for m in model_ids.split(',')]
        results = {}
        
        for model_id in model_id_list:
            if model_id not in MODELS:
                if model_storage:
                    model_data = model_storage.load_model(model_id)
                    if model_data:
                        MODELS[model_id] = model_data
            
            if model_id in MODELS:
                model_data = MODELS[model_id]
                model = model_data.get('model', model_data)
                
                try:
                    y_pred = model.predict(X)
                    
                    metrics = {
                        'accuracy': float(accuracy_score(y, y_pred)),
                        'precision': float(precision_score(y, y_pred, average='weighted')),
                        'recall': float(recall_score(y, y_pred, average='weighted')),
                        'f1': float(f1_score(y, y_pred, average='weighted'))
                    }
                    
                    if hasattr(model, 'predict_proba') and len(np.unique(y)) == 2:
                        try:
                            y_proba = model.predict_proba(X)[:, 1]
                            metrics['auc'] = float(roc_auc_score(y, y_proba))
                        except:
                            pass
                    
                    results[model_id] = metrics
                except Exception as e:
                    results[model_id] = {'error': str(e)}
        
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(
    file: UploadFile = File(...),
    model_id: str = Query(...)
):
    """Batch predictions from CSV"""
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        if model_id not in MODELS:
            if model_storage:
                model_data = model_storage.load_model(model_id)
                if model_data:
                    MODELS[model_id] = model_data
                else:
                    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
            else:
                raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
        
        model_data = MODELS[model_id]
        model = model_data.get('model', model_data)
        
        predictions = model.predict(df)
        
        results = {
            'model_id': model_id,
            'total_predictions': len(predictions),
            'predictions': predictions.tolist()
        }
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)
            if probabilities.shape[1] == 2:
                results['probabilities'] = probabilities[:, 1].tolist()
            else:
                results['probabilities'] = probabilities.max(axis=1).tolist()
        
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# Import the CSV handler
# CSV handler included below

# Add these endpoints to your API
class CSVMapper:
    """Maps various CSV formats to standard schema"""
    
    FIELD_MAPPINGS = {
        'age': ['age', 'patient_age', 'pt_age', 'edad', 'age_years'],
        'gender': ['gender', 'sex', 'patient_gender', 'pt_gender', 'sexo', 'm_f', 'male_female'],
        'appointment_date': ['appointment_date', 'appt_date', 'visit_date', 'fecha_cita', 'scheduled_date'],
        'no_show': ['no_show', 'noshow', 'missed', 'absence', 'did_not_attend', 'dna'],
        'readmitted': ['readmitted', 'readmission', 'readmit', 'readmitted_30days', 'readmission_30'],
        'length_of_stay': ['length_of_stay', 'los', 'days_in_hospital', 'stay_duration'],
    }
    
    @staticmethod
    def detect_delimiter(file_content: str) -> str:
        delimiters = [',', ';', '\t', '|']
        delimiter_counts = {}
        first_line = file_content.split('\n')[0]
        for delimiter in delimiters:
            delimiter_counts[delimiter] = first_line.count(delimiter)
        return max(delimiter_counts, key=delimiter_counts.get)
    
    @staticmethod
    def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
        df.columns = [re.sub(r'[^\w\s]', '', col) for col in df.columns]
        
        for standard, variations in CSVMapper.FIELD_MAPPINGS.items():
            for col in df.columns:
                if col in variations:
                    df.rename(columns={col: standard}, inplace=True)
                    break
        return df
    
    @staticmethod
    def detect_target_column(df: pd.DataFrame) -> Optional[str]:
        potential_targets = []
        for col in df.columns:
            if df[col].nunique() == 2:
                potential_targets.append(col)
            if any(pattern in col.lower() for pattern in ['target', 'label', 'outcome', 'result', 'class', 'no_show', 'readmit', 'mortality']):
                return col
        if len(potential_targets) == 1:
            return potential_targets[0]
        return None
    
    @staticmethod
    def infer_data_types(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        return df
    
    @staticmethod
    def process_csv(file_content: str, target_hint: Optional[str] = None) -> tuple:
        delimiter = CSVMapper.detect_delimiter(file_content)
        df = pd.read_csv(io.StringIO(file_content), delimiter=delimiter)
        df = CSVMapper.standardize_column_names(df)
        df = CSVMapper.infer_data_types(df)
        
        if target_hint:
            target = target_hint
        else:
            target = CSVMapper.detect_target_column(df)
        
        report = {
            'original_shape': df.shape,
            'delimiter_detected': delimiter,
            'columns_mapped': list(df.columns),
            'target_detected': target,
            'data_types': df.dtypes.astype(str).to_dict(),
            'sample_data': df.head(3).to_dict(),
            'processing_suggestions': []
        }
        
        if df.isnull().sum().sum() > 0:
            report['processing_suggestions'].append('Data contains missing values - will be imputed')
        if any(df.select_dtypes(include=['object']).columns):
            report['processing_suggestions'].append('Categorical columns detected - will be encoded')
        
        return report, df
    
@app.post("/data/analyze")
async def analyze_data_format(file: UploadFile = File(...)):
    """Analyze CSV format and suggest mappings"""
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        report, processed_df = CSVMapper.process_csv(content_str)
        
        return {
            "status": "success",
            "analysis": report,
            "ready_for_training": True if report['target_detected'] else False
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/train/auto", dependencies=[Depends(verify_api_key)])
async def train_auto_format(
    file: UploadFile = File(...),
    model_id: str = "auto_model",
    target_hint: Optional[str] = None,
    model_type: str = "random_forest"
):
    """Train with automatic format detection"""
    try:
        content = await file.read()
        
        # Process CSV
        report, df = CSVMapper.process_csv(content.decode('utf-8'), target_hint)
        
        if not report['target_detected']:
            if not target_hint:
                raise HTTPException(
                    status_code=400, 
                    detail="Could not detect target column. Please specify with target_hint parameter."
                )
            target = target_hint
        else:
            target = report['target_detected']
        
        # Clean data
        cleaner = DataCleaner()
        df = cleaner.clean_dataframe(df, target_col=target)
        
        # Prepare features
        X = df.drop(columns=[target])
        y = df[target]
        
        # Encode categoricals
        X = cleaner.encode_categoricals(X)
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Select and train model
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        auc = None
        if len(np.unique(y)) == 2 and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            except:
                pass
        
        # Save model with mapping info
        model_data = {
            'model': model,
            'features': list(X.columns),
            'target': target,
            'model_type': model_type,
            'accuracy': float(accuracy),
            'auc': float(auc) if auc else None,
            'field_mappings': report['columns_mapped'],
            'trained_at': datetime.now().isoformat()
        }
        
        # Save to memory
        MODELS[model_id] = model_data
        
        # Save to disk
        model_path = MODEL_DIR / f"{model_id}.pkl"
        joblib.dump(model_data, model_path)
        
        # Save to database
        if model_storage:
            model_storage.save_model(model_id, model_data)
        
        return {
            "status": "success",
            "model_id": model_id,
            "model_type": model_type,
            "accuracy": float(accuracy),
            "auc": float(auc) if auc else None,
            "target_detected": target,
            "columns_mapped": report['columns_mapped'],
            "features": list(X.columns),
            "persisted": bool(model_storage),
            "message": f"Model trained successfully with auto-detected target '{target}'"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))