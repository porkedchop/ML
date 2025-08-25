# src/ml_toolbox/serving/api_railway.py
"""
Railway-optimized FastAPI server for ML model serving
Lightweight, stateless, with external storage support
"""

import os
import json
import hashlib
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import joblib
import redis
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Railway-specific settings using Pydantic
class Settings(BaseSettings):
    # Railway provides PORT automatically
    port: int = Field(default=8000, env='PORT')
    environment: str = Field(default='production', env='ENVIRONMENT')
    
    # Database (Railway PostgreSQL)
    database_url: Optional[str] = Field(default=None, env='DATABASE_URL')
    
    # Redis (Railway Redis)
    redis_url: Optional[str] = Field(default=None, env='REDIS_URL')
    cache_ttl: int = Field(default=3600, env='CACHE_TTL')
    
    # Model storage
    model_storage_type: str = Field(default='local', env='MODEL_STORAGE_TYPE')  # 'local', 'volume', 's3'
    model_path: str = Field(default='/tmp/models', env='MODEL_PATH')  # /tmp for Railway ephemeral
    
    # S3 (optional)
    aws_access_key_id: Optional[str] = Field(default=None, env='AWS_ACCESS_KEY_ID')
    aws_secret_access_key: Optional[str] = Field(default=None, env='AWS_SECRET_ACCESS_KEY')
    s3_bucket_name: Optional[str] = Field(default=None, env='S3_BUCKET_NAME')
    s3_region: str = Field(default='us-west-2', env='S3_REGION')
    
    # API settings
    api_rate_limit: int = Field(default=100, env='API_RATE_LIMIT')
    cors_origins: str = Field(default='*', env='CORS_ORIGINS')
    max_upload_size: int = Field(default=10, env='MAX_UPLOAD_SIZE_MB')  # MB
    
    # Performance
    max_workers: int = Field(default=1, env='MAX_WORKERS')  # Railway typically single dyno
    model_cache_size: int = Field(default=3, env='MODEL_CACHE_SIZE')

    class Config:
        env_file = '.env'
        case_sensitive = False

settings = Settings()

# Model storage interface
class ModelStorage:
    """Abstract interface for model storage"""
    
    @staticmethod
    def get_storage():
        """Factory method to get appropriate storage backend"""
        if settings.model_storage_type == 's3' and settings.s3_bucket_name:
            return S3ModelStorage()
        else:
            return LocalModelStorage()
    
    def save_model(self, model_name: str, model_data: Any) -> bool:
        raise NotImplementedError
    
    def load_model(self, model_name: str) -> Any:
        raise NotImplementedError
    
    def list_models(self) -> List[str]:
        raise NotImplementedError

class LocalModelStorage(ModelStorage):
    """Local/Volume storage for models"""
    
    def __init__(self):
        self.model_dir = Path(settings.model_path)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model_name: str, model_data: Any) -> bool:
        try:
            model_path = self.model_dir / f"{model_name}.pkl"
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved locally: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
            return False
    
    def load_model(self, model_name: str) -> Any:
        model_path = self.model_dir / f"{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found")
        return joblib.load(model_path)
    
    def list_models(self) -> List[str]:
        return [f.stem for f in self.model_dir.glob("*.pkl")]

class S3ModelStorage(ModelStorage):
    """S3 storage for models"""
    
    def __init__(self):
        import boto3
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.s3_region
        )
        self.bucket = settings.s3_bucket_name
    
    def save_model(self, model_name: str, model_data: Any) -> bool:
        try:
            import io
            buffer = io.BytesIO()
            joblib.dump(model_data, buffer)
            buffer.seek(0)
            
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=f"models/{model_name}.pkl",
                Body=buffer.getvalue()
            )
            logger.info(f"Model saved to S3: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model to S3: {e}")
            return False
    
    def load_model(self, model_name: str) -> Any:
        try:
            import io
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=f"models/{model_name}.pkl"
            )
            buffer = io.BytesIO(response['Body'].read())
            return joblib.load(buffer)
        except Exception as e:
            logger.error(f"Failed to load model from S3: {e}")
            raise FileNotFoundError(f"Model {model_name} not found in S3")
    
    def list_models(self) -> List[str]:
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix="models/",
                Delimiter="/"
            )
            if 'Contents' not in response:
                return []
            return [obj['Key'].split('/')[-1].replace('.pkl', '') 
                    for obj in response['Contents']]
        except Exception:
            return []

# Global instances
MODEL_REGISTRY = {}
redis_client = None
model_storage = None

# Lifespan context manager for Railway
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global redis_client, model_storage
    
    # Startup
    logger.info(f"Starting API server in {settings.environment} environment")
    
    # Initialize Redis if available
    if settings.redis_url:
        try:
            redis_client = redis.from_url(settings.redis_url, decode_responses=True)
            redis_client.ping()
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            redis_client = None
    
    # Initialize model storage
    model_storage = ModelStorage.get_storage()
    
    # Load initial models
    try:
        load_default_models()
    except Exception as e:
        logger.warning(f"Failed to load default models: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server")
    if redis_client:
        redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="ML Toolbox API (Railway)",
    description="Lightweight ML API optimized for Railway deployment",
    version="0.2.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(','),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    model_name: str = "default_model"

class BatchPredictionRequest(BaseModel):
    data: List[Dict[str, Any]]
    model_name: str = "default_model"

class PredictionResponse(BaseModel):
    prediction: Any
    probability: Optional[float] = None
    model_name: str
    cached: bool = False
    timestamp: str

class ModelUploadResponse(BaseModel):
    model_name: str
    status: str
    message: str

# Utility functions
def load_default_models():
    """Load models from storage on startup"""
    global MODEL_REGISTRY
    
    try:
        models = model_storage.list_models()
        for model_name in models[:settings.model_cache_size]:  # Limit to cache size
            try:
                MODEL_REGISTRY[model_name] = model_storage.load_model(model_name)
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
    except Exception as e:
        logger.warning(f"Could not list models: {e}")

def get_cache_key(data: dict, model_name: str) -> str:
    """Generate cache key for prediction"""
    data_str = json.dumps(data, sort_keys=True)
    return f"pred:{model_name}:{hashlib.md5(data_str.encode()).hexdigest()}"

def prepare_features(data: Dict[str, Any]) -> pd.DataFrame:
    """Convert feature dictionary to DataFrame"""
    return pd.DataFrame([data])

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "ML Toolbox API (Railway)",
        "version": "0.2.0",
        "environment": settings.environment,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check for Railway"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.environment,
        "models_loaded": len(MODEL_REGISTRY),
        "redis": "connected" if redis_client else "not available",
        "storage_type": settings.model_storage_type
    }
    
    # Check Redis connection
    if redis_client:
        try:
            redis_client.ping()
        except:
            health_status["redis"] = "disconnected"
            health_status["status"] = "degraded"
    
    return health_status

@app.get("/models")
async def list_models():
    """List available models"""
    try:
        # Get models from storage
        storage_models = model_storage.list_models()
        
        # Get loaded models
        loaded_models = list(MODEL_REGISTRY.keys())
        
        return {
            "loaded_models": loaded_models,
            "available_models": storage_models,
            "cache_size": settings.model_cache_size,
            "storage_type": settings.model_storage_type
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {"loaded_models": list(MODEL_REGISTRY.keys()), "available_models": []}

@app.post("/predict")
async def predict(request: PredictionRequest):
    """Single prediction endpoint"""
    model_name = request.model_name
    
    # Check cache first
    cache_key = None
    if redis_client:
        try:
            cache_key = get_cache_key(request.features, model_name)
            cached_result = redis_client.get(cache_key)
            if cached_result:
                result = json.loads(cached_result)
                result['cached'] = True
                return PredictionResponse(**result)
        except Exception as e:
            logger.warning(f"Cache error: {e}")
    
    # Load model if not in memory
    if model_name not in MODEL_REGISTRY:
        try:
            MODEL_REGISTRY[model_name] = model_storage.load_model(model_name)
            
            # Evict oldest model if cache is full
            if len(MODEL_REGISTRY) > settings.model_cache_size:
                oldest = list(MODEL_REGISTRY.keys())[0]
                del MODEL_REGISTRY[oldest]
                logger.info(f"Evicted model from cache: {oldest}")
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found: {e}")
    
    # Make prediction
    try:
        model_data = MODEL_REGISTRY[model_name]
        model = model_data.get('model', model_data)  # Handle both wrapped and unwrapped models
        
        # Prepare features
        features_df = prepare_features(request.features)
        
        # Get prediction
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_df)[0]
            prediction = int(np.argmax(probabilities))
            probability = float(max(probabilities))
        else:
            prediction = model.predict(features_df)[0]
            probability = None
            
            # Convert numpy types to Python types
            if hasattr(prediction, 'item'):
                prediction = prediction.item()
        
        result = {
            "prediction": prediction,
            "probability": probability,
            "model_name": model_name,
            "cached": False,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Cache result
        if redis_client and cache_key:
            try:
                redis_client.setex(cache_key, settings.cache_ttl, json.dumps(result))
            except Exception as e:
                logger.warning(f"Failed to cache result: {e}")
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    predictions = []
    
    for data in request.data:
        try:
            pred_request = PredictionRequest(features=data, model_name=request.model_name)
            result = await predict(pred_request)
            predictions.append(result.dict())
        except Exception as e:
            predictions.append({
                "error": str(e),
                "features": data
            })
    
    return {
        "predictions": predictions,
        "total": len(predictions),
        "model_name": request.model_name,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    model_name: str = "uploaded_model"
):
    """Upload a model file (Railway has ephemeral storage, use volumes or S3)"""
    
    # Check file size
    if file.size > settings.max_upload_size * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (max {settings.max_upload_size}MB)")
    
    try:
        # Read model file
        content = await file.read()
        
        # Load model to validate
        import io
        model_data = joblib.load(io.BytesIO(content))
        
        # Save to storage
        success = model_storage.save_model(model_name, model_data)
        
        if success:
            # Load into memory if cache has space
            if len(MODEL_REGISTRY) < settings.model_cache_size:
                MODEL_REGISTRY[model_name] = model_data
            
            return ModelUploadResponse(
                model_name=model_name,
                status="success",
                message=f"Model uploaded and saved to {settings.model_storage_type}"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to save model")
            
    except Exception as e:
        logger.error(f"Model upload error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid model file: {e}")

@app.post("/train/csv")
async def train_from_csv(
    file: UploadFile = File(...),
    target_column: str = "target",
    model_name: str = "csv_model",
    background_tasks: BackgroundTasks = None
):
    """Train a simple model from CSV (for demo purposes)"""
    
    try:
        # Read CSV
        content = await file.read()
        import io
        df = pd.read_csv(io.BytesIO(content))
        
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found")
        
        # Simple training (in production, use proper pipeline)
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical columns
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].fillna('missing'))
        
        # Fill numeric missing values
        X = X.fillna(0)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X, y)
        
        # Save model
        model_data = {
            'model': model,
            'features': list(X.columns),
            'target': target_column,
            'trained_at': datetime.utcnow().isoformat()
        }
        
        success = model_storage.save_model(model_name, model_data)
        
        if success:
            MODEL_REGISTRY[model_name] = model_data
            return {
                "status": "success",
                "model_name": model_name,
                "features": list(X.columns),
                "samples": len(X),
                "message": "Model trained and saved successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save model")
            
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/cache/clear")
async def clear_cache():
    """Clear Redis cache"""
    if not redis_client:
        return {"status": "no cache configured"}
    
    try:
        redis_client.flushdb()
        return {"status": "cache cleared"}
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_railway:app",
        host="0.0.0.0",
        port=settings.port,
        workers=settings.max_workers,
        log_level=settings.environment.lower()
    )