# ML Toolbox for Virtual Primary Care - User Guide v2.0

## Overview
Production-ready machine learning platform for virtual primary care operations with PostgreSQL persistence, authentication, and monitoring capabilities. Deployed on Railway with automatic scaling and high availability.

## Live API
Base URL: https://ml-production-1e3f.up.railway.app
API Documentation: https://ml-production-1e3f.up.railway.app/docs
System Metrics: https://ml-production-1e3f.up.railway.app/metrics

## What's New in v2.0
- PostgreSQL persistence: Models survive server restarts
- API key authentication: Secure endpoints protection
- Batch predictions: Process multiple predictions efficiently
- Model comparison: Compare multiple models performance
- Metrics tracking: Monitor system usage and performance
- Database logging: Track all predictions for auditing

## Quick Start

### 1. Check API Health
curl https://ml-production-1e3f.up.railway.app/health | python -m json.tool

### 2. List Available Models
curl https://ml-production-1e3f.up.railway.app/models | python -m json.tool

### 3. View System Metrics
curl https://ml-production-1e3f.up.railway.app/metrics | python -m json.tool

## Authentication

Protected endpoints require API key authentication. Add this header to requests:
-H "X-API-Key: oRObh0_GxO9yYmLw9wUXuGrPZxLOCShVLjKllvsTMow"

Protected endpoints:
- /train/advanced
- /models/compare

Example:
curl -X POST -F "file=@data.csv" \
  -H "X-API-Key: oRObh0_GxO9yYmLw9wUXuGrPZxLOCShVLjKllvsTMow" \
  'https://ml-production-1e3f.up.railway.app/train/advanced?target=outcome&model_id=model'

## Core Features

### Data Cleaning
Automatic preprocessing handles:
- Missing value imputation (median for numeric, mode for categorical)
- Duplicate removal
- Outlier detection (IQR method)
- Categorical encoding
- Feature scaling (optional)

Preview cleaning:
curl -X POST -F "file=@data.csv" \
  'https://ml-production-1e3f.up.railway.app/clean/preview' | python -m json.tool

### Model Training

#### Simple Training (No Authentication Required)
curl -X POST -F "file=@data.csv" \
  'https://ml-production-1e3f.up.railway.app/train/simple?target=outcome&model_id=my_model'

#### Advanced Training (Authentication Required)

Random Forest:
curl -X POST -F "file=@data.csv" \
  -H "X-API-Key: oRObh0_GxO9yYmLw9wUXuGrPZxLOCShVLjKllvsTMow" \
  'https://ml-production-1e3f.up.railway.app/train/advanced?target=outcome&model_id=rf_model&model_type=random_forest&clean_data=true'

Gradient Boosting:
curl -X POST -F "file=@data.csv" \
  -H "X-API-Key: oRObh0_GxO9yYmLw9wUXuGrPZxLOCShVLjKllvsTMow" \
  'https://ml-production-1e3f.up.railway.app/train/advanced?target=outcome&model_id=gb_model&model_type=gradient_boosting&scale_features=true'

Logistic Regression:
curl -X POST -F "file=@data.csv" \
  -H "X-API-Key: oRObh0_GxO9yYmLw9wUXuGrPZxLOCShVLjKllvsTMow" \
  'https://ml-production-1e3f.up.railway.app/train/advanced?target=outcome&model_id=lr_model&model_type=logistic_regression&scale_features=true'

Parameters:
- target: Target column in CSV
- model_id: Unique identifier for model
- model_type: random_forest, gradient_boosting, or logistic_regression
- clean_data: Apply automatic cleaning (default: true)
- scale_features: Normalize features (required for logistic regression)
- test_size: Test set fraction (default: 0.2)

### Making Predictions

#### Single Prediction
curl -X POST https://ml-production-1e3f.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"features":{"age":45,"gender":0},"model_id":"model_name"}' | python -m json.tool

#### Batch Predictions
curl -X POST -F "file=@batch_data.csv" \
  'https://ml-production-1e3f.up.railway.app/predict/batch?model_id=model_name' | python -m json.tool

### Model Management

#### Get Model Details
curl https://ml-production-1e3f.up.railway.app/models/details/model_id | python -m json.tool

#### Compare Models (Authentication Required)
curl -X POST -F "file=@test_data.csv" \
  -H "X-API-Key: oRObh0_GxO9yYmLw9wUXuGrPZxLOCShVLjKllvsTMow" \
  'https://ml-production-1e3f.up.railway.app/models/compare?model_ids=model1,model2,model3&target=outcome' | python -m json.tool

## Healthcare Use Cases

### 1. No-Show Prediction
Train model:
curl -X POST -F "file=@noshow_training.csv" \
  -H "X-API-Key: oRObh0_GxO9yYmLw9wUXuGrPZxLOCShVLjKllvsTMow" \
  'https://ml-production-1e3f.up.railway.app/train/advanced?target=no_show&model_id=noshow_model&model_type=gradient_boosting'

Predict no-show risk:
curl -X POST https://ml-production-1e3f.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features":{
      "age":35,"appointment_lag":7,"prior_no_shows":1,"distance_km":10.5,
      "sms_received":1,"email_reminder":1,"hour_of_day":14,"day_of_week":2,
      "clinic_type":"primary_care","insurance_type":"commercial",
      "socioecon_index":55,"telehealth":0,"rain_mm":0.4,"is_holiday_week":0
    },
    "model_id":"noshow_model"
  }' | python -m json.tool

### 2. Hospital Readmission Risk
Train model:
curl -X POST -F "file=@readmission_training.csv" \
  -H "X-API-Key: oRObh0_GxO9yYmLw9wUXuGrPZxLOCShVLjKllvsTMow" \
  'https://ml-production-1e3f.up.railway.app/train/advanced?target=readmitted_30days&model_id=readmission_model&model_type=logistic_regression&scale_features=true'

Predict readmission:
curl -X POST https://ml-production-1e3f.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features":{
      "age":65,"gender":0,"length_of_stay":7,"num_diagnoses":5,"num_medications":12,
      "er_visits_year":3,"bmi":28.5,"systolic_bp":145,"diastolic_bp":90,"recent_hba1c":7.8,
      "chronic_conditions":3,"smoker":1,"clinic_type":"chronic_care","insurance_type":"medicare"
    },
    "model_id":"readmission_model"
  }' | python -m json.tool

## Data Format Requirements

### CSV Structure
- First row must be column headers
- Target column required for training
- No special characters in column names
- Missing values handled automatically
- UTF-8 encoding recommended

### Sample Format
age,gender,prior_visits,chronic_conditions,outcome
45,M,3,2,0
62,F,7,4,1
38,M,1,0,0

## API Responses

### Training Response
{
  "status": "success",
  "model_id": "model_name",
  "model_type": "gradient_boosting",
  "accuracy": 0.924,
  "auc": 0.663,
  "features": ["age","gender","prior_visits"],
  "persisted": true
}

### Prediction Response
{
  "prediction": 0,
  "probability": 0.237,
  "model_id": "model_name",
  "timestamp": "2025-08-26T12:00:00"
}

### Batch Prediction Response
{
  "model_id": "model_name",
  "total_predictions": 3,
  "predictions": [0,0,1],
  "probabilities": [0.12,0.08,0.89]
}

### Metrics Response
{
  "uptime_seconds": 3600,
  "models_count": 5,
  "models_in_memory": ["model1","model2"],
  "total_predictions": 150,
  "avg_predictions_per_hour": 41.67,
  "models_in_database": 5
}

## CLI Usage

### Setup
git clone https://github.com/porkedchop/ml-toolbox-primary-care.git
cd ml-toolbox-primary-care
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

### Commands
export API_URL=https://ml-production-1e3f.up.railway.app
export API_KEY=oRObh0_GxO9yYmLw9wUXuGrPZxLOCShVLjKllvsTMow

# Profile data
python -m ml_toolbox.cli profile --input data.csv

# Train locally
python -m ml_toolbox.cli train-simple --input data.csv --target outcome --name model

# List remote models
python -m ml_toolbox.cli list-models --api-url $API_URL

# Upload model
python -m ml_toolbox.cli upload --file model.pkl --name model --api-url $API_URL

## Troubleshooting

### Authentication Error
- Verify API_KEY environment variable is set
- Check key matches Railway configuration
- Ensure header format: -H "X-API-Key: oRObh0_GxO9yYmLw9wUXuGrPZxLOCShVLjKllvsTMow"

### Model Not Found
- Check model_id spelling
- List available models first
- Verify model was trained successfully
- Check if model is in database after restart

### Feature Mismatch
- Ensure prediction features match training features exactly
- Check feature names case sensitivity
- Missing features default to 0
- Review model details endpoint for feature list

### Training Failures
- Verify target column exists in CSV
- Check for special characters in column names
- Ensure sufficient data (minimum 50 rows recommended)
- Check data types consistency

### Prediction Issues
- Verify model was trained with same feature encoding
- Check if scaling was applied during training
- Review model accuracy metrics
- Test with known good data first

## Performance Guidelines

### Model Performance
- auth_test model: 92.4% accuracy, 0.59 AUC
- readmission_model_v2: 75.3% accuracy, 0.66 AUC
- AUC > 0.5 indicates better than random
- AUC > 0.7 considered good
- AUC > 0.8 considered excellent

### API Performance
- Response time: <200ms typical
- Max file upload: 10MB
- Concurrent requests: Auto-scaled by Railway
- Database queries: <50ms
- Model loading: <1s from database

## Security

### Current Implementation
- API key authentication for sensitive endpoints
- HTTPS encryption for all traffic
- PostgreSQL database encryption at rest
- Input validation on all endpoints
- Rate limiting via Railway infrastructure

### Best Practices
- Rotate API keys periodically
- Never commit keys to version control
- Use environment variables for secrets
- Monitor access logs regularly
- Implement user-specific keys for production

## Environment Variables

Required in Railway:
API_KEY=oRObh0_GxO9yYmLw9wUXuGrPZxLOCShVLjKllvsTMow
ENABLE_AUTH=true
DATABASE_URL=(automatically set by Railway)

Optional:
LOG_LEVEL=INFO
MODEL_CACHE_SIZE=5
PORT=(automatically set by Railway)

## Database Schema

### models table
- model_id: varchar(255) primary key
- model_data: bytea
- model_type: varchar(50)
- accuracy: float
- auc: float
- features: text
- target: varchar(100)
- created_at: timestamp
- updated_at: timestamp

### predictions table
- id: serial primary key
- model_id: varchar(255)
- features: text
- prediction: float
- probability: float
- created_at: timestamp

## Monitoring

Check system health:
curl https://ml-production-1e3f.up.railway.app/health | python -m json.tool

View metrics:
curl https://ml-production-1e3f.up.railway.app/metrics | python -m json.tool

Railway logs:
railway logs --tail

Database status:
Check "database": "connected" in health endpoint

## Future Roadmap
- Multi-user authentication system
- Model versioning with rollback
- Real-time performance monitoring dashboard
- Automated retraining pipelines
- Feature importance visualization
- A/B testing framework
- FHIR integration for EHR data
- Explainable AI (SHAP/LIME)
- Federated learning support
- Model drift detection

## Support

1. API Documentation: https://ml-production-1e3f.up.railway.app/docs
2. GitHub Issues: https://github.com/porkedchop/ML/issues
3. Railway Status: https://railway.app/dashboard
4. This Guide: Check for updates regularly

## License
MIT License - See LICENSE file
