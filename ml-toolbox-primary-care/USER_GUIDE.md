# ML Toolbox for Virtual Primary Care - User Guide

## Overview
This ML Toolbox is a production-ready machine learning platform designed for virtual primary care operations. It provides data cleaning, model training, and prediction capabilities through a REST API deployed on Railway.

## Live API
**Base URL**: https://ml-production-1e3f.up.railway.app
**Documentation**: https://ml-production-1e3f.up.railway.app/docs

## Quick Start

### 1. Check API Health
curl https://ml-production-1e3f.up.railway.app/health

### 2. List Available Models
curl https://ml-production-1e3f.up.railway.app/models

## Core Features

### Data Cleaning
The API automatically handles:
- Missing value imputation (median for numeric, mode for categorical)
- Duplicate removal
- Outlier detection (IQR method)
- Categorical encoding

Preview cleaning effects:
curl -X POST -F "file=@your_data.csv" 'https://ml-production-1e3f.up.railway.app/clean/preview'

### Model Training

#### Simple Training (Random Forest)
curl -X POST -F "file=@data.csv" 'https://ml-production-1e3f.up.railway.app/train/simple?target=outcome&model_id=my_model'

#### Advanced Training (Multiple Algorithms)

Random Forest with cleaning:
curl -X POST -F "file=@data.csv" 'https://ml-production-1e3f.up.railway.app/train/advanced?target=outcome&model_id=rf_model&model_type=random_forest&clean_data=true'

Gradient Boosting with scaling:
curl -X POST -F "file=@data.csv" 'https://ml-production-1e3f.up.railway.app/train/advanced?target=outcome&model_id=gb_model&model_type=gradient_boosting&scale_features=true'

Logistic Regression:
curl -X POST -F "file=@data.csv" 'https://ml-production-1e3f.up.railway.app/train/advanced?target=outcome&model_id=lr_model&model_type=logistic_regression'

Parameters:
- target: Target column name in your CSV
- model_id: Unique identifier for your model
- model_type: random_forest, gradient_boosting, or logistic_regression
- clean_data: Apply automatic data cleaning (default: true)
- scale_features: Normalize features (recommended for logistic regression)
- test_size: Fraction for test set (default: 0.2)

### Making Predictions
curl -X POST https://ml-production-1e3f.up.railway.app/predict -H "Content-Type: application/json" -d '{"features":{"age":45,"gender":0,"prior_no_shows":2,"distance_to_clinic":5.2},"model_id":"your_model_id"}'

### Model Information
curl https://ml-production-1e3f.up.railway.app/models/details/your_model_id

## Healthcare Use Cases

### 1. No-Show Prediction
Predict which patients are likely to miss appointments.

Train model:
curl -X POST -F "file=@data/noshow_dataset.csv" 'https://ml-production-1e3f.up.railway.app/train/advanced?target=no_show&model_id=noshow_predictor&model_type=gradient_boosting'

Predict no-show risk:
curl -X POST https://ml-production-1e3f.up.railway.app/predict -H "Content-Type: application/json" -d '{"features":{"age":35,"appointment_lag":7,"prior_no_shows":1,"distance_km":10.5,"sms_received":1},"model_id":"noshow_predictor"}'

### 2. Hospital Readmission Risk
Identify patients at risk of 30-day readmission.

Train model:
curl -X POST -F "file=@data/readmission_dataset.csv" 'https://ml-production-1e3f.up.railway.app/train/advanced?target=readmitted_30days&model_id=readmission_risk&model_type=logistic_regression&scale_features=true'

Predict readmission risk:
curl -X POST https://ml-production-1e3f.up.railway.app/predict -H "Content-Type: application/json" -d '{"features":{"age":65,"length_of_stay":5,"num_diagnoses":7,"num_medications":15,"er_visits_year":3},"model_id":"readmission_risk"}'

## Data Format Requirements

### CSV Structure
- First row must contain column headers
- Target column must be present for training
- No special characters in column names (spaces are okay)
- Missing values are handled automatically

### Sample Data Format
age,gender,prior_visits,condition,outcome
45,M,3,diabetes,0
62,F,7,hypertension,1
38,M,1,none,0

## CLI Usage (Local)

### Setup
git clone https://github.com/porkedchop/ml-toolbox-primary-care.git
cd ml-toolbox-primary-care
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

### CLI Commands
export API_URL=https://ml-production-1e3f.up.railway.app
python -m ml_toolbox.cli profile --input data.csv
python -m ml_toolbox.cli train-simple --input data.csv --target outcome --name model_name
python -m ml_toolbox.cli list-models --api-url $API_URL

## API Response Examples

### Successful Training Response
{
  "status": "success",
  "model_id": "noshow_gb",
  "model_type": "gradient_boosting",
  "accuracy": 0.85,
  "auc": 0.92,
  "features": ["age", "prior_no_shows", "distance_km"]
}

### Prediction Response
{
  "prediction": 0,
  "model_id": "noshow_gb",
  "timestamp": "2025-08-26T12:00:00"
}

### Model Details Response
{
  "model_id": "noshow_gb",
  "model_type": "gradient_boosting",
  "features": ["age", "prior_no_shows", "distance_km"],
  "target": "no_show",
  "accuracy": 0.85,
  "auc": 0.92,
  "trained_at": "2025-08-26T11:00:00"
}

## Troubleshooting

### Common Issues

1. Model not found error
   - Check model ID spelling
   - List available models first
   - Ensure model was successfully trained

2. Feature mismatch error
   - Ensure prediction features match training features
   - Check feature names are exact matches
   - Missing features default to 0

3. Training fails
   - Verify target column exists
   - Check CSV format is valid
   - Ensure no special characters in column names

4. Predictions return unexpected results
   - Verify feature encoding matches training
   - Check if scaling was used during training
   - Review model accuracy metrics

## Advanced Configuration

### Environment Variables (for deployment)
ENVIRONMENT=production
PORT=8000
MODEL_STORAGE_TYPE=local
LOG_LEVEL=INFO

### Railway Deployment
git add .
git commit -m "Update"
git push origin main
railway logs --tail

## Performance Metrics

### Model Evaluation
- Accuracy: Overall correct predictions
- AUC: Area under ROC curve (binary classification)
- Feature Importance: Available for tree-based models

### API Performance
- Average response time: <200ms
- Maximum file upload: 10MB
- Concurrent requests: Handled automatically by Railway

## Security Considerations
- No authentication currently implemented (add for production)
- Data is temporarily stored during processing
- Models are stored in memory (lost on restart)
- Use HTTPS for all API calls

## Future Enhancements
- Persistent model storage (PostgreSQL)
- User authentication
- Batch predictions
- Model versioning
- A/B testing framework
- Real-time monitoring dashboard

## Support
1. Check API docs: https://ml-production-1e3f.up.railway.app/docs
2. Review this guide
3. Check Railway logs for errors
4. Create GitHub issue for bugs

## License
MIT License - See LICENSE file for details