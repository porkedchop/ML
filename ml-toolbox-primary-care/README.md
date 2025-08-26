# ML Toolbox for Virtual Primary Care

Production-ready machine learning platform for virtual primary care operations. Features PostgreSQL persistence, API authentication, batch processing, and comprehensive monitoring.

## Live Deployment

Base URL: https://ml-production-1e3f.up.railway.app
API Docs: https://ml-production-1e3f.up.railway.app/docs
Metrics: https://ml-production-1e3f.up.railway.app/metrics

## Current Features

### Core Capabilities
- Multiple ML algorithms (Random Forest, Gradient Boosting, Logistic Regression)
- Automatic data cleaning and preprocessing
- PostgreSQL model persistence (survives restarts)
- API key authentication for protected endpoints
- Batch prediction processing
- Model comparison tools
- Real-time metrics tracking

### Healthcare Models
- No-show prediction (92.4% accuracy)
- 30-day readmission risk (75.3% accuracy)
- Custom model training for any CSV dataset

## Quick Start

### 1. Clone and Setup
git clone https://github.com/porkedchop/ML.git
cd ml-toolbox-primary-care
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

### 2. Test Locally
python -m uvicorn src.ml_toolbox.serving.api:app --reload --port 8000

### 3. Use the API
# Check health
curl https://ml-production-1e3f.up.railway.app/health | python -m json.tool

# List models
curl https://ml-production-1e3f.up.railway.app/models | python -m json.tool

# Train model (requires API key)
curl -X POST -F "file=@data.csv" -H "X-API-Key: your-key" 'https://ml-production-1e3f.up.railway.app/train/advanced?target=outcome&model_id=my_model'

# Make prediction
curl -X POST https://ml-production-1e3f.up.railway.app/predict -H "Content-Type: application/json" -d '{"features":{"age":45,"gender":0},"model_id":"my_model"}' | python -m json.tool

## Project Structure

ml-toolbox-primary-care/
├── src/ml_toolbox/
│   ├── cli.py
│   ├── serving/
│   │   ├── api.py
│   │   └── api_enhanced.py
│   └── tools/
├── data/
├── models/
├── configs/
├── tests/
├── railway.toml
├── requirements.txt
├── Dockerfile
├── setup.py
├── USER_GUIDE.md
└── README.md

## Railway Deployment

### Prerequisites
- Railway account (https://railway.app)
- GitHub repository
- PostgreSQL service (auto-provisioned)

### Deploy
1. Push to GitHub
2. Connect repo in Railway dashboard
3. Set environment variables:
  - API_KEY: your-secure-key
  - ENABLE_AUTH: true
  - DATABASE_URL: (auto-set)

### Update
git add .
git commit -m "Update"
git push origin main

### Monitor
railway logs --tail

## API Endpoints

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| /health | GET | No | System health |
| /metrics | GET | No | Usage metrics |
| /models | GET | No | List models |
| /models/details/{id} | GET | No | Model info |
| /train/simple | POST | No | Basic training |
| /train/advanced | POST | Yes | Advanced training |
| /predict | POST | No | Single prediction |
| /predict/batch | POST | No | Batch predictions |
| /models/compare | POST | Yes | Compare models |
| /clean/preview | POST | No | Preview data cleaning |

## CLI Usage

export API_URL=https://ml-production-1e3f.up.railway.app
export API_KEY=your-api-key

# Profile data
python -m ml_toolbox.cli profile --input data.csv

# Train locally
python -m ml_toolbox.cli train-simple --input data.csv --target outcome --name model

# List remote models
python -m ml_toolbox.cli list-models --api-url $API_URL

## Authentication

Protected endpoints require:
-H "X-API-Key: your-api-key"

Generate key:
python -c "import secrets; print(secrets.token_urlsafe(32))"

## Sample Datasets

- noshow_training.csv: Appointment no-show prediction
- readmission_training.csv: 30-day readmission risk
- ml_primary_care_synthetic.csv: Combined healthcare metrics

## Performance

- auth_test: 92.4% accuracy, 0.59 AUC
- readmission_model_v2: 75.3% accuracy, 0.66 AUC

## Troubleshooting

### Models disappear
- Check DATABASE_URL configured
- Verify "persisted": true in response
- Check /metrics for models_in_database

### Auth errors
- Verify API_KEY in Railway
- Set ENABLE_AUTH=true
- Include API key header

### Feature mismatch
- Check /models/details/{id}
- Match exact feature names
- Missing features default to 0

## Tips for Railway

- Memory limits: Monitor usage
- Storage: PostgreSQL for persistence
- Scaling: Auto-handled
- Costs: Check dashboard
- Debugging: railway logs

## Next Steps

- Add more training data
- Implement user management
- Create web interface
- Add monitoring dashboard
- Integrate with EHR systems

## Repository

GitHub: https://github.com/porkedchop/ML
Issues: https://github.com/porkedchop/ML/issues

## License

MIT License

---

Built for Virtual Primary Care
