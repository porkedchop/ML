# ML Toolbox for Virtual Primary Care 🏥

A lightweight, production-ready ML toolkit optimized for Railway deployment, designed specifically for virtual primary care operations.

## 🚀 Quick Start

### 1. Organize Project Structure
```bash
# First, organize your existing files
make organize

# Or manually:
mkdir -p src/ml_toolbox/serving src/ml_toolbox/tools
mkdir -p configs data models notebooks tests

# Move your existing files
mv api.py src/ml_toolbox/serving/api.py
mv cli.py src/ml_toolbox/cli.py
mv requiremets.txt requirements.txt  # Fix typo
```

### 2. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 3. Run Locally
```bash
# Start API server
uvicorn src.ml_toolbox.serving.api:app --reload --port 8000

# Or using make
make run-api

# Use CLI
python -m ml_toolbox.cli --help
```

## 📁 Project Structure

```
ml-toolbox-primary-care/
├── src/
│   └── ml_toolbox/
│       ├── __init__.py
│       ├── cli.py              # Command-line interface
│       ├── serving/
│       │   ├── __init__.py
│       │   └── api.py          # FastAPI server
│       └── tools/
│           ├── __init__.py
│           └── (ml tools)      # ML utilities
├── configs/
│   └── default.yaml            # Configuration
├── data/                       # Data files (gitignored)
├── models/                     # Model files (gitignored)
├── notebooks/                  # Jupyter notebooks
├── tests/                      # Test files
├── railway.toml               # Railway config
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup
├── Makefile                   # Dev commands
├── .env.example               # Environment template
├── .gitignore                 # Git exclusions
└── README.md                  # This file
```

## 🚂 Railway Deployment

### Prerequisites
- Railway account ([sign up here](https://railway.app))
- Railway CLI (optional): `npm install -g @railway/cli`
- GitHub repository

### Method 1: Railway Dashboard (Easiest)

1. **Push to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/ml-toolbox-primary-care.git
git push -u origin main
```

2. **Deploy on Railway:**
- Go to [Railway Dashboard](https://railway.app/dashboard)
- Click "New Project" → "Deploy from GitHub repo"
- Select your repository
- Add services: PostgreSQL and Redis (optional)
- Railway will auto-deploy!

### Method 2: Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Link to existing project (if you have one)
railway link

# Add services
railway add

# Deploy
railway up

# Open deployed app
railway open
```

### Environment Variables

Set these in Railway dashboard or CLI:

```bash
# Railway provides automatically
PORT=8000
DATABASE_URL=postgresql://...
REDIS_URL=redis://...

# You need to set
ENVIRONMENT=production
MODEL_STORAGE_TYPE=local
LOG_LEVEL=INFO
```

## 🔧 API Usage

### Health Check
```bash
curl https://your-app.railway.app/health
```

### Upload Model
```bash
curl -X POST -F "file=@model.pkl" \
  https://your-app.railway.app/models/upload?model_name=my_model
```

### Make Prediction
```bash
curl -X POST https://your-app.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 45,
      "prior_no_shows": 2,
      "distance_to_clinic": 5.2
    },
    "model_name": "my_model"
  }'
```

### Train from CSV
```bash
curl -X POST -F "file=@data.csv" \
  https://your-app.railway.app/train/csv?target_column=no_show&model_name=new_model
```

## 🖥️ CLI Usage

### Profile Data
```bash
ml-toolbox profile --input data.csv --output profile.json
```

### Train Model
```bash
ml-toolbox train-simple --input data.csv --target no_show --name my_model
```

### Make Predictions
```bash
ml-toolbox predict --input test.csv --model my_model --output results.csv
```

### Check Status
```bash
ml-toolbox status
```

## 📊 Features

- **No-Show Prediction**: Predict appointment no-shows
- **Patient Triage**: Classify message urgency
- **Data Profiling**: Analyze data quality and statistics
- **Auto-ML**: Automatic model selection and training
- **Model Management**: Upload, store, and version models
- **Batch Processing**: Handle multiple predictions efficiently
- **Caching**: Redis integration for performance
- **Health Monitoring**: Built-in health checks and metrics

## 🛠️ Development

### Install Development Dependencies
```bash
pip install pytest black flake8 mypy pre-commit
```

### Run Tests
```bash
pytest tests/
```

### Format Code
```bash
black src/ tests/
```

### Lint Code
```bash
flake8 src/ tests/
```

## 📈 Monitoring

Railway provides built-in metrics. Additionally, you can:

1. View logs: `railway logs`
2. Check metrics in Railway dashboard
3. Set up custom alerts (Railway integrations)

## 🔐 Security

- Environment variables for sensitive data
- Input validation with Pydantic
- Rate limiting on API endpoints
- CORS configuration
- PHI handling considerations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

MIT License - see LICENSE file

## 💡 Tips for Railway

1. **Memory Management**: Railway has memory limits. Use model caching wisely.
2. **Storage**: Use Redis for temporary data, PostgreSQL for persistent data.
3. **Scaling**: Railway auto-scales, but monitor your usage.
4. **Costs**: Monitor your Railway usage dashboard to control costs.
5. **Logs**: Use `railway logs` to debug issues.

## 🆘 Troubleshooting

### Common Issues

1. **Import errors**: Make sure you installed the package: `pip install -e .`
2. **Port issues**: Railway provides PORT automatically, don't hardcode it
3. **Memory issues**: Reduce MODEL_CACHE_SIZE or use smaller models
4. **Storage issues**: Use external storage (S3) for large models

### Get Help

- Railway Discord: https://discord.gg/railway
- GitHub Issues: Create an issue in your repo
- Railway Docs: https://docs.railway.app

## 🎯 Next Steps

1. ✅ Organize project structure
2. ✅ Install dependencies  
3. ✅ Test locally
4. ⬜ Push to GitHub
5. ⬜ Deploy to Railway
6. ⬜ Add PostgreSQL/Redis
7. ⬜ Upload your models
8. ⬜ Start making predictions!

---

Built with ❤️ for Virtual Primary Care