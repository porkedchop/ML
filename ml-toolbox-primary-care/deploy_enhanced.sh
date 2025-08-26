#!/bin/bash
# Deploy enhanced API

# Copy enhanced API to replace current API
cp src/ml_toolbox/serving/api_enhanced.py src/ml_toolbox/serving/api.py

# Update requirements.txt with new dependencies
cat >> requirements.txt << 'EOReq'
psycopg2-binary==2.9.9
EOReq

# Commit and deploy
git add -A
git commit -m "Add PostgreSQL persistence, authentication, and monitoring"
git push origin main

echo "Enhanced API deployed. Set these environment variables in Railway:"
echo "  API_KEY=your-secure-api-key"
echo "  ENABLE_AUTH=true"
echo "  DATABASE_URL is already set by Railway"
