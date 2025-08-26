FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway dynamically assigns PORT - we must respect it
CMD python -m uvicorn src.ml_toolbox.serving.api:app --host 0.0.0.0 --port ${PORT}

# Copy both API versions
COPY src/ml_toolbox/serving/api_v2.py src/ml_toolbox/serving/api_v2.py
