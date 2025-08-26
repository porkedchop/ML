FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# The PORT environment variable is set by Railway
# Use it or default to 8000
CMD python -m uvicorn src.ml_toolbox.serving.api:app --host 0.0.0.0 --port ${PORT:-8000}
