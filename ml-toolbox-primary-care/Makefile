.PHONY: help install run-api

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make run-api    - Run the API server"
	@echo "  make deploy     - Deploy to Railway"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

run-api:
	@echo "Starting API server..."
	@. venv/bin/activate && python -m uvicorn src.ml_toolbox.serving.api:app --reload --host 0.0.0.0 --port 8000

deploy:
	railway up

test:
	curl http://localhost:8000/health
