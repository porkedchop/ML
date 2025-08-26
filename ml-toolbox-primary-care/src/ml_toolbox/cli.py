# cli_railway.py
"""
Lightweight CLI for Railway deployment
Can be run as a web service or one-off job
"""

import click
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
import requests
from typing import Optional

# For Railway, we can use environment variables for configuration
API_URL = os.getenv('API_URL', 'http://localhost:8000')
STORAGE_TYPE = os.getenv('MODEL_STORAGE_TYPE', 'local')

@click.group()
@click.option('--api-url', default=None, help='API URL for remote operations')
@click.pass_context
def cli(ctx, api_url):
    """ML Toolbox CLI for Railway"""
    ctx.ensure_object(dict)
    ctx.obj['api_url'] = api_url or API_URL
    
    # For Railway, always use lightweight operations
    click.echo(f"ML Toolbox CLI (Railway Mode)")
    click.echo(f"API URL: {ctx.obj['api_url']}")

@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path())
@click.option('--format', type=click.Choice(['json', 'csv']), default='json')
def profile(input, output, format):
    """Quick data profiling (Railway-optimized)"""
    click.echo(f"Profiling: {input}")
    
    # Load data
    df = pd.read_csv(input)
    
    # Basic profiling (lightweight for Railway)
    profile = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    # Numeric stats (limited to avoid memory issues)
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
    if len(numeric_cols) > 0:
        profile['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    # Output
    if output:
        if format == 'json':
            with open(output, 'w') as f:
                json.dump(profile, f, indent=2, default=str)
        else:
            pd.DataFrame(profile).to_csv(output)
        click.echo(f"Profile saved to: {output}")
    else:
        click.echo(json.dumps(profile, indent=2, default=str))

@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True))
@click.option('--target', '-t', required=True)
@click.option('--name', '-n', default='model')
@click.option('--upload', is_flag=True, help='Upload to API after training')
@click.pass_context
def train_simple(ctx, input, target, name, upload):
    """Train a simple model (Railway-friendly)"""
    click.echo(f"Training simple model from: {input}")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib
    
    # Load data
    df = pd.read_csv(input)
    
    if target not in df.columns:
        click.echo(f"Error: Target '{target}' not found", err=True)
        return
    
    # Prepare data
    X = df.drop(columns=[target])
    y = df[target]
    
    # Handle categorical columns simply
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.Categorical(X[col]).codes
    
    X = X.fillna(0)
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    accuracy = accuracy_score(y_test, model.predict(X_test))
    click.echo(f"Accuracy: {accuracy:.3f}")
    
    # Save model
    model_data = {
        'model': model,
        'features': list(X.columns),
        'target': target,
        'accuracy': accuracy,
        'trained_at': datetime.now().isoformat()
    }
    
    model_path = f"/tmp/{name}.pkl"
    joblib.dump(model_data, model_path)
    click.echo(f"Model saved to: {model_path}")
    
    # Upload to API if requested
    if upload and ctx.obj.get('api_url'):
        with open(model_path, 'rb') as f:
            files = {'file': (f'{name}.pkl', f, 'application/octet-stream')}
            response = requests.post(
                f"{ctx.obj['api_url']}/models/upload",
                files=files,
                params={'model_name': name}
            )
            
            if response.status_code == 200:
                click.echo(f"✓ Model uploaded to API: {name}")
            else:
                click.echo(f"✗ Upload failed: {response.text}", err=True)

@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True))
@click.option('--model', '-m', default='default_model')
@click.option('--output', '-o', type=click.Path())
@click.pass_context
def predict(ctx, input, model, output):
    """Make predictions using API"""
    api_url = ctx.obj.get('api_url')
    
    if not api_url:
        click.echo("Error: API URL not configured", err=True)
        return
    
    # Load data
    df = pd.read_csv(input)
    click.echo(f"Making predictions for {len(df)} samples...")
    
    # Prepare batch request
    data = df.to_dict('records')
    
    # Send to API
    response = requests.post(
        f"{api_url}/predict/batch",
        json={'data': data, 'model_name': model}
    )
    
    if response.status_code == 200:
        results = response.json()
        predictions = results['predictions']
        
        # Add predictions to dataframe
        df['prediction'] = [p.get('prediction', None) for p in predictions]
        df['probability'] = [p.get('probability', None) for p in predictions]
        
        # Save results
        if output:
            df.to_csv(output, index=False)
            click.echo(f"✓ Predictions saved to: {output}")
        else:
            click.echo(df[['prediction', 'probability']].head(10))
    else:
        click.echo(f"✗ Prediction failed: {response.text}", err=True)

@cli.command()
@click.pass_context
def status(ctx):
    """Check API and system status"""
    api_url = ctx.obj.get('api_url')
    
    if api_url:
        try:
            # Check API health
            response = requests.get(f"{api_url}/health")
            if response.status_code == 200:
                health = response.json()
                click.echo("API Status: ✓ Healthy")
                click.echo(f"  Environment: {health.get('environment')}")
                click.echo(f"  Models loaded: {health.get('models_loaded')}")
                click.echo(f"  Redis: {health.get('redis')}")
                click.echo(f"  Storage: {health.get('storage_type')}")
            else:
                click.echo("API Status: ✗ Unhealthy")
        except Exception as e:
            click.echo(f"API Status: ✗ Unreachable ({e})")
    
    # Check local environment
    click.echo("\nEnvironment:")
    click.echo(f"  Storage Type: {STORAGE_TYPE}")
    click.echo(f"  Temp Dir: {Path('/tmp').exists()}")
    click.echo(f"  Memory Available: {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024 / 1024:.0f} MB")

@cli.command()
@click.option('--file', '-f', required=True, type=click.Path(exists=True))
@click.option('--name', '-n', required=True)
@click.pass_context
def upload(ctx, file, name):
    """Upload a model to API"""
    api_url = ctx.obj.get('api_url')
    
    if not api_url:
        click.echo("Error: API URL not configured", err=True)
        return
    
    with open(file, 'rb') as f:
        files = {'file': (os.path.basename(file), f, 'application/octet-stream')}
        response = requests.post(
            f"{api_url}/models/upload",
            files=files,
            params={'model_name': name}
        )
    
    if response.status_code == 200:
        click.echo(f"✓ Model uploaded: {name}")
    else:
        click.echo(f"✗ Upload failed: {response.text}", err=True)

@cli.command()
@click.pass_context
def list_models(ctx):
    """List available models"""
    api_url = ctx.obj.get('api_url')
    
    if not api_url:
        click.echo("Error: API URL not configured", err=True)
        return
    
    response = requests.get(f"{api_url}/models")
    
    if response.status_code == 200:
        data = response.json()
        click.echo("Loaded Models:")
        for model in data.get('loaded_models', []):
            click.echo(f"  ✓ {model}")
        
        click.echo("\nAvailable Models:")
        for model in data.get('available_models', []):
            click.echo(f"  - {model}")
    else:
        click.echo(f"Error: {response.text}", err=True)

if __name__ == '__main__':
    cli()