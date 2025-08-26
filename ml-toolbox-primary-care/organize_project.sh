#!/bin/bash
# organize_project.sh
# Script to organize the ML Toolbox project structure for Railway

echo "🚀 Organizing ML Toolbox Project Structure..."

# Create the proper directory structure
mkdir -p ml-toolbox-primary-care/src/ml_toolbox/serving
mkdir -p ml-toolbox-primary-care/src/ml_toolbox/tools
mkdir -p ml-toolbox-primary-care/configs
mkdir -p ml-toolbox-primary-care/data
mkdir -p ml-toolbox-primary-care/models
mkdir -p ml-toolbox-primary-care/notebooks
mkdir -p ml-toolbox-primary-care/tests

# Move existing files to proper locations (if they exist)
if [ -f "ml-toolbox-primary-care/api.py" ]; then
    mv ml-toolbox-primary-care/api.py ml-toolbox-primary-care/src/ml_toolbox/serving/api.py
    echo "✓ Moved api.py to src/ml_toolbox/serving/"
fi

if [ -f "ml-toolbox-primary-care/cli.py" ]; then
    mv ml-toolbox-primary-care/cli.py ml-toolbox-primary-care/src/ml_toolbox/cli.py
    echo "✓ Moved cli.py to src/ml_toolbox/"
fi

# Fix the typo in requirements filename if it exists
if [ -f "ml-toolbox-primary-care/requiremets.txt" ]; then
    mv ml-toolbox-primary-care/requiremets.txt ml-toolbox-primary-care/requirements.txt
    echo "✓ Renamed requiremets.txt to requirements.txt"
fi

# Create __init__.py files for Python packages
touch ml-toolbox-primary-care/src/ml_toolbox/__init__.py
touch ml-toolbox-primary-care/src/ml_toolbox/serving/__init__.py
touch ml-toolbox-primary-care/src/ml_toolbox/tools/__init__.py

# Create .gitkeep files for empty directories
touch ml-toolbox-primary-care/data/.gitkeep
touch ml-toolbox-primary-care/models/.gitkeep
touch ml-toolbox-primary-care/notebooks/.gitkeep
touch ml-toolbox-primary-care/tests/.gitkeep

echo "✅ Project structure organized successfully!"
echo ""
echo "📁 New structure:"
tree ml-toolbox-primary-care -I '__pycache__' --dirsfirst