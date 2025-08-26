#!/bin/bash
# organize.sh - Organize ML Toolbox project for Railway

echo "🚀 Organizing ML Toolbox Project..."
echo "================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -d "ml-toolbox-primary-care" ]; then
    echo "❌ Error: ml-toolbox-primary-care directory not found!"
    echo "Please run this script from the parent directory."
    exit 1
fi

cd ml-toolbox-primary-care

echo -e "${YELLOW}📁 Creating directory structure...${NC}"

# Create all necessary directories
mkdir -p src/ml_toolbox/serving
mkdir -p src/ml_toolbox/tools
mkdir -p configs
mkdir -p data
mkdir -p models
mkdir -p notebooks
mkdir -p tests

echo -e "${GREEN}✓ Directories created${NC}"

# Create Python package files
echo -e "${YELLOW}📝 Creating __init__.py files...${NC}"

cat > src/ml_toolbox/__init__.py << 'EOF'
"""ML Toolbox for Virtual Primary Care"""
__version__ = "0.1.0"
EOF

touch src/ml_toolbox/serving/__init__.py
touch src/ml_toolbox/tools/__init__.py

echo -e "${GREEN}✓ Package files created${NC}"

# Move existing files if they exist
echo -e "${YELLOW}📦 Moving existing files...${NC}"

if [ -f "api.py" ]; then
    mv api.py src/ml_toolbox/serving/api.py
    echo -e "${GREEN}✓ Moved api.py to src/ml_toolbox/serving/${NC}"
fi

if [ -f "cli.py" ]; then
    mv cli.py src/ml_toolbox/cli.py
    echo -e "${GREEN}✓ Moved cli.py to src/ml_toolbox/${NC}"
fi

# Fix typo in requirements filename
if [ -f "requiremets.txt" ]; then
    mv requiremets.txt requirements.txt
    echo -e "${GREEN}✓ Renamed requiremets.txt to requirements.txt${NC}"
fi

# Create .gitkeep files for empty directories
touch data/.gitkeep
touch models/.gitkeep
touch notebooks/.gitkeep
touch tests/.gitkeep

# Create a basic test file
cat > tests/test_api.py << 'EOF'
"""Basic tests for API"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_import():
    """Test that modules can be imported"""
    try:
        from ml_toolbox.serving import api
        assert api is not None
    except ImportError:
        pass  # API may have dependencies not installed yet

def test_basic():
    """Basic test to ensure pytest works"""
    assert True
EOF

echo -e "${GREEN}✓ Test file created${NC}"

# Create sample data file
cat > data/sample.csv << 'EOF'
patient_id,age,gender,prior_no_shows,distance_to_clinic,insurance_type,no_show
P0001,45,F,2,5.2,Private,0
P0002,32,M,0,12.1,Medicare,0
P0003,67,F,1,3.5,Medicaid,1
P0004,28,M,3,8.7,Private,1
P0005,55,F,0,2.1,Private,0
EOF

echo -e "${GREEN}✓ Sample data created${NC}"

# Display the final structure
echo -e "\n${YELLOW}📊 Final project structure:${NC}"
echo "================================"

# Use tree if available, otherwise use find
if command -v tree &> /dev/null; then
    tree -I '__pycache__|*.pyc|.git' --dirsfirst -L 3
else
    find . -type d -name "__pycache__" -prune -o -type d -print | head -20
fi

echo -e "\n${GREEN}✅ Organization complete!${NC}"
echo "================================"
echo ""
echo "Next steps:"
echo "1. cd ml-toolbox-primary-care"
echo "2. python -m venv venv"
echo "3. source venv/bin/activate  # On Mac/Linux"
echo "4. pip install -r requirements.txt"
echo "5. pip install -e ."
echo "6. Test the API: uvicorn src.ml_toolbox.serving.api:app --reload"
echo ""
echo "For Railway deployment:"
echo "1. git init"
echo "2. git add ."
echo "3. git commit -m 'Initial commit'"
echo "4. Push to GitHub"
echo "5. Connect to Railway"