#!/bin/bash
# Setup script for FBCache + TeaCache optimization testing

set -e

echo "Setting up FBCache + TeaCache optimization environment..."

# Create and activate virtual environment
echo "Creating virtual environment..."
python -m venv venv_fbcache_teacache
source venv_fbcache_teacache/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install PyTorch with CUDA support (if not already installed)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clone TeaCache repository for integration
echo "Cloning TeaCache repository..."
if [ ! -d "TeaCache" ]; then
    git clone https://github.com/ali-vilab/TeaCache.git
    echo "TeaCache repository cloned successfully"
else
    echo "TeaCache repository already exists"
fi

# Install TeaCache dependencies
if [ -d "TeaCache/TeaCache4FLUX" ]; then
    echo "Installing TeaCache4FLUX dependencies..."
    cd TeaCache/TeaCache4FLUX
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    cd ../..
fi

# Create output directories
echo "Creating output directories..."
mkdir -p outputs/fbcache
mkdir -p outputs/teacache
mkdir -p outputs/combined

# Set Hugging Face token if provided
if [ ! -z "$HF_TOKEN" ]; then
    echo "Hugging Face token detected in environment"
else
    echo "Warning: HF_TOKEN not set. Please set it before running tests:"
    echo "export HF_TOKEN=\"your_token_here\""
fi

echo "Setup completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "source venv_fbcache_teacache/bin/activate"
echo ""
echo "To run tests:"
echo "python fbcache_test.py          # Test FBCache alone"
echo "python teacache_test.py         # Test TeaCache alone"
echo "python combined_test.py         # Test both together"
