#!/bin/bash
# FLUX Schnell torch.compile Baseline Setup Script

echo "🚀 Setting up FLUX Schnell torch.compile baseline environment..."

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Run this script from the project directory."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv_torch_compile
source venv_torch_compile/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (CUDA 12.1 compatible)
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo "📚 Installing other dependencies..."
pip install -r requirements.txt

# Verify installation
echo "✅ Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"No GPU\"}')"

echo "🎉 Setup complete! Activate the environment with:"
echo "source venv_torch_compile/bin/activate"
echo ""
echo "Then run the baseline test with:"
echo "python flux_baseline.py"
