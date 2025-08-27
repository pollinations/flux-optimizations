#!/bin/bash

# xDiT Multi-GPU Environment Setup Script
set -e

echo "🚀 Setting up xDiT Multi-GPU Environment..."

# Check if we're in the right directory
if [[ ! -f "README.md" ]]; then
    echo "❌ Please run this script from the 04_xdit_multi_gpu directory"
    exit 1
fi

# Check GPU availability
echo "🔍 Checking GPU setup..."
nvidia-smi
echo ""
echo "📊 GPU Topology:"
nvidia-smi topo -m
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "🐍 Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv_xdit
source venv_xdit/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install xDiT
echo "⚡ Installing xDiT..."
pip install xfuser

# Install additional dependencies
echo "📚 Installing additional dependencies..."
pip install diffusers transformers accelerate huggingface_hub
pip install Pillow safetensors

# Verify installation
echo "✅ Verifying installation..."
python3 -c "
import torch
import xfuser
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ GPU count: {torch.cuda.device_count()}')
print(f'✅ xFuser installed successfully')
"

# Set Hugging Face token
echo "🔐 Setting up Hugging Face authentication..."
export HF_TOKEN="your_huggingface_token_here"

echo ""
echo "🎉 Environment setup complete!"
echo "📝 To activate the environment, run: source venv_xdit/bin/activate"
echo "🚀 Ready to run xDiT multi-GPU inference!"
