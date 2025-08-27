#!/bin/bash
# Efficient installation script for TorchAO quantization

echo "=== TorchAO Dependencies Installation ==="

# Check if we can use system Python or need virtual environment
echo "Checking Python installation..."
python3 --version

# Option 1: Try system-wide installation (if permissions allow)
echo "Attempting system-wide installation..."
pip3 install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if [ $? -eq 0 ]; then
    echo "PyTorch installed successfully"
    pip3 install --user torchao diffusers transformers accelerate huggingface_hub Pillow safetensors
else
    echo "System installation failed, trying virtual environment in current directory..."
    
    # Option 2: Create minimal virtual environment in current directory
    python3 -m venv venv_minimal
    source venv_minimal/bin/activate
    
    # Install only essential packages
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install --no-cache-dir torchao diffusers transformers accelerate huggingface_hub Pillow safetensors
fi

echo "Installation complete!"
echo "To test: python3 flux_fp8_weight_only.py"
