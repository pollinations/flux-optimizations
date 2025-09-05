#!/bin/bash
# Complete setup script for Flux Schnell FP8 on remote instances
# Usage: 
#   On new instance: curl -sSL https://raw.githubusercontent.com/pollinations/flux-optimizations/main/10_standalone_fp8_replicate_based/setup_instance.sh | bash
#   Or locally: ./setup_instance.sh

set -e  # Exit on any error

echo "🚀 Starting Flux Schnell FP8 setup..."

# Clone repository if we're not already in it
if [ ! -f "flux_schnell_fp8_standalone.py" ]; then
    echo "📦 Cloning repository..."
    git clone https://github.com/pollinations/flux-optimizations.git
    cd flux-optimizations/10_standalone_fp8_replicate_based
fi

# Update system packages
echo "📦 Updating system packages..."
sudo apt update
sudo apt install -y python3-pip python3.12-venv git

# Create virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Install additional missing dependencies
echo "🔧 Installing additional dependencies..."
pip install pybase64 loguru

# Download models if they don't exist
echo "📥 Checking and downloading models..."
if [ ! -d "model-cache" ] || [ ! -f "model-cache/schnell-fp8/schnell-fp8.safetensors" ]; then
    echo "Models not found, downloading..."
    CUDA_VISIBLE_DEVICES=0 python flux_schnell_fp8_standalone.py --prompt "test" --download-models --no-compile || true
else
    echo "Models already downloaded, skipping..."
fi

# Test with a safe resolution
echo "🧪 Testing with 768x768 resolution..."
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python flux_schnell_fp8_standalone.py --prompt "a simple test" --steps 2 --no-compile --width 768 --height 768 || {
    echo "⚠️ Test failed, but setup is complete."
}

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Test image generation: CUDA_VISIBLE_DEVICES=0 python flux_schnell_fp8_standalone.py --prompt 'your prompt' --no-compile --width 768 --height 768"
echo "3. Start the server: ./start_server.sh [PORT] [GPU_ID]"
echo "4. Start multiple servers: ./start_servers.sh [BASE_PORT]"
echo ""
echo "💡 Memory recommendations for RTX 4090:"
echo "- Use 768x768 or smaller for reliable generation"
echo "- Always set CUDA_VISIBLE_DEVICES=0 for single GPU"
echo "- 1024x1024 may work but can hit memory limits"
