# GGUF Quantization Setup - Disk Space Solutions

## Current Situation
- **System**: H100 GPU with 81GB VRAM (38GB currently used)
- **Disk Space**: 97% full (only 3GB available)
- **Status**: Cannot install PyTorch + dependencies (~10-15GB needed)

## Immediate Solutions

### Option 1: Free Up Disk Space
```bash
# Check largest files/directories
du -sh /home/* | sort -rh | head -10
du -sh /tmp/* | sort -rh | head -10

# Clean common space consumers
sudo apt clean
sudo apt autoremove
rm -rf ~/.cache/pip
rm -rf ~/.cache/huggingface
docker system prune -a  # if Docker is used
```

### Option 2: Use External Storage
```bash
# Mount external drive or network storage
# Install packages to external location
pip install --target /external/path torch diffusers gguf
export PYTHONPATH="/external/path:$PYTHONPATH"
```

### Option 3: Minimal Installation (Recommended)
```bash
# Remove virtual environment (saves ~500MB)
rm -rf venv_gguf

# Install system-wide with minimal cache
pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cu121
pip install --no-cache-dir --user diffusers transformers gguf huggingface_hub
```

### Option 4: Use Different Environment
- **Google Colab**: Free GPU access with pre-installed packages
- **Kaggle Notebooks**: 30GB disk space, GPU access
- **Local machine**: If available with more storage

## Quick Test (No Installation)
```python
# Test if packages are already available system-wide
python3 -c "import torch, diffusers; print('Ready for GGUF testing')"
```

## Next Steps
1. Choose one of the above solutions
2. Run `python3 install_minimal.py` (handles disk space constraints)
3. Execute `python3 gguf_test.py` to test GGUF quantization

## Expected Results
- **Q8_0**: ~40GB model, best quality
- **Q4_K**: ~20GB model, balanced (recommended)
- **Q2_K**: ~10GB model, maximum compression
