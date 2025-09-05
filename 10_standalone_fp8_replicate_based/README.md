# Approach 10: Standalone FP8 Implementation (Replicate-based)

This approach provides a standalone Flux Schnell FP8 implementation based on Replicate's cog-flux code that works on cluster environments with multiple GPUs while forcing single GPU usage.

## Overview

- **Method**: Standalone Python script with FP8 quantization
- **Performance**: ~3 seconds per 1024x1024 image generation
- **Memory**: Optimized with FP8 quantization
- **Compatibility**: Works on multi-GPU clusters by forcing single GPU usage

## Key Features

- ✅ **Single GPU enforcement** on multi-GPU clusters
- ✅ **FP8 quantization** for memory efficiency
- ✅ **No Cog dependency** - pure PyTorch implementation
- ✅ **Clean device management** using PyTorch context managers
- ✅ **Automatic model downloading** and caching
- ✅ **Multiple aspect ratios** support

## Files

- `flux_schnell_fp8_standalone.py` - Main standalone implementation
- `requirements.txt` - Python dependencies with tested versions
- `README.md` - This documentation

## Installation

### Automated Setup (Recommended)

```bash
# One-line setup for new instances
curl -sSL https://raw.githubusercontent.com/pollinations/flux-optimizations/main/10_standalone_fp8_replicate_based/setup_instance.sh | bash
```

Or if you already have the repository:

```bash
# Run the setup script
./setup_instance.sh
```

### Manual Installation

```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3-pip python3.12-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install pybase64 loguru

# The script will automatically download models on first run
```

## Usage

### Basic Usage
```bash
python flux_schnell_fp8_standalone.py --prompt "a cat in a hat"
```

### Advanced Options
```bash
python flux_schnell_fp8_standalone.py \
    --prompt "a beautiful landscape" \
    --width 1024 \
    --height 1024 \
    --steps 4 \
    --num-images 2 \
    --seed 42 \
    --output-dir ./outputs \
    --no-compile
```

### Available Arguments
- `--prompt` - Text prompt for generation
- `--width` - Image width (default: 1024)
- `--height` - Image height (default: 1024)
- `--aspect-ratio` - Preset aspect ratios (1:1, 16:9, 9:16, etc.)
- `--steps` - Number of inference steps (default: 4)
- `--num-images` - Number of images to generate (default: 1)
- `--seed` - Random seed for reproducibility
- `--output-dir` - Output directory (default: ./outputs)
- `--no-compile` - Disable model compilation for compatibility

## Technical Details

### Single GPU Enforcement
The implementation uses multiple PyTorch techniques to force single GPU usage:

1. **Environment restriction**: `CUDA_VISIBLE_DEVICES="0"`
2. **Device context**: `with torch.cuda.device(0):`
3. **Explicit device assignment**: All models forced to `cuda:0`

```python
# Force single GPU usage before any CUDA initialization
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)

# Create pipeline with explicit single GPU device assignment
with torch.cuda.device(0):
    self.pipeline = FluxPipeline(
        flux_device="cuda:0",
        ae_device="cuda:0",
        clip_device="cuda:0", 
        t5_device="cuda:0",
    )
```

### FP8 Quantization
- Uses FP8 quantization for the main Flux model
- Disables T5 quantization to avoid tensor inference issues
- Maintains bfloat16 precision for other components

### Model Management
- Automatic model downloading and caching
- Models stored in `model-cache/` directory
- Supports both local and remote model paths

## Performance

- **Generation time**: ~3 seconds per 1024x1024 image
- **Memory usage**: Optimized with FP8 quantization
- **GPU utilization**: Single GPU (H100 tested)
- **Batch processing**: Supports multiple images per run

## Troubleshooting

### Multi-GPU Issues
If you encounter "Expected all tensors to be on the same device" errors:
- Ensure `CUDA_VISIBLE_DEVICES=0` is set
- Use `--no-compile` flag
- Check that only one GPU is visible to the process

### Memory Issues
- Use FP8 quantization (enabled by default)
- Reduce batch size with `--num-images 1`
- Enable model offloading if needed

### Model Download Issues
- Check internet connectivity
- Verify Hugging Face Hub access
- Models are cached in `model-cache/` directory

## Dependencies

Key dependencies and their tested versions:
- PyTorch 2.7.0 with CUDA 12.1 support
- Transformers 4.43.3
- Diffusers 0.32.2
- TorchAO 0.12.0 for quantization
- Safetensors 0.4.3

See `requirements.txt` for complete dependency list.
