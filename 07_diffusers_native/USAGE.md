# Flux Schnell with Vanilla Diffusers - Usage Guide

## Quick Start

### Basic Usage
```bash
# Simple generation with default optimizations
python3 flux_schnell_diffusers.py --prompt "A beautiful landscape"

# With memory optimizations for limited VRAM
python3 flux_schnell_diffusers.py --prompt "A cyberpunk city" --vae-slicing --attention-slicing

# Maximum memory efficiency (slower but uses less VRAM)
python3 flux_schnell_diffusers.py --prompt "A fantasy castle" --sequential-offload --vae-slicing --vae-tiling
```

### Run the Example
```bash
python3 run_example.py
```

## Available Optimizations

### Memory-Efficient Attention
- **SDPA (default)**: Automatic in PyTorch 2.0+, no flags needed
- **xFormers**: `--xformers` - May provide better performance on some hardware

### Model Offloading
- **CPU Offload**: `--cpu-offload` - Moves models to CPU when not in use
- **Sequential Offload**: `--sequential-offload` - Maximum memory savings, slower

### VAE Optimizations
- **VAE Slicing**: `--vae-slicing` - Reduces VAE memory usage
- **VAE Tiling**: `--vae-tiling` - Enables ultra-high resolution generation

### Additional Optimizations
- **Attention Slicing**: `--attention-slicing` - Reduces attention memory usage
- **Channels Last**: `--channels-last` - Better memory layout for performance
- **Torch Compile**: `--compile` - JIT compilation for speed (experimental)

## Memory Usage Recommendations

### High VRAM (24GB+)
```bash
python3 flux_schnell_diffusers.py --prompt "Your prompt" --channels-last
```

### Medium VRAM (12-24GB)
```bash
python3 flux_schnell_diffusers.py --prompt "Your prompt" --vae-slicing --attention-slicing
```

### Low VRAM (8-12GB)
```bash
python3 flux_schnell_diffusers.py --prompt "Your prompt" --sequential-offload --vae-slicing --vae-tiling --attention-slicing
```

## Benchmarking

Run performance benchmarks:
```bash
python3 flux_schnell_diffusers.py --benchmark --vae-slicing --attention-slicing
```

## Python API Usage

```python
from flux_schnell_diffusers import FluxSchnellOptimized

# Create instance
flux = FluxSchnellOptimized()

# Load with optimizations
flux.load_pipeline(
    vae_slicing=True,
    attention_slicing=True
)

# Generate image
image = flux.generate_image(
    prompt="A beautiful sunset",
    width=1024,
    height=1024,
    num_inference_steps=4,
    seed=42,
    output_path="output.png"
)
```

## Performance Tips

1. **Start with basic optimizations**: `--vae-slicing --attention-slicing`
2. **For memory issues**: Add `--sequential-offload`
3. **For high resolution**: Add `--vae-tiling`
4. **SDPA is enabled by default** - no need for additional flags
5. **Monitor memory usage** - the script reports GPU memory consumption

## Troubleshooting

- **Out of memory**: Use `--sequential-offload --vae-slicing --vae-tiling`
- **Slow generation**: Remove offloading flags if you have enough VRAM
- **Import errors**: Ensure all dependencies are installed from requirements.txt
