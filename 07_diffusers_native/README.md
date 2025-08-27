# Test Plan: Diffusers Native Optimizations

## Objective
Test built-in diffusers optimizations including memory-efficient attention, model offloading, and VAE optimizations.

## Expected Performance
- Target: 20-40% speedup with significant memory reduction
- Expected time: ~7-8 seconds for 1024px image (4 steps)
- Primary benefit: Memory efficiency and stability

## Prerequisites Check
1. **System Requirements**:
   - NVIDIA H100 GPU
   - Python 3.10+
   - PyTorch 2.0+ (for SDPA)
   - At least 8GB GPU memory

2. **Check PyTorch Version**:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'SDPA available: {hasattr(torch.nn.functional, \"scaled_dot_product_attention\")}')"
   ```

3. **Hugging Face Authentication**:
   ```bash
   # Set your Hugging Face token
   export HF_TOKEN="your_huggingface_token_here"
   # Or use in Python:
   # from huggingface_hub import login
   # login(token="your_huggingface_token_here")
   ```

## Virtual Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv_diffusers_native
source venv_diffusers_native/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Installation Steps
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install latest diffusers
pip install diffusers transformers accelerate huggingface_hub

# Install xFormers for memory-efficient attention
pip install xformers

# Install additional dependencies
pip install Pillow safetensors
```

## Implementation Plan
1. **Test Scaled Dot-Product Attention (SDPA)**:
   - Default in PyTorch 2.0+
   - Automatic backend selection
2. **Test xFormers memory-efficient attention**
3. **Test model offloading strategies**:
   - CPU offloading
   - Sequential offloading
4. **Test VAE optimizations**:
   - VAE slicing
   - VAE tiling
5. **Test attention slicing**
6. **Test channels_last memory format**

## Key Optimizations to Test

### Memory-Efficient Attention
```python
# SDPA (automatic)
# No code changes required with PyTorch 2.0+

# xFormers (explicit)
pipeline.enable_xformers_memory_efficient_attention()
```

### Model Offloading
```python
# CPU offloading
pipeline.enable_model_cpu_offload()

# Sequential offloading (maximum memory savings)
pipeline.enable_sequential_cpu_offload()
```

### VAE Optimizations
```python
# VAE slicing for high-resolution
pipeline.enable_vae_slicing()

# VAE tiling for ultra-high resolution
pipeline.enable_vae_tiling()
```

### Attention Slicing
```python
# Reduce attention memory usage
pipeline.enable_attention_slicing()
```

### Memory Format
```python
# Channels last for better performance
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.vae.to(memory_format=torch.channels_last)
```

## Success Criteria
- [ ] SDPA working automatically
- [ ] xFormers installation successful
- [ ] Model offloading reducing memory usage
- [ ] VAE optimizations enabling higher resolutions
- [ ] Attention slicing working
- [ ] Memory format optimizations applied
- [ ] Performance improvement measured
- [ ] Memory usage significantly reduced

## Benchmarking Protocol
1. **Baseline measurement** (no optimizations)
2. **Individual optimization testing**:
   - SDPA vs xFormers performance
   - Model offloading memory savings
   - VAE optimization benefits
3. **Combined optimizations**:
   - Best combination for performance
   - Best combination for memory efficiency
4. **High-resolution testing**:
   - 2048px generation with VAE tiling
   - Memory usage at different resolutions

## Special Considerations
- SDPA automatically selects best backend
- xFormers may provide better performance on some hardware
- Model offloading trades speed for memory
- VAE optimizations crucial for high-resolution
- Some optimizations have quality implications

## Progress Tracking
**IMPORTANT**: Create a `progress.md` file in this folder to track your progress.

### Progress Template
```markdown
# Progress: Diffusers Native Optimizations

## Overall Progress: [X]%

### Completed Tasks:
- [ ] Environment setup
- [ ] xFormers installed
- [ ] SDPA functionality verified
- [ ] Model offloading tested
- [ ] VAE optimizations implemented
- [ ] Benchmarking completed
- [ ] Results documented

### Current Status:
[Brief description of current work]

### Issues/Blockers:
[Any problems encountered]

### Next Steps:
[What's planned next]
```

## Output Requirements
- Optimization effectiveness comparison
- Memory usage reduction measurements
- Performance impact analysis
- High-resolution generation capability
- Stability and reliability assessment
- **progress.md file with percentage completion**

## Cleanup After Testing
**IMPORTANT**: Clean up caches after testing to prevent disk space issues:
```bash
# Remove Hugging Face cache
rm -rf ~/.cache/huggingface

# Remove PyTorch compilation cache
rm -rf ~/.cache/torch

# Remove xFormers cache
rm -rf ~/.cache/xformers

# Remove pip cache
pip cache purge

# Remove virtual environment if no longer needed
# rm -rf venv_diffusers_native
```
