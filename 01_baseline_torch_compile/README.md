# Test Plan: Baseline torch.compile Optimization

## Objective
Establish baseline performance using standard torch.compile optimizations on FLUX Schnell with H100-specific configurations.

## Expected Performance
- Target: 1.5x speedup over unoptimized baseline
- Expected time: ~6.7 seconds for 1024px image (4 steps)

## Prerequisites Check
1. **System Requirements**:
   - NVIDIA H100 GPU with CUDA 12.2+
   - Python 3.10+
   - At least 16GB GPU memory

2. **Check GPU**:
   ```bash
   nvidia-smi
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name()}')"
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
python -m venv venv_torch_compile
source venv_torch_compile/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Installation Steps
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install diffusers and dependencies
pip install diffusers transformers accelerate huggingface_hub

# Install additional dependencies
pip install Pillow safetensors
```

## Implementation Plan ✅ COMPLETED
1. **✅ Create baseline script** with standard FLUX pipeline
2. **✅ Add torch.compile optimizations**:
   - Configure H100-specific inductor flags
   - Compile transformer with fullgraph=True
   - Use max-autotune mode
3. **✅ Add memory format optimizations** (channels_last)
4. **✅ Benchmark and measure**:
   - Cold start compilation time
   - Warm inference time
   - Memory usage
   - Image quality validation

## Quick Start
```bash
# 1. Run setup script
./setup.sh

# 2. Activate environment
source venv_torch_compile/bin/activate

# 3. Run baseline test
python flux_baseline.py
```

## Files Created
- `flux_baseline.py` - Main implementation with torch.compile optimizations
- `requirements.txt` - Python dependencies
- `setup.sh` - Automated environment setup script

## Key Optimizations to Test
- `torch._inductor.config.conv_1x1_as_mm = True`
- `torch._inductor.config.coordinate_descent_tuning = True`
- `torch._inductor.config.epilogue_fusion = False`
- `torch._inductor.config.coordinate_descent_check_all_directions = True`
- Regional compilation with `compile_repeated_blocks()`

## Success Criteria
- [ ] Successful environment setup
- [ ] Baseline inference working
- [ ] torch.compile optimization applied
- [ ] 1.5x+ speedup achieved
- [ ] Compilation time < 10 seconds (warm cache)
- [ ] Generated images match quality expectations

## Benchmarking Protocol
1. Run 3 warmup iterations
2. Measure 5 inference runs
3. Record: min, max, mean, std deviation
4. Save sample images for quality comparison
5. Log memory usage patterns

## Progress Tracking
**IMPORTANT**: Create a `progress.md` file in this folder to track your progress.

### Progress Template
```markdown
# Progress: Baseline torch.compile Optimization

## Overall Progress: [X]%

### Completed Tasks:
- [ ] Environment setup
- [ ] Dependencies installed
- [ ] Baseline script created
- [ ] torch.compile optimizations implemented
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
- Performance benchmarks with error bars
- Memory usage analysis
- Compilation time measurements
- Quality assessment with sample images
- **progress.md file with percentage completion**

## Cleanup After Testing
**IMPORTANT**: Clean up caches after testing to prevent disk space issues:
```bash
# Remove Hugging Face cache
rm -rf ~/.cache/huggingface

# Remove PyTorch compilation cache
rm -rf ~/.cache/torch

# Remove pip cache
pip cache purge

# Remove virtual environment if no longer needed
# rm -rf venv_torch_compile
