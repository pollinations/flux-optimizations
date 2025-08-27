# Test Plan: Flux-Fast Optimization

## Objective
Test the official Hugging Face flux-fast repository optimizations for maximum H100 performance.

## Expected Performance
- Target: 2.5x speedup over baseline
- Expected time: ~4 seconds for 1024px image (4 steps)
- Best-in-class single GPU performance

## Prerequisites Check
1. **System Requirements**:
   - NVIDIA H100 GPU with CUDA 12.6+
   - Python 3.10+
   - At least 20GB GPU memory
   - Linux environment (required for nightly builds)

2. **Check GPU and CUDA**:
   ```bash
   nvidia-smi
   nvcc --version
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
python -m venv venv_flux_fast
source venv_flux_fast/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Installation Steps
```bash
# Install PyTorch nightly (required for latest optimizations)
pip install --pre torch==2.8.0.dev20250605+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126

# Install TorchAO nightly (required for FP8 quantization)
pip install --pre torchao==0.12.0.dev20250610+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126

# Install diffusers with latest fixes
pip install -U diffusers

# Install Flash Attention v3 (H100 specific)
pip install flash-attn==3.0.0b1 --no-build-isolation

# Install additional dependencies
pip install huggingface_hub[hf_xet] accelerate transformers Pillow safetensors
```

## Implementation Plan
1. **Clone flux-fast repository**
2. **Test all optimization layers**:
   - BFloat16 precision
   - torch.compile with fullgraph + max-autotune
   - Fused q,k,v projections
   - torch.channels_last memory format
   - Flash Attention v3 with FP8 conversion
   - Dynamic float8 quantization
   - Inductor tuning flags
   - torch.export + AOTI + CUDAGraphs
   - Cache acceleration (DBCache)

3. **Benchmark incremental optimizations**
4. **Test cached model performance**

## Key Features to Test
- All optimizations enabled by default
- Pre-cached binary models via torch.export + AOTI
- Individual optimization toggles for ablation study
- FP8 quantization quality vs speed tradeoff

## Success Criteria
- [ ] Successful nightly PyTorch installation
- [ ] Flash Attention v3 working on H100
- [ ] All flux-fast optimizations enabled
- [ ] 2.5x+ speedup achieved
- [ ] Cached model generation working
- [ ] Sub-4 second inference time
- [ ] Quality preservation with optimizations

## Benchmarking Protocol
1. Test baseline flux-fast performance
2. Test cached model performance
3. Run ablation study (disable optimizations one by one)
4. Compare quality with/without FP8 quantization
5. Measure compilation time vs runtime gains

## Special Notes
- Cached binaries are hardware-specific (H100 only)
- First run will be slow due to compilation
- FP8 quantization has minor quality impact
- Requires specific PyTorch/TorchAO versions

## Progress Tracking
**IMPORTANT**: Create a `progress.md` file in this folder to track your progress.

### Progress Template
```markdown
# Progress: Flux-Fast Optimization

## Overall Progress: [X]%

### Completed Tasks:
- [ ] Environment setup
- [ ] PyTorch nightly installed
- [ ] Flash Attention v3 installed
- [ ] flux-fast repository cloned
- [ ] All optimizations tested
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
- Performance comparison (baseline vs flux-fast)
- Memory usage analysis
- Quality assessment with sample images
- Installation and setup documentation
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
# rm -rf venv_flux_fast
