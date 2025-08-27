# Test Plan: xDiT Multi-GPU Optimization

## Objective
Test xDiT's hybrid parallel inference across multiple H100 GPUs for maximum cluster performance.

## Expected Performance
- Target: 2.6x speedup on 4xH100, sub-1 second on 8xH100
- Expected time: ~1.6 seconds for 1024px image (4 steps) on 4xH100
- Best multi-GPU scaling performance

## Prerequisites Check
1. **System Requirements**:
   - Multiple NVIDIA H100 GPUs (2-8 recommended)
   - NVLink interconnect (critical for performance)
   - Python 3.10+
   - CUDA 12.2+
   - At least 16GB per GPU

2. **Check Multi-GPU Setup**:
   ```bash
   nvidia-smi
   nvidia-smi topo -m  # Check NVLink topology
   python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
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
python -m venv venv_xdit
source venv_xdit/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Installation Steps
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install xDiT
pip install xdit

# Install additional dependencies
pip install diffusers transformers accelerate huggingface_hub
pip install Pillow safetensors
```

## Implementation Plan
1. **Test single GPU baseline** with xDiT
2. **Test parallel strategies**:
   - Ulysses sequence parallelism (best for NVLink)
   - Ring attention parallelism
   - Hybrid combinations (Ulysses + Ring)
   - PipeFusion for 8+ GPUs
   - VAE parallel for high-resolution

3. **Benchmark scaling efficiency**
4. **Test torch.compile integration**
5. **Evaluate memory distribution**

## Key Configurations to Test

### 2xH100 Setup
```bash
# Ring-2 (optimal for 2 GPUs)
python -m xdit.run --model black-forest-labs/FLUX.1-schnell --ring_degree 2
```

### 4xH100 Setup
```bash
# Ulysses-4 (best performance)
python -m xdit.run --model black-forest-labs/FLUX.1-schnell --ulysses_degree 4

# Hybrid: Ulysses-2 x Ring-2
python -m xdit.run --model black-forest-labs/FLUX.1-schnell --ulysses_degree 2 --ring_degree 2
```

### 8xH100 Setup
```bash
# Ulysses-8 (maximum parallelism)
python -m xdit.run --model black-forest-labs/FLUX.1-schnell --ulysses_degree 8

# Hybrid with PipeFusion
python -m xdit.run --model black-forest-labs/FLUX.1-schnell --ulysses_degree 4 --pipefusion_degree 2
```

## Advanced Features to Test
- **VAE Parallel**: `--use_parallel_vae` for >2048px generation
- **torch.compile**: `--compile` for additional speedup
- **High Resolution**: Test 2048px and 4096px generation
- **Batch Processing**: Multiple prompts in parallel

## Success Criteria
- [ ] xDiT installation successful
- [ ] Multi-GPU detection working
- [ ] Parallel inference functional
- [ ] Linear scaling achieved (2x GPUs = ~2x speed)
- [ ] NVLink utilization confirmed
- [ ] Memory distribution balanced
- [ ] High-resolution generation working

## Benchmarking Protocol
1. **Single GPU baseline** measurement
2. **Scaling efficiency**:
   - 1 GPU vs 2 GPU vs 4 GPU vs 8 GPU
   - Measure actual speedup vs theoretical
3. **Strategy comparison**:
   - Ulysses vs Ring vs Hybrid
4. **Memory analysis**:
   - Per-GPU memory usage
   - Communication overhead
5. **Quality validation**: Ensure distributed generation matches single GPU

## Special Considerations
- NVLink bandwidth critical for performance
- PCIe interconnect significantly slower
- Memory must be balanced across GPUs
- Some strategies better for different resolutions
- PipeFusion beneficial for 8+ GPUs

## Progress Tracking
**IMPORTANT**: Create a `progress.md` file in this folder to track your progress.

### Progress Template
```markdown
# Progress: xDiT Multi-GPU Optimization

## Overall Progress: [X]%

### Completed Tasks:
- [ ] Environment setup
- [ ] Multi-GPU detection verified
- [ ] xDiT installed
- [ ] Parallel strategies tested
- [ ] Scaling efficiency measured
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
- Multi-GPU scaling efficiency charts
- Strategy performance comparison
- Memory usage distribution analysis
- NVLink utilization metrics
- Quality consistency validation
- **progress.md file with percentage completion**

## Cleanup After Testing
**IMPORTANT**: Clean up caches after testing to prevent disk space issues:
```bash
# Remove Hugging Face cache
rm -rf ~/.cache/huggingface

# Remove PyTorch compilation cache
rm -rf ~/.cache/torch

# Remove xDiT cache
rm -rf ~/.cache/xdit

# Remove pip cache
pip cache purge

# Remove virtual environment if no longer needed
# rm -rf venv_xdit
```
