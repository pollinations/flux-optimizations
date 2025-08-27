# Test Plan: First Block Cache + TeaCache Optimization

## Objective
Test First Block Cache (FBCache) and TeaCache dynamic caching techniques for FLUX Schnell, both individually and in combination for maximum acceleration.

## Expected Performance
- **FBCache alone**: 2-3x speedup (~1.7-2.5 seconds for 1024px image)
- **TeaCache alone**: 1.3-1.6x speedup (~3.1-3.8 seconds for 1024px image)
- **Combined FBCache + TeaCache**: Up to 4x speedup (~1.2-1.5 seconds for 1024px image)
- **Quality impact**: Minimal with proper threshold tuning

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
   ```

## Virtual Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv_fbcache_teacache
source venv_fbcache_teacache/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Installation Steps
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ParaAttention for FBCache
pip install para-attn

# Install diffusers and dependencies
pip install diffusers transformers accelerate huggingface_hub

# Install additional dependencies
pip install Pillow safetensors numpy matplotlib seaborn

# Clone TeaCache repository for integration
git clone https://github.com/ali-vilab/TeaCache.git
cd TeaCache/TeaCache4FLUX
pip install -r requirements.txt
cd ../..
```

## Implementation Plan
1. **Test FBCache alone**:
   - Implement basic FBCache with diffusers
   - Test different residual_diff_threshold values
   - Benchmark performance and quality

2. **Test TeaCache alone**:
   - Integrate TeaCache4FLUX implementation
   - Test different caching strategies
   - Benchmark performance and quality

3. **Test FBCache + TeaCache combination**:
   - Implement both caching methods together
   - Optimize threshold combinations
   - Measure cumulative speedup

4. **Threshold optimization**:
   - Grid search optimal thresholds
   - Quality vs speed analysis
   - Production recommendations

## Key Features to Test

### FBCache Configuration
- **Residual diff thresholds**: 0.0, 0.05, 0.1, 0.12, 0.15, 0.2
- **Integration with torch.compile**
- **Memory usage optimization**

### TeaCache Configuration
- **Caching strategies**: Timestep-based caching
- **Rescaling coefficients**: FLUX-specific tuning
- **Cache hit rate optimization**

### Combined Configuration
- **Optimal threshold combinations**
- **Cache interaction analysis**
- **Memory efficiency with dual caching**

## Success Criteria
- [ ] ParaAttention (FBCache) installation successful
- [ ] TeaCache integration working
- [ ] FBCache achieving 2x+ speedup
- [ ] TeaCache achieving 1.3x+ speedup
- [ ] Combined approach achieving 3x+ speedup
- [ ] Quality degradation < 5% (subjective assessment)
- [ ] Threshold optimization completed
- [ ] Production recommendations documented

## Benchmarking Protocol
1. **Baseline measurement** (no caching)
2. **FBCache testing**:
   - Multiple threshold values
   - Performance and quality metrics
3. **TeaCache testing**:
   - Different caching strategies
   - Performance and quality metrics
4. **Combined testing**:
   - Optimal threshold combinations
   - Cumulative performance gains
5. **Quality assessment**:
   - Side-by-side image comparisons
   - CLIP score analysis (optional)

## Test Configurations

### Recommended FBCache Thresholds for FLUX
- **Conservative**: 0.05 (minimal quality loss, ~1.5x speedup)
- **Balanced**: 0.12 (good balance, ~2x speedup)
- **Aggressive**: 0.2 (maximum speed, ~2.5x speedup)

### Recommended Test Prompts
- Simple: "A red apple on a white table"
- Complex: "A futuristic cityscape at sunset with flying cars and neon lights"
- Detailed: "A photorealistic portrait of an elderly man with weathered hands holding a vintage camera"

## Progress Tracking
**IMPORTANT**: Create a `progress.md` file in this folder to track your progress.

### Progress Template
```markdown
# Progress: FBCache + TeaCache Optimization

## Overall Progress: [X]%

### Completed Tasks:
- [ ] Environment setup
- [ ] ParaAttention installed
- [ ] TeaCache integrated
- [ ] FBCache testing completed
- [ ] TeaCache testing completed
- [ ] Combined testing completed
- [ ] Threshold optimization completed
- [ ] Results documented

### Current Status:
[Brief description of current work]

### Issues/Blockers:
[Any problems encountered]

### Next Steps:
[What's planned next]
```

## Output Requirements
- FBCache vs TeaCache performance comparison
- Combined optimization results
- Threshold optimization charts
- Quality assessment images
- Memory usage analysis
- Production deployment recommendations
- **progress.md file with percentage completion**

## Special Considerations
- FBCache and TeaCache operate at different levels (can be combined)
- Threshold tuning is critical for quality preservation
- Memory usage may increase with dual caching
- torch.compile compatibility with both methods
- Batch size effects on caching efficiency

## Cleanup After Testing
**IMPORTANT**: Clean up caches after testing to prevent disk space issues:
```bash
# Remove Hugging Face cache
rm -rf ~/.cache/huggingface

# Remove PyTorch compilation cache
rm -rf ~/.cache/torch

# Remove ParaAttention cache
rm -rf ~/.cache/para_attn

# Remove pip cache
pip cache purge

# Remove TeaCache repository if no longer needed
# rm -rf TeaCache

# Remove virtual environment if no longer needed
# rm -rf venv_fbcache_teacache
```
