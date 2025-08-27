# Test Plan: SVDQuant + Nunchaku 4-bit Optimization

## Objective
Test aggressive 4-bit quantization using SVDQuant with Nunchaku inference engine for maximum memory efficiency.

## Expected Performance
- Target: 3.5× memory reduction, 8.7× latency reduction vs NF4 baseline
- Expected time: ~3 seconds for 1024px image (4 steps)
- Enables FLUX on 16GB GPUs with high performance

## Prerequisites Check
1. **System Requirements**:
   - NVIDIA GPU with compute capability 7.5+ (RTX 20 series or newer)
   - Python 3.10+
   - CUDA 12.2+
   - At least 16GB GPU memory recommended

2. **Check GPU Compatibility**:
   ```bash
   nvidia-smi
   python -c "import torch; print(f'CUDA Compute Capability: {torch.cuda.get_device_capability()}')"
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
python -m venv venv_svdquant
source venv_svdquant/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Installation Steps
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clone and install Nunchaku
git clone https://github.com/mit-han-lab/nunchaku.git
cd nunchaku
pip install -e .

# Install additional dependencies
pip install diffusers transformers accelerate huggingface_hub
pip install Pillow safetensors numpy scipy
```

## Implementation Plan
1. **Install Nunchaku inference engine**
2. **Test SVDQuant 4-bit quantization**:
   - Weight and activation quantization to 4-bit
   - Low-rank branch for outlier absorption
   - Fused kernel optimization
3. **Compare with NF4 baseline**
4. **Test LoRA integration**
5. **Memory and performance benchmarking**

## Key Features to Test
- **4-bit W4A4 quantization** (weights + activations)
- **Low-rank branch** for outlier handling
- **Fused kernels** for minimal overhead
- **LoRA compatibility** without requantization
- **Memory efficiency** on consumer GPUs

## SVDQuant Process
1. **Smoothing**: Migrate outliers from activations to weights
2. **SVD Decomposition**: Extract low-rank component for outliers
3. **4-bit Quantization**: Quantize residual weights and activations
4. **Kernel Fusion**: Combine low-rank and low-bit computations

## Success Criteria
- [ ] Nunchaku installation successful
- [ ] SVDQuant 4-bit quantization working
- [ ] Memory usage reduced by 3.5×
- [ ] Performance improvement vs NF4
- [ ] Quality preservation acceptable
- [ ] LoRA integration functional
- [ ] Fused kernels providing speedup

## Benchmarking Protocol
1. **Memory efficiency**:
   - Peak GPU memory usage
   - Compare with FP16 and NF4 baselines
2. **Performance comparison**:
   - Inference time vs NF4 W4A16
   - Kernel fusion effectiveness
3. **Quality assessment**:
   - Visual quality vs full precision
   - Text alignment preservation
4. **LoRA testing**:
   - Style transfer with different LoRAs
   - Quality with LoRA vs without

## Special Considerations
- Requires specific Nunchaku installation
- 4-bit quantization is aggressive (quality tradeoffs)
- Low-rank branch adds computational overhead (mitigated by fusion)
- LoRA integration is a key differentiator
- May require model-specific calibration

## Progress Tracking
**IMPORTANT**: Create a `progress.md` file in this folder to track your progress.

### Progress Template
```markdown
# Progress: SVDQuant + Nunchaku 4-bit Optimization

## Overall Progress: [X]%

### Completed Tasks:
- [ ] Environment setup
- [ ] Nunchaku installed
- [ ] SVDQuant 4-bit quantization working
- [ ] Memory efficiency measured
- [ ] LoRA integration tested
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
- Memory usage comparison (FP16 vs NF4 vs SVDQuant)
- Performance benchmarks with error bars
- Quality assessment with sample images
- LoRA integration demonstration
- Kernel fusion efficiency analysis
- **progress.md file with percentage completion**

## Cleanup After Testing
**IMPORTANT**: Clean up caches after testing to prevent disk space issues:
```bash
# Remove Hugging Face cache
rm -rf ~/.cache/huggingface

# Remove PyTorch compilation cache
rm -rf ~/.cache/torch

# Remove Nunchaku cache
rm -rf ~/.cache/nunchaku

# Remove pip cache
pip cache purge

# Remove virtual environment if no longer needed
# rm -rf venv_svdquant
```
