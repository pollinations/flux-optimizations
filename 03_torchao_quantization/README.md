# Test Plan: TorchAO Quantization Optimization

## Objective
Test comprehensive TorchAO quantization techniques for FLUX Schnell, focusing on FP8 dynamic row-wise quantization.

## Expected Performance
- Target: 53.88% speedup (3.4x faster than baseline)
- Expected time: ~2.966 seconds for 1024px image (4 steps)
- Significant memory reduction with minimal quality loss

## Prerequisites Check
1. **System Requirements**:
   - NVIDIA H100 GPU (CUDA compute capability 8.9+)
   - Python 3.10+
   - CUDA 12.2+
   - At least 16GB GPU memory

2. **Verify H100 Support**:
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
python -m venv venv_torchao
source venv_torchao/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Installation Steps
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install TorchAO for quantization
pip install torchao

# Install diffusers and dependencies
pip install diffusers transformers accelerate huggingface_hub

# Install additional dependencies
pip install Pillow safetensors
```

## Implementation Plan
1. **Clone diffusers-torchao repository** for reference implementations
2. **Test multiple quantization schemes**:
   - FP8 dynamic activation + FP8 weight (row-wise scaling)
   - FP8 dynamic activation + FP8 weight (tensor-wise scaling)
   - INT8 dynamic quantization
   - INT4 weight-only quantization
   - Semi-structured sparsity + INT8 (batch size 16+)

3. **Benchmark each quantization method**
4. **Quality assessment** for each technique
5. **Memory usage analysis**

## Key Quantization Techniques to Test

### FP8 Row-wise (Primary Target)
```python
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, PerRow
quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()))
```

### FP8 Tensor-wise
```python
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, PerTensor
quantize_(model, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))
```

### INT8 Dynamic
```python
from torchao.quantization import Int8DynamicActivationIntxWeightConfig
quantize_(model, Int8DynamicActivationIntxWeightConfig())
```

## Success Criteria
- [ ] TorchAO installation successful
- [ ] FP8 quantization working on H100
- [ ] fp8dqrow achieving target performance
- [ ] Quality degradation < 5% (subjective assessment)
- [ ] Memory usage reduction measured
- [ ] Batch size scaling tested
- [ ] Semi-structured sparsity evaluated

## Benchmarking Protocol
1. **Baseline measurement** (no quantization)
2. **Individual quantization schemes**:
   - Performance timing
   - Memory usage
   - Quality assessment
3. **Batch size scaling** (1, 4, 8, 16)
4. **Quality comparison** with reference images
5. **Memory efficiency analysis**

## Special Considerations
- FP8 requires H100 or newer (compute capability 8.9+)
- Semi-structured sparsity needs CUDA 12.4+ and specific Docker
- Quality vs speed tradeoffs vary by quantization method
- Batch size affects optimal quantization choice

## Progress Tracking
**IMPORTANT**: Create a `progress.md` file in this folder to track your progress.

### Progress Template
```markdown
# Progress: TorchAO Quantization Optimization

## Overall Progress: [X]%

### Completed Tasks:
- [ ] Environment setup
- [ ] TorchAO installed
- [ ] H100 compatibility verified
- [ ] FP8 quantization implemented
- [ ] Multiple schemes tested
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
- Quantization performance matrix
- Memory usage comparison
- Quality assessment images
- Batch size scaling results
- Recommendations for production use
- **progress.md file with percentage completion**

## Cleanup After Testing
**IMPORTANT**: Clean up caches after testing to prevent disk space issues:
```bash
# Remove Hugging Face cache
rm -rf ~/.cache/huggingface

# Remove PyTorch compilation cache
rm -rf ~/.cache/torch

# Remove TorchAO cache
rm -rf ~/.cache/torchao

# Remove pip cache
pip cache purge

# Remove virtual environment if no longer needed
# rm -rf venv_torchao
```
