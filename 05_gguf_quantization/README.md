# Test Plan: GGUF Quantization Optimization

## Objective
Test GGUF block-wise quantization for FLUX Schnell with dynamic dequantization on H100.

## Expected Performance
- Target: Significant memory reduction with moderate speed improvement
- Expected time: Variable based on quantization level (Q2_K, Q4_K, Q8_0)
- Primary benefit: Memory efficiency for larger batch sizes

## Prerequisites Check
1. **System Requirements**:
   - NVIDIA H100 GPU
   - Python 3.10+
   - At least 8GB GPU memory (due to quantization)
   - CUDA 12.2+

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
python -m venv venv_gguf
source venv_gguf/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Installation Steps
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install diffusers and GGUF support
pip install diffusers transformers accelerate huggingface_hub
pip install gguf

# Install additional dependencies
pip install Pillow safetensors
```

## Implementation Plan
1. **Download pre-quantized GGUF models**:
   - FLUX.1-schnell Q2_K (most aggressive)
   - FLUX.1-schnell Q4_K (balanced)
   - FLUX.1-schnell Q8_0 (conservative)

2. **Test different quantization levels**
3. **Benchmark memory usage vs performance**
4. **Quality assessment across quantization levels**
5. **Test with model offloading combinations**

## GGUF Models to Test

### Q2_K Quantization (Most Aggressive)
```python
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
transformer = FluxTransformer2DModel.from_single_file(
    "https://huggingface.co/city96/FLUX.1-schnell-gguf/blob/main/flux1-schnell-Q2_K.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
)
```

### Q4_K Quantization (Balanced)
```python
transformer = FluxTransformer2DModel.from_single_file(
    "https://huggingface.co/city96/FLUX.1-schnell-gguf/blob/main/flux1-schnell-Q4_K.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
)
```

### Q8_0 Quantization (Conservative)
```python
transformer = FluxTransformer2DModel.from_single_file(
    "https://huggingface.co/city96/FLUX.1-schnell-gguf/blob/main/flux1-schnell-Q8_0.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
)
```

## Key Features to Test
- Dynamic dequantization during forward pass
- Memory usage comparison across quantization levels
- CPU offloading compatibility
- Batch size scaling with reduced memory
- Quality preservation assessment

## Success Criteria
- [ ] GGUF library installation successful
- [ ] Pre-quantized models loading correctly
- [ ] Dynamic dequantization working
- [ ] Memory usage significantly reduced
- [ ] Quality acceptable across quantization levels
- [ ] CPU offloading integration working
- [ ] Larger batch sizes possible

## Benchmarking Protocol
1. **Memory usage measurement**:
   - Peak GPU memory for each quantization level
   - Compare with FP16 baseline
2. **Performance timing**:
   - Inference speed for each quantization level
   - Dynamic dequantization overhead
3. **Quality assessment**:
   - Visual comparison across quantization levels
   - Text alignment preservation
4. **Batch size scaling**:
   - Maximum achievable batch size
   - Performance per image in batch

## Special Considerations
- GGUF models are single-file format
- Dynamic dequantization adds compute overhead
- Memory savings enable larger batch processing
- Quality degradation varies by quantization level
- Compatible with CPU offloading for extreme memory efficiency

## Progress Tracking
**IMPORTANT**: Create a `progress.md` file in this folder to track your progress.

### Progress Template
```markdown
# Progress: GGUF Quantization Optimization

## Overall Progress: [X]%

### Completed Tasks:
- [ ] Environment setup
- [ ] GGUF library installed
- [ ] Pre-quantized models downloaded
- [ ] Different quantization levels tested
- [ ] Memory usage measured
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
- Memory usage comparison chart
- Performance vs quality tradeoff analysis
- Batch size scaling results
- Quality comparison images
- Recommendations for different use cases
- **progress.md file with percentage completion**

## Cleanup After Testing
**IMPORTANT**: Clean up caches after testing to prevent disk space issues:
```bash
# Remove Hugging Face cache
rm -rf ~/.cache/huggingface

# Remove PyTorch compilation cache
rm -rf ~/.cache/torch

# Remove GGUF cache
rm -rf ~/.cache/gguf

# Remove pip cache
pip cache purge

# Remove virtual environment if no longer needed
# rm -rf venv_gguf
```
