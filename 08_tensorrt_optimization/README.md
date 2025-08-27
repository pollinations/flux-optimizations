# Test Plan: TensorRT Optimization

## Objective
Test NVIDIA TensorRT Model Optimizer and TensorRT engine compilation for FLUX Schnell with FP8/FP4 quantization.

## Expected Performance
- Target: Significant speedup with kernel fusion and FP8 quantization
- Expected time: ~2-3 seconds for 1024px image (4 steps)
- Hardware-optimized inference with minimal quality loss

## Prerequisites Check
1. **System Requirements**:
   - NVIDIA H100 GPU (for FP8 support)
   - Python 3.10+
   - CUDA 12.2+
   - TensorRT 10.0+
   - At least 16GB GPU memory

2. **Check TensorRT Installation**:
   ```bash
   nvidia-smi
   python -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
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
python -m venv venv_tensorrt
source venv_tensorrt/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Installation Steps
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install TensorRT (if not system-installed)
pip install tensorrt

# Install TensorRT Model Optimizer
pip install nvidia-modelopt[torch]

# Install diffusers and dependencies
pip install diffusers transformers accelerate huggingface_hub

# Install ONNX for model export
pip install onnx onnxruntime-gpu

# Install additional dependencies
pip install Pillow safetensors
```

## Implementation Plan
1. **Setup TensorRT Model Optimizer**
2. **Configure quantization schema**:
   - FP8 post-training quantization
   - FP4 quantization (if supported)
   - Per-tensor vs per-block quantization
3. **Calibration dataset preparation**
4. **Model optimization and export**:
   - Inject quantization layers
   - Post-training calibration
   - ONNX export
   - TensorRT engine compilation
5. **Performance benchmarking**

## TensorRT Optimization Process

### 1. Quantization Configuration
```python
# Define FP8 quantization config
from modelopt.torch.quantization import config
quant_config = config.INT8_DEFAULT_CFG
# Customize for FP8 if available
```

### 2. Model Preparation
```python
# Load FLUX model
# Apply quantization configuration
# Prepare calibration dataset
```

### 3. Calibration and Export
```python
# Run post-training calibration
# Export to ONNX format
# Compile TensorRT engine
```

### 4. Inference with TensorRT
```python
# Load TensorRT engine
# Run optimized inference
```

## Key Features to Test
- **FP8 quantization** with minimal quality loss
- **Kernel fusion** for transformer operations
- **Dynamic shapes** support
- **Calibration dataset** effectiveness
- **Engine compilation** time vs runtime gains

## Success Criteria
- [ ] TensorRT Model Optimizer installation successful
- [ ] Quantization configuration working
- [ ] Calibration process completing
- [ ] ONNX export successful
- [ ] TensorRT engine compilation working
- [ ] Optimized inference functional
- [ ] Performance improvement achieved
- [ ] Quality preservation acceptable

## Benchmarking Protocol
1. **Baseline measurement** (PyTorch)
2. **Optimization process timing**:
   - Calibration time
   - ONNX export time
   - Engine compilation time
3. **Runtime performance**:
   - First inference (cold)
   - Warm inference timing
   - Memory usage
4. **Quality assessment**:
   - Compare with FP16 baseline
   - Quantization artifacts evaluation

## Special Considerations
- TensorRT engines are hardware-specific
- FP8 requires H100 or newer architecture
- Calibration dataset quality affects results
- Engine compilation can be time-consuming
- Dynamic shapes may impact performance

## Progress Tracking
**IMPORTANT**: Create a `progress.md` file in this folder to track your progress.

### Progress Template
```markdown
# Progress: TensorRT Optimization

## Overall Progress: [X]%

### Completed Tasks:
- [ ] Environment setup
- [ ] TensorRT Model Optimizer installed
- [ ] Quantization configuration defined
- [ ] Model calibration completed
- [ ] TensorRT engine compiled
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
- TensorRT optimization pipeline documentation
- Performance comparison (PyTorch vs TensorRT)
- Engine compilation time analysis
- Quality assessment with quantization
- Memory usage optimization results
- **progress.md file with percentage completion**

## Cleanup After Testing
**IMPORTANT**: Clean up caches after testing to prevent disk space issues:
```bash
# Remove Hugging Face cache
rm -rf ~/.cache/huggingface

# Remove PyTorch compilation cache
rm -rf ~/.cache/torch

# Remove TensorRT cache
rm -rf ~/.cache/tensorrt

# Remove pip cache
pip cache purge

# Remove virtual environment if no longer needed
# rm -rf venv_tensorrt
```
