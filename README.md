# FLUX Schnell H100 Optimization Testing Framework

## Overview
This directory contains 8 parallel testing environments for evaluating different FLUX Schnell optimization approaches on H100 GPUs. Each subfolder has its own virtual environment and detailed testing plan.

## Testing Structure

### 01_baseline_torch_compile/ ✅ COMPLETED
**Target**: Establish baseline with standard torch.compile optimizations
- **Status**: 100% complete - torch.compile working, memory constraints identified
- **Key Features**: H100-specific inductor flags, regional compilation
- **Virtual Environment**: Create fresh when needed: `python -m venv venv_torch_compile`

### 02_flux_fast/
**Target**: Official Hugging Face flux-fast optimizations
- **Expected Performance**: 2.5x speedup (~4s)
- **Key Features**: Flash Attention v3, FP8, torch.export + AOTI
- **Virtual Environment**: Create fresh when needed: `python -m venv venv_flux_fast`

### 03_torchao_quantization/
**Target**: TorchAO quantization techniques
- **Expected Performance**: 3.4x speedup (~2.966s)
- **Key Features**: FP8 row-wise quantization, INT8 dynamic
- **Virtual Environment**: Create fresh when needed: `python -m venv venv_torchao`

### 04_xdit_multi_gpu/ 🔄 60% COMPLETE
**Target**: Multi-GPU parallel inference
- **Expected Performance**: 2.6x on 4xH100, sub-1s on 8xH100
- **Key Features**: Ulysses/Ring parallelism, PipeFusion
- **Virtual Environment**: Create fresh when needed: `python -m venv venv_xdit`

### 05_gguf_quantization/
**Target**: GGUF block-wise quantization
- **Expected Performance**: Memory efficiency, moderate speedup
- **Key Features**: Q2_K/Q4_K/Q8_0 quantization, dynamic dequantization
- **Virtual Environment**: Create fresh when needed: `python -m venv venv_gguf`

### 06_svdquant_nunchaku/ ❌ H100 INCOMPATIBLE
**Target**: Aggressive 4-bit quantization
- **Status**: Not compatible with H100 GPUs (lacks 4-bit tensor cores)
- **Key Features**: W4A4 quantization, low-rank branch, LoRA support
- **Virtual Environment**: Skip - incompatible with hardware

### 07_diffusers_native/
**Target**: Built-in diffusers optimizations
- **Expected Performance**: 20-40% speedup, significant memory reduction
- **Key Features**: SDPA, xFormers, model offloading, VAE optimizations
- **Virtual Environment**: Create fresh when needed: `python -m venv venv_diffusers_native`

### 08_tensorrt_optimization/
**Target**: NVIDIA TensorRT engine optimization
- **Expected Performance**: Hardware-optimized inference (~2-3s)
- **Key Features**: FP8 quantization, kernel fusion, engine compilation
- **Virtual Environment**: Create fresh when needed: `python -m venv venv_tensorrt`

## Parallel Testing Instructions

### For Each Sub-Agent:
1. **Navigate to assigned folder**:
   ```bash
   cd flux_optimization_tests/[folder_name]
   ```

2. **Read the README.md** for detailed instructions

3. **Follow the plan exactly**:
   - Check prerequisites
   - Create virtual environment
   - Install dependencies
   - Implement optimizations
   - Run benchmarks
   - Generate reports

### Standardized Output Format
Each test should produce:
- **Performance metrics** (JSON format)
- **Sample images** for quality comparison
- **Memory usage logs**
- **Error logs** (if any)
- **Summary report** with recommendations

## Coordination
- Each test runs independently in its own environment
- Results will be aggregated for final comparison
- Focus on your assigned optimization technique
- Document any issues or unexpected findings

## Success Criteria
- All 8 optimization approaches tested
- Performance comparison matrix generated
- Quality assessment completed
- Memory usage analysis documented
- Production recommendations provided
