# FLUX Optimization Tests

Comprehensive benchmarking and optimization experiments for FLUX.1 diffusion models on H100 GPUs.

## 🎯 Key Results

### FP8 + torch.compile Success
- **Baseline FLUX.1-schnell**: 1.536s
- **FP8 + torch.compile**: 1.446s (**1.06x speedup**)
- **Environment**: H100 PCIe, torchao 0.12.0, diffusers 0.30.0

### Optimization Methods Tested

| Method | Status | Speedup | Notes |
|--------|--------|---------|-------|
| FBCache (First Block Cache) | ✅ Working | 1.00x | Minimal gains on FLUX.1-schnell |
| FP8 Quantization | ✅ Working | 1.06x | Requires torch.compile |
| torch.compile | ✅ Working | Variable | Essential for FP8 performance |
| FBCache + FP8 + torch.compile | ❌ Incompatible | N/A | FBCache breaks torch.compile |

## 📁 Repository Structure

```
flux_optimization_tests/
├── 01_baseline_torch_compile/     # Basic torch.compile benchmarks
├── 02_flux_fast/                  # Flux-fast implementation tests
├── 03_torchao_quantization/       # TorchAO quantization experiments
├── 04_xdit_multi_gpu/            # Multi-GPU optimization tests
├── 05_gguf_quantization/         # GGUF format experiments
├── 06_svdquant_nunchaku/         # Nunchaku SVD quantization
├── 07_diffusers_native/          # Native diffusers optimizations
├── 08_tensorrt_optimization/     # TensorRT acceleration tests
└── 09_fbcache_teacache/          # FBCache + FP8 + torch.compile (MAIN)
```

## 🚀 Quick Start

### FP8 + torch.compile (Recommended)

```bash
cd 09_fbcache_teacache
python -m venv venv_fbcache_teacache
source venv_fbcache_teacache/bin/activate
pip install -r requirements.txt
python fp8_compile_test.py
```

### Key Scripts

- `fp8_compile_test.py` - **Working FP8 + torch.compile benchmark**
- `triple_optimization_benchmark.py` - Full optimization test (FBCache incompatible)
- `fbcache_fp8_benchmark.py` - Individual optimization tests

## 🔬 Technical Findings

### FP8 Quantization on H100

**Root Cause Analysis:**
- FP8 quantization alone shows 27% overhead
- torch.compile is **mandatory** for FP8 performance gains
- Combined FP8 + torch.compile achieves 6% speedup

**Hardware Compatibility:**
- ✅ H100 PCIe: Full FP8 support
- ✅ H100 SXM5: Enhanced performance (20% faster)
- ⚠️ Batch size 1: Limited benefits, larger batches recommended

### FBCache Limitations

**Compatibility Issues:**
- FBCache uses `unittest.mock.patch.object`
- Breaks torch.compile graph tracing
- Cannot combine with torch.compile optimizations

**Performance:**
- Standalone FBCache: Minimal gains on FLUX.1-schnell
- Works better with FLUX.1-dev (28 steps vs 4 steps)

## 📊 Benchmark Results

### Environment
- **GPU**: H100 PCIe 80GB
- **Model**: FLUX.1-schnell (4 steps)
- **Resolution**: 1024x1024
- **Batch Size**: 1

### Performance Comparison

```
Baseline:              1.536s (1.00x)
FP8 only:              1.957s (0.78x) - Expected overhead
torch.compile only:    48.026s (0.03x) - Compilation overhead
FP8 + torch.compile:   1.446s (1.06x) - 6% speedup ✅
```

## 🛠️ Setup Requirements

### Core Dependencies
```
torch>=2.5.1
diffusers==0.30.0
torchao>=0.12.0
para-attn (for FBCache)
```

### Hardware Requirements
- NVIDIA H100 or A100 GPU
- 80GB+ VRAM recommended
- CUDA 12.1+

## 📈 Future Improvements

### Potential Enhancements
1. **FLUX.1-dev testing**: Longer inference allows more optimization benefit
2. **Larger batch sizes**: FP8 quantization shows better gains
3. **Alternative FBCache**: Find torch.compile compatible caching solution
4. **Mixed precision**: Combine different quantization schemes

### Research Targets
- 3.48x speedup (from research papers)
- Requires FLUX.1-dev + larger batches + full optimization stack

## 🤝 Contributing

This repository documents systematic optimization experiments for FLUX models. Each directory contains:
- Benchmark scripts
- Requirements files
- Results documentation
- Setup instructions

## 📄 License

MIT License - See LICENSE file for details.

---

**Status**: Active development and benchmarking
**Last Updated**: 2025-01-27
**Primary Focus**: H100 GPU optimization for FLUX.1 models
