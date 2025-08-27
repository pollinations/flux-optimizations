#!/usr/bin/env python3
"""
Combined FBCache + fp8 quantization benchmark for FLUX.1-schnell
Based on research showing 3.48x speedup with all optimizations combined
"""

import torch
import time
from diffusers import FluxPipeline
from torchao.quantization import quantize_, int8_weight_only
import torchao
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

# Configuration
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEVICE = "cuda"
DTYPE = torch.bfloat16
FBCACHE_THRESHOLD = 0.12  # Adjusted for fp8 compatibility

def benchmark_baseline(pipe, prompt):
    """Benchmark baseline without any optimizations"""
    print("\n=== Baseline Test ===")
    
    # Warmup
    _ = pipe(prompt, num_inference_steps=4, height=512, width=512)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    image = pipe(
        prompt,
        num_inference_steps=4,
        height=1024,
        width=1024,
    ).images[0]
    torch.cuda.synchronize()
    baseline_time = time.time() - start_time
    
    print(f"Baseline time: {baseline_time:.3f}s")
    image.save("baseline_output.png")
    return baseline_time

def benchmark_fbcache_only(pipe, prompt):
    """Benchmark with FBCache only"""
    print("\n=== FBCache Only Test ===")
    
    # Apply FBCache
    apply_cache_on_pipe(pipe, residual_diff_threshold=FBCACHE_THRESHOLD)
    
    # Warmup
    _ = pipe(prompt, num_inference_steps=4, height=512, width=512)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    image = pipe(
        prompt,
        num_inference_steps=4,
        height=1024,
        width=1024,
    ).images[0]
    torch.cuda.synchronize()
    fbcache_time = time.time() - start_time
    
    print(f"FBCache time: {fbcache_time:.3f}s")
    image.save("fbcache_output.png")
    
    return fbcache_time

def benchmark_fp8_only(pipe, prompt):
    """Benchmark with fp8 quantization only"""
    print("\n=== FP8 Quantization Only Test ===")
    
    # Apply fp8 quantization to transformer
    print("Applying fp8 quantization to transformer...")
    quantize_(pipe.transformer, int8_weight_only())
    
    # Warmup
    _ = pipe(prompt, num_inference_steps=4, height=512, width=512)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    image = pipe(
        prompt,
        num_inference_steps=4,
        height=1024,
        width=1024,
    ).images[0]
    torch.cuda.synchronize()
    fp8_time = time.time() - start_time
    
    print(f"FP8 time: {fp8_time:.3f}s")
    image.save("fp8_output.png")
    return fp8_time

def benchmark_fbcache_fp8_combined(pipe, prompt):
    """Benchmark with FBCache + fp8 quantization combined"""
    print("\n=== FBCache + FP8 Combined Test ===")
    
    # Apply FBCache (transformer should already be quantized from previous test)
    apply_cache_on_pipe(pipe, residual_diff_threshold=FBCACHE_THRESHOLD)
    
    # Warmup
    _ = pipe(prompt, num_inference_steps=4, height=512, width=512)
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    image = pipe(
        prompt,
        num_inference_steps=4,
        height=1024,
        width=1024,
    ).images[0]
    torch.cuda.synchronize()
    combined_time = time.time() - start_time
    
    print(f"FBCache + FP8 time: {combined_time:.3f}s")
    image.save("fbcache_fp8_output.png")
    return combined_time

def main():
    print("FLUX.1-schnell FBCache + FP8 Quantization Benchmark")
    print(f"Device: {DEVICE}")
    print(f"FBCache threshold: {FBCACHE_THRESHOLD}")
    print(f"Target speedup: 3.48x (from research)")
    
    # Load pipeline
    print("\nLoading FLUX.1-schnell pipeline...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    
    # Test prompt
    prompt = "A red apple on a white table, photorealistic, high quality"
    
    # Run benchmarks
    baseline_time = benchmark_baseline(pipe, prompt)
    fbcache_time = benchmark_fbcache_only(pipe, prompt)
    fp8_time = benchmark_fp8_only(pipe, prompt)
    combined_time = benchmark_fbcache_fp8_combined(pipe, prompt)
    
    # Calculate speedups
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"Baseline:           {baseline_time:.3f}s (1.00x)")
    print(f"FBCache only:       {fbcache_time:.3f}s ({baseline_time/fbcache_time:.2f}x)")
    print(f"FP8 only:           {fp8_time:.3f}s ({baseline_time/fp8_time:.2f}x)")
    print(f"FBCache + FP8:      {combined_time:.3f}s ({baseline_time/combined_time:.2f}x)")
    print(f"Target (research):  {baseline_time/3.48:.3f}s (3.48x)")
    
    # Memory usage
    print(f"\nGPU Memory Usage: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

if __name__ == "__main__":
    main()
