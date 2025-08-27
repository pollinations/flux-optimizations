#!/usr/bin/env python3
"""
Triple optimization benchmark for FLUX.1-schnell: FBCache + FP8 + torch.compile
Based on sayakpaul/diffusers-torchao research showing 3.48x speedup
"""

import torch
import time
from diffusers import FluxPipeline
from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
from torchao.quantization.quant_api import PerRow
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

# Optimize for H100/A100 performance
torch.set_float32_matmul_precision("high")

# Configuration
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEVICE = "cuda"
DTYPE = torch.bfloat16
FBCACHE_THRESHOLD = 0.12  # Recommended for fp8 compatibility
ENABLE_COMPILE = True  # Enable torch.compile for full optimization testing

def benchmark_baseline(pipe, prompt):
    """Benchmark baseline without any optimizations"""
    print("\n=== Baseline Test ===")
    
    # Warmup runs
    for _ in range(3):
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

def benchmark_fbcache_only():
    """Benchmark with FBCache only"""
    print("\n=== FBCache Only Test ===")
    
    # Fresh pipeline for FBCache
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    
    # Apply FBCache
    apply_cache_on_pipe(pipe, residual_diff_threshold=FBCACHE_THRESHOLD)
    
    prompt = "A red apple on a white table, photorealistic, high quality"
    
    # Warmup
    for _ in range(3):
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

def benchmark_fp8_only():
    """Benchmark with fp8 quantization only"""
    print("\n=== FP8 Quantization Only Test ===")
    
    # Fresh pipeline for fp8
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    
    # Apply fp8dqrow quantization (best performing according to research)
    print("Applying fp8dqrow quantization to transformer...")
    quantize_(pipe.transformer, float8_dynamic_activation_float8_weight(granularity=PerRow()))
    
    prompt = "A red apple on a white table, photorealistic, high quality"
    
    # Warmup
    for _ in range(3):
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

def benchmark_compile_only():
    """Benchmark with torch.compile only"""
    if not ENABLE_COMPILE:
        print("\n=== torch.compile Only Test (SKIPPED) ===")
        print("torch.compile disabled for faster testing")
        return None
        
    print("\n=== torch.compile Only Test ===")
    
    # Fresh pipeline for compile
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    
    # Apply torch.compile
    print("Compiling transformer with max-autotune-no-cudagraphs...")
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs", fullgraph=True)
    
    prompt = "A red apple on a white table, photorealistic, high quality"
    
    # Warmup (compilation happens here)
    print("Warming up (compilation in progress)...")
    for _ in range(3):
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
    compile_time = time.time() - start_time
    
    print(f"torch.compile time: {compile_time:.3f}s")
    image.save("compile_output.png")
    return compile_time

def benchmark_fbcache_fp8_combined():
    """Benchmark with FBCache + FP8 (torch.compile disabled)"""
    print(f"\n=== FBCache + FP8 Combined Test ===")
    
    # Fresh pipeline for combined optimization
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    
    # Step 1: Apply FBCache
    print("1. Applying FBCache...")
    apply_cache_on_pipe(pipe, residual_diff_threshold=FBCACHE_THRESHOLD)
    
    # Step 2: Apply fp8 quantization
    print("2. Applying fp8dqrow quantization...")
    quantize_(pipe.transformer, float8_dynamic_activation_float8_weight(granularity=PerRow()))
    
    if ENABLE_COMPILE:
        # Step 3: Apply torch.compile
        print("3. Compiling transformer...")
        pipe.transformer.to(memory_format=torch.channels_last)
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs", fullgraph=True)
        optimization_name = "Triple optimization (FBCache + FP8 + torch.compile)"
        output_file = "triple_output.png"
    else:
        print("3. Skipping torch.compile (disabled)")
        optimization_name = "FBCache + FP8 combined"
        output_file = "fbcache_fp8_output.png"
    
    prompt = "A red apple on a white table, photorealistic, high quality"
    
    # Warmup
    warmup_msg = "Warming up (compilation in progress)..." if ENABLE_COMPILE else "Warming up..."
    print(warmup_msg)
    for _ in range(3):
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
    
    print(f"{optimization_name} time: {combined_time:.3f}s")
    image.save(output_file)
    return combined_time

def main():
    print("FLUX.1-schnell Triple Optimization Benchmark")
    print("FBCache + FP8 + torch.compile")
    print(f"Device: {DEVICE}")
    print(f"FBCache threshold: {FBCACHE_THRESHOLD}")
    print(f"Target speedup: 3.48x (from research)")
    
    # Load baseline pipeline
    print("\nLoading FLUX.1-schnell pipeline for baseline...")
    baseline_pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    baseline_pipe.set_progress_bar_config(disable=True)
    
    prompt = "A red apple on a white table, photorealistic, high quality"
    
    # Run all benchmarks
    baseline_time = benchmark_baseline(baseline_pipe, prompt)
    fbcache_time = benchmark_fbcache_only()
    fp8_time = benchmark_fp8_only()
    compile_time = benchmark_compile_only()
    combined_time = benchmark_fbcache_fp8_combined()
    
    # Calculate speedups
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Baseline:              {baseline_time:.3f}s (1.00x)")
    print(f"FBCache only:          {fbcache_time:.3f}s ({baseline_time/fbcache_time:.2f}x)")
    print(f"FP8 only:              {fp8_time:.3f}s ({baseline_time/fp8_time:.2f}x)")
    if compile_time:
        print(f"torch.compile only:    {compile_time:.3f}s ({baseline_time/compile_time:.2f}x)")
    else:
        print(f"torch.compile only:    SKIPPED")
    
    optimization_label = "FBCache + FP8 + torch.compile:" if ENABLE_COMPILE else "FBCache + FP8:"
    print(f"{optimization_label:<22} {combined_time:.3f}s ({baseline_time/combined_time:.2f}x)")
    print(f"Target (research):     {baseline_time/3.48:.3f}s (3.48x)")
    
    # Memory usage
    print(f"\nGPU Memory Usage: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    
    # Quality check
    print(f"\nGenerated images saved:")
    print(f"- baseline_output.png")
    print(f"- fbcache_output.png") 
    print(f"- fp8_output.png")
    if compile_time:
        print(f"- compile_output.png")
    if ENABLE_COMPILE:
        print(f"- triple_output.png")
    else:
        print(f"- fbcache_fp8_output.png")

if __name__ == "__main__":
    main()
