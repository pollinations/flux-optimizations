#!/usr/bin/env python3
"""
FP8 + torch.compile test for FLUX.1-schnell (without FBCache due to compatibility issues)
Testing if torch.compile fixes the FP8 performance issue
"""

import torch
import time
from diffusers import FluxPipeline
from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
from torchao.quantization.quant_api import PerRow

# Optimize for H100/A100 performance
torch.set_float32_matmul_precision("high")

# Configuration
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEVICE = "cuda"
DTYPE = torch.bfloat16

def benchmark_baseline():
    """Benchmark baseline without any optimizations"""
    print("\n=== Baseline Test ===")
    
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    
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
    baseline_time = time.time() - start_time
    
    print(f"Baseline time: {baseline_time:.3f}s")
    image.save("baseline_fp8_test.png")
    return baseline_time

def benchmark_fp8_only():
    """Benchmark FP8 quantization without torch.compile"""
    print("\n=== FP8 Only Test ===")
    
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    
    # Apply fp8 quantization
    print("Applying fp8dqrow quantization...")
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
    
    print(f"FP8 only time: {fp8_time:.3f}s")
    image.save("fp8_only_test.png")
    return fp8_time

def benchmark_compile_only():
    """Benchmark torch.compile without FP8"""
    print("\n=== torch.compile Only Test ===")
    
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    
    # Apply torch.compile
    print("Compiling transformer...")
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
    
    print(f"torch.compile only time: {compile_time:.3f}s")
    image.save("compile_only_test.png")
    return compile_time

def benchmark_fp8_compile():
    """Benchmark FP8 + torch.compile combined"""
    print("\n=== FP8 + torch.compile Combined Test ===")
    
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    
    # Apply fp8 quantization first
    print("1. Applying fp8dqrow quantization...")
    quantize_(pipe.transformer, float8_dynamic_activation_float8_weight(granularity=PerRow()))
    
    # Apply torch.compile
    print("2. Compiling transformer...")
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
    combined_time = time.time() - start_time
    
    print(f"FP8 + torch.compile time: {combined_time:.3f}s")
    image.save("fp8_compile_test.png")
    return combined_time

def main():
    print("FLUX.1-schnell FP8 + torch.compile Test")
    print("Testing if torch.compile fixes FP8 performance issues")
    print(f"Device: {DEVICE}")
    
    # Run benchmarks
    baseline_time = benchmark_baseline()
    fp8_time = benchmark_fp8_only()
    compile_time = benchmark_compile_only()
    combined_time = benchmark_fp8_compile()
    
    # Results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Baseline:              {baseline_time:.3f}s (1.00x)")
    print(f"FP8 only:              {fp8_time:.3f}s ({baseline_time/fp8_time:.2f}x)")
    print(f"torch.compile only:    {compile_time:.3f}s ({baseline_time/compile_time:.2f}x)")
    print(f"FP8 + torch.compile:   {combined_time:.3f}s ({baseline_time/combined_time:.2f}x)")
    
    # Analysis
    print(f"\nAnalysis:")
    print(f"FP8 overhead without compile: {((fp8_time/baseline_time - 1) * 100):+.1f}%")
    print(f"torch.compile speedup: {((baseline_time/compile_time - 1) * 100):+.1f}%")
    print(f"Combined optimization: {((baseline_time/combined_time - 1) * 100):+.1f}%")
    
    print(f"\nGPU Memory Usage: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

if __name__ == "__main__":
    main()
