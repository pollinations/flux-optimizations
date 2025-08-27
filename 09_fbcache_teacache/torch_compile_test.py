#!/usr/bin/env python3
"""
FLUX.1-schnell with torch.compile optimization test
Since FBCache has compatibility issues, let's test torch.compile as an alternative optimization
"""

import torch
import time
from diffusers import FluxPipeline

# Configuration
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEVICE = "cuda"
DTYPE = torch.bfloat16

def main():
    print("FLUX.1-schnell torch.compile optimization test")
    print(f"Device: {DEVICE}")
    
    # Load pipeline
    print("Loading FLUX.1-schnell pipeline...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    
    # Test prompt
    prompt = "A red apple on a white table"
    
    # Baseline test
    print("\n1. Baseline test (no optimization)...")
    start_time = time.time()
    image_baseline = pipe(
        prompt,
        num_inference_steps=4,
        height=1024,
        width=1024,
    ).images[0]
    baseline_time = time.time() - start_time
    print(f"Baseline time: {baseline_time:.3f}s")
    
    # Apply torch.compile
    print("\n2. Applying torch.compile...")
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")
    
    # Warmup run for compilation
    print("3. Warmup run (compilation)...")
    start_time = time.time()
    _ = pipe(
        prompt,
        num_inference_steps=4,
        height=1024,
        width=1024,
    ).images[0]
    compile_time = time.time() - start_time
    print(f"Compile + first run time: {compile_time:.3f}s")
    
    # Optimized test
    print("4. Optimized test (torch.compile)...")
    start_time = time.time()
    image_optimized = pipe(
        prompt,
        num_inference_steps=4,
        height=1024,
        width=1024,
    ).images[0]
    optimized_time = time.time() - start_time
    print(f"Optimized time: {optimized_time:.3f}s")
    
    # Results
    speedup = baseline_time / optimized_time
    print(f"\nResults:")
    print(f"Baseline:    {baseline_time:.3f}s")
    print(f"Optimized:   {optimized_time:.3f}s")
    print(f"Speedup:     {speedup:.2f}x")
    print(f"Compile time: {compile_time:.3f}s")
    
    # Save images
    image_baseline.save("baseline_flux.png")
    image_optimized.save("optimized_flux.png")
    print(f"\nImages saved: baseline_flux.png, optimized_flux.png")

if __name__ == "__main__":
    main()
