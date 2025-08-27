#!/usr/bin/env python3
"""
Quick FBCache Test for FLUX.1-schnell
Simple test to verify FBCache is working with FLUX.1-schnell
"""

import torch
import time
from diffusers import FluxPipeline
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

# Configuration
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEVICE = "cuda"
DTYPE = torch.bfloat16

def main():
    print("Quick FBCache test with FLUX.1-schnell")
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
    print("\n1. Baseline test (no caching)...")
    start_time = time.time()
    image_baseline = pipe(
        prompt,
        num_inference_steps=4,
        height=1024,
        width=1024,
    ).images[0]
    baseline_time = time.time() - start_time
    print(f"Baseline time: {baseline_time:.3f}s")
    
    # Apply FBCache
    print("\n2. Applying FBCache...")
    apply_cache_on_pipe(pipe, residual_diff_threshold=0.12)
    
    # FBCache test
    print("3. FBCache test...")
    start_time = time.time()
    image_fbcache = pipe(
        prompt,
        num_inference_steps=4,
        height=1024,
        width=1024,
    ).images[0]
    fbcache_time = time.time() - start_time
    print(f"FBCache time: {fbcache_time:.3f}s")
    
    # Results
    speedup = baseline_time / fbcache_time
    print(f"\nResults:")
    print(f"Baseline: {baseline_time:.3f}s")
    print(f"FBCache:  {fbcache_time:.3f}s")
    print(f"Speedup:  {speedup:.2f}x")
    
    # Save images
    image_baseline.save("baseline_test.png")
    image_fbcache.save("fbcache_test.png")
    print(f"\nImages saved: baseline_test.png, fbcache_test.png")

if __name__ == "__main__":
    main()
