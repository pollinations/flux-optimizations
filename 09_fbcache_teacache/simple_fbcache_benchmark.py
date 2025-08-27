#!/usr/bin/env python3
"""
Simple FBCache Benchmark for FLUX.1-schnell
Tests multiple FBCache thresholds with a single pipeline instance
"""

import time
import torch
from diffusers import FluxPipeline
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

def main():
    print("Simple FBCache Benchmark for FLUX.1-schnell")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Test parameters
    prompt = "A futuristic city skyline at sunset with flying cars"
    num_inference_steps = 4  # FLUX.1-schnell uses 4 steps
    seed = 42
    
    # FBCache thresholds to test
    thresholds = [0.08, 0.10, 0.12, 0.15, 0.20]
    
    print("\nLoading FLUX.1-schnell pipeline...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    results = []
    
    # 1. Baseline test (no FBCache)
    print("\n1. Baseline test (no FBCache)...")
    start_time = time.time()
    image_baseline = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        height=1024,
        width=1024,
        generator=torch.Generator("cuda").manual_seed(seed)
    ).images[0]
    baseline_time = time.time() - start_time
    
    image_baseline.save("output/simple_baseline.png")
    print(f"Baseline time: {baseline_time:.3f}s")
    results.append(("Baseline", baseline_time, 1.0, "output/simple_baseline.png"))
    
    # 2. FBCache tests with different thresholds
    for i, threshold in enumerate(thresholds):
        print(f"\n{i+2}. FBCache test (threshold={threshold})...")
        
        # Apply FBCache
        apply_cache_on_pipe(pipe, residual_diff_threshold=threshold)
        
        start_time = time.time()
        image_fbcache = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            height=1024,
            width=1024,
            generator=torch.Generator("cuda").manual_seed(seed)
        ).images[0]
        fbcache_time = time.time() - start_time
        
        speedup = baseline_time / fbcache_time
        filename = f"output/simple_fbcache_{threshold}.png"
        image_fbcache.save(filename)
        
        print(f"FBCache time: {fbcache_time:.3f}s")
        print(f"Speedup: {speedup:.2f}x")
        results.append((f"FBCache {threshold}", fbcache_time, speedup, filename))
    
    # 3. Performance summary
    print("\n" + "="*60)
    print("FBCACHE PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Configuration':<20} {'Time (s)':<10} {'Speedup':<10} {'Image':<25}")
    print("-"*60)
    
    for config, exec_time, speedup, filename in results:
        print(f"{config:<20} {exec_time:<10.3f} {speedup:<10.2f}x {filename:<25}")
    
    print(f"\nPrompt: {prompt}")
    print(f"Steps: {num_inference_steps}")
    print(f"Resolution: 1024x1024")
    print(f"Model: FLUX.1-schnell")
    print(f"Device: {torch.cuda.get_device_name()}")
    
    # Calculate best speedup
    best_speedup = max(results[1:], key=lambda x: x[2])
    print(f"\nBest FBCache speedup: {best_speedup[2]:.2f}x with threshold {best_speedup[0].split()[-1]}")
    
    # Memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory usage: {memory_used:.2f} GB")

if __name__ == "__main__":
    main()
