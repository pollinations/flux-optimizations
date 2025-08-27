#!/usr/bin/env python3
"""
Extended FBCache Benchmark for FLUX.1-schnell
Tests wider threshold range and combined optimizations
"""

import time
import torch
from diffusers import FluxPipeline
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

def main():
    print("Extended FBCache Benchmark for FLUX.1-schnell")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Test parameters
    prompt = "A futuristic city skyline at sunset with flying cars"
    num_inference_steps = 4  # FLUX.1-schnell uses 4 steps
    seed = 42
    
    # Extended FBCache thresholds to test
    thresholds = [0.01, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    
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
    
    image_baseline.save("output/extended_baseline.png")
    print(f"Baseline time: {baseline_time:.3f}s")
    results.append(("Baseline", baseline_time, 1.0, "output/extended_baseline.png"))
    
    # 2. FBCache tests with extended threshold range
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
        filename = f"output/extended_fbcache_{threshold}.png"
        image_fbcache.save(filename)
        
        print(f"FBCache time: {fbcache_time:.3f}s")
        print(f"Speedup: {speedup:.2f}x")
        results.append((f"FBCache {threshold}", fbcache_time, speedup, filename))
    
    # 3. Performance summary
    print("\n" + "="*70)
    print("EXTENDED FBCACHE PERFORMANCE SUMMARY")
    print("="*70)
    print(f"{'Configuration':<20} {'Time (s)':<10} {'Speedup':<10} {'Image':<30}")
    print("-"*70)
    
    for config, exec_time, speedup, filename in results:
        print(f"{config:<20} {exec_time:<10.3f} {speedup:<10.2f}x {filename:<30}")
    
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
    
    # Analyze threshold performance
    print(f"\nThreshold Analysis:")
    fbcache_results = results[1:]
    low_thresh = [r for r in fbcache_results if float(r[0].split()[-1]) <= 0.12]
    high_thresh = [r for r in fbcache_results if float(r[0].split()[-1]) > 0.12]
    
    if low_thresh:
        avg_low = sum(r[2] for r in low_thresh) / len(low_thresh)
        print(f"Average speedup (threshold ≤ 0.12): {avg_low:.2f}x")
    
    if high_thresh:
        avg_high = sum(r[2] for r in high_thresh) / len(high_thresh)
        print(f"Average speedup (threshold > 0.12): {avg_high:.2f}x")

if __name__ == "__main__":
    main()
