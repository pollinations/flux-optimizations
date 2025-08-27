#!/usr/bin/env python3
"""
FBCache Quality Comparison Test for FLUX.1-schnell
Generates images with different thresholds for visual quality assessment
"""

import time
import torch
from diffusers import FluxPipeline
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

def main():
    print("FBCache Quality Comparison Test for FLUX.1-schnell")
    print("Generating images for visual quality assessment...")
    
    # Test parameters - using a detailed prompt to better assess quality differences
    prompt = "A detailed portrait of a wise old wizard with a long white beard, wearing a blue robe with golden stars, holding a glowing crystal orb, intricate details, fantasy art style, high resolution"
    num_inference_steps = 4
    seed = 42
    
    # Quality-focused threshold comparison
    quality_thresholds = [
        ("baseline", None, "No caching - highest quality"),
        ("conservative", 0.01, "Very conservative - near baseline quality"),
        ("recommended", 0.08, "Recommended default - good balance"),
        ("moderate", 0.15, "Moderate - some quality trade-off"),
        ("aggressive", 0.30, "Aggressive - more quality trade-off"),
        ("extreme", 0.50, "Extreme - maximum speed, lowest quality")
    ]
    
    print(f"\nLoading FLUX.1-schnell pipeline...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    results = []
    
    for i, (name, threshold, description) in enumerate(quality_thresholds):
        print(f"\n{i+1}. Testing {name} ({description})...")
        
        if threshold is not None:
            # Apply FBCache with specific threshold
            apply_cache_on_pipe(pipe, residual_diff_threshold=threshold)
            print(f"   Applied FBCache with threshold: {threshold}")
        else:
            print("   Using baseline (no FBCache)")
        
        start_time = time.time()
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            height=1024,
            width=1024,
            generator=torch.Generator("cuda").manual_seed(seed)
        ).images[0]
        exec_time = time.time() - start_time
        
        filename = f"output/quality_{name}.png"
        image.save(filename)
        
        print(f"   Time: {exec_time:.3f}s")
        print(f"   Saved: {filename}")
        
        results.append((name, threshold, exec_time, filename, description))
    
    # Summary
    print("\n" + "="*80)
    print("FBCACHE QUALITY COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Test':<12} {'Threshold':<10} {'Time (s)':<10} {'Speedup':<10} {'Description':<30}")
    print("-"*80)
    
    baseline_time = results[0][2]  # First result is baseline
    
    for name, threshold, exec_time, filename, description in results:
        speedup = baseline_time / exec_time if threshold is not None else 1.0
        thresh_str = str(threshold) if threshold is not None else "None"
        print(f"{name:<12} {thresh_str:<10} {exec_time:<10.3f} {speedup:<10.2f}x {description:<30}")
    
    print(f"\nPrompt: {prompt[:60]}...")
    print(f"Resolution: 1024x1024, Steps: {num_inference_steps}")
    print(f"Device: {torch.cuda.get_device_name()}")
    
    print(f"\n📸 QUALITY ASSESSMENT GUIDE:")
    print(f"1. Compare 'quality_baseline.png' (reference) with other images")
    print(f"2. Look for differences in:")
    print(f"   - Fine details (beard texture, robe patterns)")
    print(f"   - Color accuracy and saturation")
    print(f"   - Overall sharpness and clarity")
    print(f"   - Lighting and shadows")
    print(f"3. Recommended for production: threshold 0.01-0.08 (best quality/speed balance)")
    
    # Memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nPeak GPU memory usage: {memory_used:.2f} GB")

if __name__ == "__main__":
    main()
