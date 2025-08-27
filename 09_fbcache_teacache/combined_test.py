#!/usr/bin/env python3
"""
Combined FBCache + TeaCache Test Script for FLUX Schnell
Tests both caching methods together for maximum acceleration.
"""

import torch
import time
import json
import os
import sys
from pathlib import Path
from diffusers import FluxPipeline
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

# Add TeaCache to path
TEACACHE_PATH = Path("TeaCache/TeaCache4FLUX")
if TEACACHE_PATH.exists():
    sys.path.insert(0, str(TEACACHE_PATH))

# Configuration
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEVICE = "cuda"
DTYPE = torch.bfloat16
OUTPUT_DIR = Path("outputs/combined")
BENCHMARK_FILE = "combined_benchmark.json"

# Test configurations
TEST_PROMPTS = [
    "A red apple on a white table",
    "A futuristic cityscape at sunset with flying cars and neon lights",
    "A photorealistic portrait of an elderly man with weathered hands holding a vintage camera"
]

# Combined optimization configurations
COMBINED_CONFIGS = [
    # Conservative combinations
    {"fbcache_threshold": 0.05, "teacache_interval": 2, "name": "conservative"},
    {"fbcache_threshold": 0.1, "teacache_interval": 2, "name": "balanced"},
    {"fbcache_threshold": 0.12, "teacache_interval": 1, "name": "aggressive"},
    {"fbcache_threshold": 0.15, "teacache_interval": 1, "name": "maximum"},
]

NUM_INFERENCE_STEPS = 4  # FLUX Schnell default
IMAGE_SIZE = 1024
NUM_WARMUP_RUNS = 2
NUM_BENCHMARK_RUNS = 3

def setup_pipeline():
    """Initialize FLUX pipeline with optimizations."""
    print("Loading FLUX.1-schnell pipeline...")
    
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    
    # Enable memory efficient attention
    pipe.enable_model_cpu_offload()
    
    return pipe

def benchmark_baseline(pipe, prompt):
    """Benchmark baseline performance without caching."""
    print(f"Benchmarking baseline for: '{prompt[:50]}...'")
    
    # Warmup runs
    for i in range(NUM_WARMUP_RUNS):
        _ = pipe(
            prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
        ).images[0]
    
    # Benchmark runs
    times = []
    for i in range(NUM_BENCHMARK_RUNS):
        torch.cuda.synchronize()
        start_time = time.time()
        
        image = pipe(
            prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
        ).images[0]
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        inference_time = end_time - start_time
        times.append(inference_time)
        print(f"  Run {i+1}: {inference_time:.3f}s")
    
    avg_time = sum(times) / len(times)
    print(f"  Average: {avg_time:.3f}s")
    
    return {
        "times": times,
        "avg_time": avg_time,
        "image": image
    }

def benchmark_fbcache_only(pipe, prompt, threshold):
    """Benchmark FBCache only."""
    print(f"Benchmarking FBCache only (threshold={threshold}) for: '{prompt[:50]}...'")
    
    # Apply FBCache
    apply_cache_on_pipe(pipe, residual_diff_threshold=threshold)
    
    # Warmup and benchmark
    times = []
    for i in range(NUM_WARMUP_RUNS + NUM_BENCHMARK_RUNS):
        torch.cuda.synchronize()
        start_time = time.time()
        
        image = pipe(
            prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
        ).images[0]
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        if i >= NUM_WARMUP_RUNS:  # Only record benchmark runs
            inference_time = end_time - start_time
            times.append(inference_time)
            print(f"  Run {i-NUM_WARMUP_RUNS+1}: {inference_time:.3f}s")
    
    avg_time = sum(times) / len(times)
    print(f"  Average: {avg_time:.3f}s")
    
    return {
        "threshold": threshold,
        "times": times,
        "avg_time": avg_time,
        "image": image
    }

def benchmark_combined(pipe, prompt, config):
    """Benchmark combined FBCache + TeaCache performance."""
    fbcache_threshold = config["fbcache_threshold"]
    teacache_interval = config["teacache_interval"]
    config_name = config["name"]
    
    print(f"Benchmarking combined ({config_name}: FBCache={fbcache_threshold}, TeaCache={teacache_interval}) for: '{prompt[:50]}...'")
    
    # Apply FBCache
    apply_cache_on_pipe(pipe, residual_diff_threshold=fbcache_threshold)
    
    # Apply TeaCache if available
    try:
        if TEACACHE_PATH.exists():
            from teacache_flux import apply_teacache
            teacache_config = {
                "cache_interval": teacache_interval,
                "cache_layer_id": 0,
                "cache_block_id": 0
            }
            apply_teacache(pipe, **teacache_config)
            print(f"  Applied both FBCache and TeaCache")
        else:
            print(f"  TeaCache not available, using FBCache only")
    except Exception as e:
        print(f"  Warning: TeaCache failed ({e}), using FBCache only")
    
    # Warmup and benchmark
    times = []
    for i in range(NUM_WARMUP_RUNS + NUM_BENCHMARK_RUNS):
        torch.cuda.synchronize()
        start_time = time.time()
        
        image = pipe(
            prompt,
            num_inference_steps=NUM_INFERENCE_STEPS,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
        ).images[0]
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        if i >= NUM_WARMUP_RUNS:  # Only record benchmark runs
            inference_time = end_time - start_time
            times.append(inference_time)
            print(f"  Run {i-NUM_WARMUP_RUNS+1}: {inference_time:.3f}s")
    
    avg_time = sum(times) / len(times)
    print(f"  Average: {avg_time:.3f}s")
    
    return {
        "config": config,
        "times": times,
        "avg_time": avg_time,
        "image": image
    }

def save_results(results, output_file):
    """Save benchmark results to JSON file."""
    # Convert images to None for JSON serialization
    json_results = {}
    for prompt, prompt_results in results.items():
        json_results[prompt] = {}
        for config_name, data in prompt_results.items():
            if data is not None:
                json_data = data.copy()
                if 'image' in json_data:
                    del json_data['image']  # Remove image for JSON
                json_results[prompt][config_name] = json_data
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {output_file}")

def save_images(results, output_dir):
    """Save generated images for quality comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for prompt_idx, (prompt, prompt_results) in enumerate(results.items()):
        prompt_name = f"prompt_{prompt_idx+1}"
        
        for config_name, data in prompt_results.items():
            if data is not None and 'image' in data and data['image'] is not None:
                if config_name == 'baseline':
                    filename = f"{prompt_name}_baseline.png"
                elif config_name == 'fbcache_only':
                    threshold = data['threshold']
                    filename = f"{prompt_name}_fbcache_only_{threshold}.png"
                else:
                    config_info = data['config']
                    config_name_str = config_info['name']
                    filename = f"{prompt_name}_combined_{config_name_str}.png"
                
                filepath = output_dir / filename
                data['image'].save(filepath)
                print(f"Saved: {filepath}")

def print_summary(results):
    """Print performance summary."""
    print("\n" + "="*90)
    print("COMBINED FBCACHE + TEACACHE PERFORMANCE SUMMARY")
    print("="*90)
    
    for prompt_idx, (prompt, prompt_results) in enumerate(results.items()):
        print(f"\nPrompt {prompt_idx+1}: {prompt[:60]}...")
        print("-" * 85)
        
        if 'baseline' not in prompt_results:
            continue
            
        baseline_time = prompt_results['baseline']['avg_time']
        print(f"{'Configuration':<20} {'Avg Time (s)':<15} {'Speedup':<10} {'FBCache':<10} {'TeaCache':<10}")
        print("-" * 85)
        print(f"{'Baseline':<20} {baseline_time:<15.3f} {'1.00x':<10} {'-':<10} {'-':<10}")
        
        # FBCache only
        if 'fbcache_only' in prompt_results and prompt_results['fbcache_only'] is not None:
            data = prompt_results['fbcache_only']
            avg_time = data['avg_time']
            speedup = baseline_time / avg_time
            threshold = data['threshold']
            print(f"{'FBCache Only':<20} {avg_time:<15.3f} {speedup:<10.2f}x {threshold:<10} {'-':<10}")
        
        # Combined configurations
        for config_name, data in prompt_results.items():
            if config_name.startswith('combined_') and data is not None:
                avg_time = data['avg_time']
                speedup = baseline_time / avg_time
                config_info = data['config']
                fb_threshold = config_info['fbcache_threshold']
                tea_interval = config_info['teacache_interval']
                name = config_info['name']
                print(f"{'Combined (' + name + ')':<20} {avg_time:<15.3f} {speedup:<10.2f}x {fb_threshold:<10} {tea_interval:<10}")

def main():
    """Main execution function."""
    print("Starting Combined FBCache + TeaCache optimization test for FLUX Schnell")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_ID}")
    print(f"Test configurations: {len(COMBINED_CONFIGS)} combined configs")
    
    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    results = {}
    
    # Test each prompt
    for prompt in TEST_PROMPTS:
        print(f"\n{'='*60}")
        print(f"Testing prompt: {prompt}")
        print('='*60)
        
        results[prompt] = {}
        
        # Baseline test
        pipe = setup_pipeline()
        baseline_result = benchmark_baseline(pipe, prompt)
        results[prompt]['baseline'] = baseline_result
        
        # FBCache only test (using best threshold from individual tests)
        pipe = setup_pipeline()
        fbcache_result = benchmark_fbcache_only(pipe, prompt, 0.12)  # Balanced threshold
        results[prompt]['fbcache_only'] = fbcache_result
        
        # Combined tests
        for i, config in enumerate(COMBINED_CONFIGS):
            pipe = setup_pipeline()
            combined_result = benchmark_combined(pipe, prompt, config)
            results[prompt][f'combined_{config["name"]}'] = combined_result
    
    # Save results
    save_results(results, BENCHMARK_FILE)
    save_images(results, OUTPUT_DIR)
    
    # Print summary
    print_summary(results)
    
    print(f"\nTest completed! Check {OUTPUT_DIR} for generated images.")
    print(f"Benchmark data saved to {BENCHMARK_FILE}")

if __name__ == "__main__":
    main()
