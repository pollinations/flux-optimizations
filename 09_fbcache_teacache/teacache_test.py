#!/usr/bin/env python3
"""
TeaCache Test Script for FLUX Schnell
Tests TeaCache integration with FLUX and benchmarks performance.
"""

import torch
import time
import json
import os
import sys
from pathlib import Path
from diffusers import FluxPipeline

# Add TeaCache to path
TEACACHE_PATH = Path("TeaCache/TeaCache4FLUX")
if TEACACHE_PATH.exists():
    sys.path.insert(0, str(TEACACHE_PATH))

try:
    from teacache_flux import apply_teacache
except ImportError:
    print("TeaCache not found. Please run: git clone https://github.com/ali-vilab/TeaCache.git")
    sys.exit(1)

# Configuration
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEVICE = "cuda"
DTYPE = torch.bfloat16
OUTPUT_DIR = Path("outputs/teacache")
BENCHMARK_FILE = "teacache_benchmark.json"

# Test configurations
TEST_PROMPTS = [
    "A red apple on a white table",
    "A futuristic cityscape at sunset with flying cars and neon lights",
    "A photorealistic portrait of an elderly man with weathered hands holding a vintage camera"
]

# TeaCache configurations to test
TEACACHE_CONFIGS = [
    {"cache_interval": 1, "cache_layer_id": 0, "cache_block_id": 0},
    {"cache_interval": 2, "cache_layer_id": 0, "cache_block_id": 0},
    {"cache_interval": 3, "cache_layer_id": 0, "cache_block_id": 0},
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

def benchmark_teacache(pipe, prompt, config):
    """Benchmark TeaCache performance with given configuration."""
    cache_interval = config["cache_interval"]
    print(f"Benchmarking TeaCache (interval={cache_interval}) for: '{prompt[:50]}...'")
    
    # Apply TeaCache with specified configuration
    try:
        apply_teacache(pipe, **config)
    except Exception as e:
        print(f"Error applying TeaCache: {e}")
        return None
    
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
                else:
                    cache_interval = data['config']['cache_interval']
                    filename = f"{prompt_name}_teacache_interval_{cache_interval}.png"
                
                filepath = output_dir / filename
                data['image'].save(filepath)
                print(f"Saved: {filepath}")

def print_summary(results):
    """Print performance summary."""
    print("\n" + "="*80)
    print("TEACACHE PERFORMANCE SUMMARY")
    print("="*80)
    
    for prompt_idx, (prompt, prompt_results) in enumerate(results.items()):
        print(f"\nPrompt {prompt_idx+1}: {prompt[:60]}...")
        print("-" * 70)
        
        if 'baseline' not in prompt_results:
            continue
            
        baseline_time = prompt_results['baseline']['avg_time']
        print(f"{'Configuration':<25} {'Avg Time (s)':<15} {'Speedup':<10} {'Cache Interval':<15}")
        print("-" * 70)
        print(f"{'Baseline':<25} {baseline_time:<15.3f} {'1.00x':<10} {'-':<15}")
        
        for config_name, data in prompt_results.items():
            if config_name != 'baseline' and data is not None:
                avg_time = data['avg_time']
                speedup = baseline_time / avg_time
                cache_interval = data['config']['cache_interval']
                print(f"{'TeaCache':<25} {avg_time:<15.3f} {speedup:<10.2f}x {cache_interval:<15}")

def main():
    """Main execution function."""
    print("Starting TeaCache optimization test for FLUX Schnell")
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_ID}")
    print(f"Test configurations: {len(TEACACHE_CONFIGS)} configs")
    
    # Check if TeaCache is available
    if not TEACACHE_PATH.exists():
        print("TeaCache repository not found. Please run:")
        print("git clone https://github.com/ali-vilab/TeaCache.git")
        return
    
    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipe = setup_pipeline()
    
    # Results storage
    results = {}
    
    # Test each prompt
    for prompt in TEST_PROMPTS:
        print(f"\n{'='*60}")
        print(f"Testing prompt: {prompt}")
        print('='*60)
        
        results[prompt] = {}
        
        # Baseline test
        baseline_result = benchmark_baseline(pipe, prompt)
        results[prompt]['baseline'] = baseline_result
        
        # TeaCache tests with different configurations
        for i, config in enumerate(TEACACHE_CONFIGS):
            # Reload pipeline to reset cache state
            pipe = setup_pipeline()
            
            teacache_result = benchmark_teacache(pipe, prompt, config)
            results[prompt][f'teacache_config_{i}'] = teacache_result
    
    # Save results
    save_results(results, BENCHMARK_FILE)
    save_images(results, OUTPUT_DIR)
    
    # Print summary
    print_summary(results)
    
    print(f"\nTest completed! Check {OUTPUT_DIR} for generated images.")
    print(f"Benchmark data saved to {BENCHMARK_FILE}")

if __name__ == "__main__":
    main()
