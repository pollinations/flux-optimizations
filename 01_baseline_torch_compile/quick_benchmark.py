#!/usr/bin/env python3
"""
Quick FLUX Schnell Benchmark - Memory Conservative
Simple performance test with multiple runs
"""

import torch
import time
import json
import os
from diffusers import FluxPipeline
import gc


def setup_environment():
    """Setup cache and config"""
    cache_dir = "/home/ionet_baremetal/torch_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    
    # Minimal inductor config
    config = torch._inductor.config
    config.conv_1x1_as_mm = True
    config.disable_progress = True
    print("✅ Environment configured")


def load_and_optimize_pipeline():
    """Load FLUX pipeline with basic optimizations"""
    print("📥 Loading FLUX Schnell...")
    
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
        variant="fp16"
    )
    pipe.to("cuda")
    
    # Enable CPU offload to save memory
    pipe.enable_model_cpu_offload()
    
    print("✅ Pipeline loaded with CPU offload")
    return pipe


def benchmark_runs(pipe, num_runs=5):
    """Run multiple inference tests"""
    print(f"📊 Running {num_runs} benchmark tests...")
    
    prompt = "A beautiful mountain landscape, highly detailed"
    times = []
    
    for i in range(num_runs):
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Measure inference time
        torch.cuda.synchronize()
        start_time = time.time()
        
        image = pipe(
            prompt,
            output_type="pil",
            num_inference_steps=4,
            height=512,
            width=512,
            guidance_scale=0.0  # Schnell doesn't need guidance
        ).images[0]
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        inference_time = end_time - start_time
        times.append(inference_time)
        
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Run {i+1}: {inference_time:.2f}s, {memory_used:.1f}GB")
        
        # Save first image
        if i == 0:
            image.save("sample_output.png")
            print("💾 Sample saved as sample_output.png")
        
        # Clean up
        del image
        torch.cuda.empty_cache()
    
    return times


def main():
    print("🚀 Quick FLUX Schnell Benchmark")
    print("=" * 40)
    
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    
    # Setup
    setup_environment()
    
    # Load pipeline
    pipe = load_and_optimize_pipeline()
    
    # Benchmark
    times = benchmark_runs(pipe)
    
    # Calculate statistics
    mean_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    
    # Results
    print("\n" + "=" * 40)
    print("📊 RESULTS")
    print("=" * 40)
    print(f"Mean time: {mean_time:.2f}s ± {std_time:.2f}s")
    print(f"Range: {min_time:.2f}s - {max_time:.2f}s")
    print(f"Target: 6.7s (baseline)")
    
    speedup = 6.7 / mean_time if mean_time > 0 else 0
    print(f"Speedup vs target: {speedup:.1f}x")
    
    # Save results
    results = {
        "times": times,
        "mean_time": mean_time,
        "std_time": std_time,
        "min_time": min_time,
        "max_time": max_time,
        "speedup_vs_target": speedup,
        "gpu": torch.cuda.get_device_name(),
        "pytorch_version": torch.__version__
    }
    
    with open("quick_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("💾 Results saved to quick_benchmark_results.json")
    
    # Success check
    if mean_time < 6.7:
        print("✅ Target performance achieved!")
    else:
        print("⚠️ Performance below target")


if __name__ == "__main__":
    main()
