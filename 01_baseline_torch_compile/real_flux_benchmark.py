#!/usr/bin/env python3
"""
Real FLUX Schnell Image Generation Benchmark
Measures actual image generation times and saves images for inspection
"""

import torch
import time
import json
import os
from diffusers import FluxPipeline
import gc


def setup_environment():
    """Setup cache and minimal config"""
    cache_dir = "/home/ionet_baremetal/torch_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    print("✅ Environment configured")


def load_flux_pipeline():
    """Load FLUX Schnell with memory optimizations"""
    print("📥 Loading FLUX Schnell pipeline...")
    
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    # Enable memory optimizations
    pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
    
    print("✅ FLUX pipeline loaded with memory optimizations")
    return pipe


def benchmark_flux_generation(pipe, num_runs=3):
    """Benchmark actual FLUX image generation"""
    print(f"📊 Running {num_runs} FLUX image generation tests...")
    
    prompts = [
        "A beautiful mountain landscape at sunset, highly detailed",
        "A futuristic city with flying cars, cyberpunk style",
        "A peaceful forest with a crystal clear lake"
    ]
    
    results = []
    
    for i in range(num_runs):
        prompt = prompts[i % len(prompts)]
        print(f"\nRun {i+1}: '{prompt[:50]}...'")
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Reset memory tracking
        torch.cuda.reset_peak_memory_stats()
        
        # Time the generation
        torch.cuda.synchronize()
        start_time = time.time()
        
        image = pipe(
            prompt,
            output_type="pil",
            num_inference_steps=4,  # Schnell uses 4 steps
            height=1024,
            width=1024,
            guidance_scale=0.0  # Schnell doesn't use guidance
        ).images[0]
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Record results
        inference_time = end_time - start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        
        # Save image with detailed filename
        filename = f"flux_run_{i+1}_{inference_time:.1f}s.png"
        image.save(filename)
        
        result = {
            "run": i + 1,
            "prompt": prompt,
            "inference_time": inference_time,
            "peak_memory_gb": peak_memory,
            "filename": filename
        }
        results.append(result)
        
        print(f"  Time: {inference_time:.2f}s")
        print(f"  Memory: {peak_memory:.1f}GB")
        print(f"  Saved: {filename}")
        
        # Clean up
        del image
        torch.cuda.empty_cache()
    
    return results


def main():
    print("🚀 Real FLUX Schnell Benchmark")
    print("=" * 50)
    
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Setup
    setup_environment()
    
    try:
        # Load pipeline
        pipe = load_flux_pipeline()
        
        # Benchmark
        results = benchmark_flux_generation(pipe)
        
        # Calculate statistics
        times = [r["inference_time"] for r in results]
        mean_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 REAL FLUX BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Mean generation time: {mean_time:.2f}s")
        print(f"Range: {min_time:.2f}s - {max_time:.2f}s")
        print(f"Target time: 6.7s")
        
        speedup = 6.7 / mean_time if mean_time > 0 else 0
        print(f"Performance vs target: {speedup:.1f}x")
        
        print("\nGenerated images:")
        for result in results:
            print(f"  {result['filename']} - {result['inference_time']:.2f}s")
        
        # Save detailed results
        final_results = {
            "benchmark_type": "Real FLUX Schnell Image Generation",
            "mean_time": mean_time,
            "min_time": min_time,
            "max_time": max_time,
            "target_time": 6.7,
            "speedup_vs_target": speedup,
            "individual_results": results,
            "gpu": torch.cuda.get_device_name()
        }
        
        with open("real_flux_benchmark_results.json", "w") as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n💾 Results saved to real_flux_benchmark_results.json")
        
        if mean_time <= 6.7:
            print("✅ Target performance achieved!")
        else:
            print("⚠️ Performance below target - consider torch.compile optimizations")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 This may be due to memory constraints")


if __name__ == "__main__":
    main() Schnell Image Generation Benchmark
Measures actual image generation times and saves images for inspection
"""

import torch
import time
import json
import os
from diffusers import FluxPipeline
import gc


def setup_environment():
    """Setup cache and minimal config"""
    cache_dir = "/home/ionet_baremetal/torch_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    print("✅ Environment configured")


def load_flux_pipeline():
    """Load FLUX Schnell with memory optimizations"""
    print("📥 Loading FLUX Schnell pipeline...")
    
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    # Enable memory optimizations
    pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
    
    print("✅ FLUX pipeline loaded with memory optimizations")
    return pipe


def benchmark_flux_generation(pipe, num_runs=3):
    """Benchmark actual FLUX image generation"""
    print(f"📊 Running {num_runs} FLUX image generation tests...")
    
    prompts = [
        "A beautiful mountain landscape at sunset, highly detailed",
        "A futuristic city with flying cars, cyberpunk style",
        "A peaceful forest with a crystal clear lake"
    ]
    
    results = []
    
    for i in range(num_runs):
        prompt = prompts[i % len(prompts)]
        print(f"\nRun {i+1}: '{prompt[:50]}...'")
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Reset memory tracking
        torch.cuda.reset_peak_memory_stats()
        
        # Time the generation
        torch.cuda.synchronize()
        start_time = time.time()
        
        image = pipe(
            prompt,
            output_type="pil",
            num_inference_steps=4,  # Schnell uses 4 steps
            height=1024,
            width=1024,
            guidance_scale=0.0  # Schnell doesn't use guidance
        ).images[0]
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Record results
        inference_time = end_time - start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        
        # Save image with detailed filename
        filename = f"flux_run_{i+1}_{inference_time:.1f}s.png"
        image.save(filename)
        
        result = {
            "run": i + 1,
            "prompt": prompt,
            "inference_time": inference_time,
            "peak_memory_gb": peak_memory,
            "filename": filename
        }
        results.append(result)
        
        print(f"  Time: {inference_time:.2f}s")
        print(f"  Memory: {peak_memory:.1f}GB")
        print(f"  Saved: {filename}")
        
        # Clean up
        del image
        torch.cuda.empty_cache()
    
    return results


def main():
    print("🚀 Real FLUX Schnell Benchmark")
    print("=" * 50)
    
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Setup
    setup_environment()
    
    try:
        # Load pipeline
        pipe = load_flux_pipeline()
        
        # Benchmark
        results = benchmark_flux_generation(pipe)
        
        # Calculate statistics
        times = [r["inference_time"] for r in results]
        mean_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 REAL FLUX BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Mean generation time: {mean_time:.2f}s")
        print(f"Range: {min_time:.2f}s - {max_time:.2f}s")
        print(f"Target time: 6.7s")
        
        speedup = 6.7 / mean_time if mean_time > 0 else 0
        print(f"Performance vs target: {speedup:.1f}x")
        
        print("\nGenerated images:")
        for result in results:
            print(f"  {result['filename']} - {result['inference_time']:.2f}s")
        
        # Save detailed results
        final_results = {
            "benchmark_type": "Real FLUX Schnell Image Generation",
            "mean_time": mean_time,
            "min_time": min_time,
            "max_time": max_time,
            "target_time": 6.7,
            "speedup_vs_target": speedup,
            "individual_results": results,
            "gpu": torch.cuda.get_device_name()
        }
        
        with open("real_flux_benchmark_results.json", "w") as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n💾 Results saved to real_flux_benchmark_results.json")
        
        if mean_time <= 6.7:
            print("✅ Target performance achieved!")
        else:
            print("⚠️ Performance below target - consider torch.compile optimizations")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 This may be due to memory constraints")


if __name__ == "__main__":
    main()
