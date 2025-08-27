#!/usr/bin/env python3
"""
FLUX Schnell Baseline with torch.compile Optimization
Simple implementation based on Modal and PyTorch best practices
"""

import torch
import time
import json
import os
from pathlib import Path
from diffusers import FluxPipeline
from PIL import Image
import gc


def setup_inductor_config():
    """Configure torch inductor for H100 optimization"""
    import tempfile
    import shutil
    
    # Set cache directory to avoid /tmp space issues
    cache_dir = "/home/ionet_baremetal/torch_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    
    config = torch._inductor.config
    config.conv_1x1_as_mm = True
    config.coordinate_descent_tuning = True
    config.coordinate_descent_check_all_directions = True
    config.epilogue_fusion = False
    config.disable_progress = False  # Show progress bar
    print("✅ Inductor config set for H100 optimization")
    print(f"✅ Cache directory set to: {cache_dir}")


def optimize_pipeline(pipe, compile_model=True):
    """
    Apply optimizations to FLUX pipeline
    Based on Modal and PyTorch recommendations
    """
    print("🔧 Applying pipeline optimizations...")
    
    # Fuse QKV projections for better performance
    pipe.transformer.fuse_qkv_projections()
    pipe.vae.fuse_qkv_projections()
    
    # Switch to channels_last memory format (H100 optimized)
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)
    
    if not compile_model:
        print("✅ Basic optimizations applied (no compilation)")
        return pipe
    
    # Apply torch.compile with max-autotune for H100
    print("🔥 Compiling transformer and VAE with torch.compile...")
    pipe.transformer = torch.compile(
        pipe.transformer,
        mode="max-autotune",
        fullgraph=True
    )
    
    pipe.vae.decode = torch.compile(
        pipe.vae.decode,
        mode="max-autotune", 
        fullgraph=True
    )
    
    print("✅ torch.compile optimizations applied")
    return pipe


def warmup_pipeline(pipe, num_inference_steps=4):
    """Warmup pipeline with dummy generation to trigger compilation"""
    print("🔦 Running warmup to trigger torch compilation...")
    start_time = time.time()
    
    # Dummy generation to trigger compilation
    _ = pipe(
        "dummy prompt for compilation warmup",
        output_type="pil",
        num_inference_steps=num_inference_steps,
        height=512,
        width=512
    )
    
    compilation_time = time.time() - start_time
    print(f"🔦 Compilation completed in {compilation_time:.2f} seconds")
    return compilation_time


def benchmark_inference(pipe, prompt, num_runs=5, num_inference_steps=4):
    """Benchmark inference performance"""
    print(f"📊 Running {num_runs} benchmark iterations...")
    
    times = []
    for i in range(num_runs):
        torch.cuda.synchronize()
        start_time = time.time()
        
        image = pipe(
            prompt,
            output_type="pil",
            num_inference_steps=num_inference_steps,
            height=1024,
            width=1024
        ).images[0]
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        inference_time = end_time - start_time
        times.append(inference_time)
        print(f"  Run {i+1}: {inference_time:.3f}s")
        
        # Save first image as sample
        if i == 0:
            image.save("sample_output.png")
            print("💾 Sample image saved as sample_output.png")
    
    return times, image


def get_memory_stats():
    """Get GPU memory usage statistics"""
    if torch.cuda.is_available():
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3
        }
    return {}


def main():
    print("🚀 FLUX Schnell torch.compile Baseline Test")
    print("=" * 50)
    
    # Check prerequisites
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Setup
    setup_inductor_config()
    
    # Load pipeline
    print("📥 Loading FLUX Schnell pipeline...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    
    # Apply optimizations
    pipe = optimize_pipeline(pipe, compile_model=True)
    
    # Warmup and compilation
    compilation_time = warmup_pipeline(pipe)
    
    # Clear cache after warmup
    torch.cuda.empty_cache()
    gc.collect()
    
    # Benchmark
    prompt = "A beautiful landscape with mountains and a lake, highly detailed, 8k"
    times, sample_image = benchmark_inference(pipe, prompt)
    
    # Calculate statistics
    min_time = min(times)
    max_time = max(times)
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    
    # Memory stats
    memory_stats = get_memory_stats()
    
    # Results
    results = {
        "compilation_time_seconds": compilation_time,
        "inference_times_seconds": times,
        "min_time_seconds": min_time,
        "max_time_seconds": max_time,
        "mean_time_seconds": mean_time,
        "std_time_seconds": std_time,
        "target_time_seconds": 6.7,
        "speedup_vs_target": 6.7 / mean_time if mean_time > 0 else 0,
        "memory_stats_gb": memory_stats,
        "prompt": prompt,
        "num_inference_steps": 4,
        "image_size": "1024x1024"
    }
    
    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Compilation time: {compilation_time:.2f}s")
    print(f"Mean inference time: {mean_time:.3f}s ± {std_time:.3f}s")
    print(f"Min/Max: {min_time:.3f}s / {max_time:.3f}s")
    print(f"Target time: 6.7s")
    print(f"Speedup vs target: {results['speedup_vs_target']:.2f}x")
    print(f"GPU Memory: {memory_stats.get('allocated_gb', 0):.1f}GB allocated")
    print("💾 Results saved to benchmark_results.json")
    
    # Success criteria check
    success_criteria = {
        "compilation_under_10s": compilation_time < 10,
        "mean_under_target": mean_time < 6.7,
        "speedup_achieved": results['speedup_vs_target'] >= 1.5
    }
    
    print("\n🎯 SUCCESS CRITERIA:")
    for criterion, passed in success_criteria.items():
        status = "✅" if passed else "❌"
        print(f"{status} {criterion}: {passed}")


if __name__ == "__main__":
    main()
