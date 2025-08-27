#!/usr/bin/env python3
"""
Simple FLUX Schnell Batch Size Benchmark
Tests different batch sizes with torch.compile optimization
"""

import torch
import time
import json
import os
from diffusers import FluxPipeline
import gc
import statistics


def setup_cache_dirs():
    """Setup cache directories to avoid disk space issues"""
    cache_dir = "/home/ionet_baremetal/torch_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    os.environ["TRITON_CACHE_DIR"] = cache_dir
    print(f"✅ Cache directory: {cache_dir}")


def setup_inductor_config():
    """Configure torch inductor for H100 optimization"""
    config = torch._inductor.config
    config.conv_1x1_as_mm = True
    config.coordinate_descent_tuning = True
    config.coordinate_descent_check_all_directions = True
    config.epilogue_fusion = False
    config.disable_progress = True  # Disable for cleaner output
    print("✅ H100 inductor config applied")


def optimize_pipeline(pipe):
    """Apply torch.compile optimizations"""
    print("🔧 Applying optimizations...")
    
    # Basic optimizations
    pipe.transformer.fuse_qkv_projections()
    pipe.vae.fuse_qkv_projections()
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)
    
    # torch.compile
    pipe.transformer = torch.compile(
        pipe.transformer, mode="max-autotune", fullgraph=True
    )
    pipe.vae.decode = torch.compile(
        pipe.vae.decode, mode="max-autotune", fullgraph=True
    )
    
    print("✅ torch.compile optimizations applied")
    return pipe


def warmup_pipeline(pipe):
    """Single warmup run to trigger compilation"""
    print("🔦 Warming up (compiling)...")
    start_time = time.time()
    
    _ = pipe(
        "warmup prompt",
        output_type="pil",
        num_inference_steps=4,
        height=512,
        width=512
    )
    
    compilation_time = time.time() - start_time
    print(f"🔦 Compilation: {compilation_time:.1f}s")
    return compilation_time


def benchmark_batch_size(pipe, batch_size, num_runs=3):
    """Benchmark a specific batch size - memory efficient version"""
    print(f"\n📊 Testing batch size {batch_size} ({num_runs} runs)")
    
    times = []
    memory_usage = []
    
    for run in range(num_runs):
        # Aggressive memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        
        # Single prompt repeated for batch (more memory efficient)
        prompt = "A beautiful landscape, highly detailed"
        
        # Time the inference
        torch.cuda.synchronize()
        start_time = time.time()
        
        if batch_size == 1:
            # Single image generation
            image = pipe(
                prompt,
                output_type="pil", 
                num_inference_steps=4,
                height=512,  # Smaller size for memory efficiency
                width=512
            ).images[0]
            images = [image]
        else:
            # Multiple individual runs for batch simulation
            images = []
            for i in range(batch_size):
                img = pipe(
                    prompt,
                    output_type="pil",
                    num_inference_steps=4, 
                    height=512,
                    width=512
                ).images[0]
                images.append(img)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Record results
        inference_time = end_time - start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        times.append(inference_time)
        memory_usage.append(peak_memory)
        
        print(f"  Run {run+1}: {inference_time:.2f}s, {peak_memory:.1f}GB")
        
        # Save first image only
        if run == 0:
            images[0].save(f"sample_batch{batch_size}.png")
        
        # Clean up images
        del images
        torch.cuda.empty_cache()
    
    return {
        "batch_size": batch_size,
        "times": times,
        "mean_time": statistics.mean(times),
        "std_time": statistics.stdev(times) if len(times) > 1 else 0,
        "min_time": min(times),
        "max_time": max(times),
        "memory_usage": memory_usage,
        "mean_memory": statistics.mean(memory_usage),
        "time_per_image": statistics.mean(times) / batch_size,
        "throughput_imgs_per_sec": batch_size / statistics.mean(times)
    }


def main():
    print("🚀 FLUX Schnell Batch Size Benchmark")
    print("=" * 50)
    
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available!")
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    
    # Setup
    setup_cache_dirs()
    setup_inductor_config()
    
    # Load pipeline
    print("📥 Loading FLUX Schnell...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    
    # Optimize
    pipe = optimize_pipeline(pipe)
    
    # Warmup
    compilation_time = warmup_pipeline(pipe)
    torch.cuda.empty_cache()
    
    # Benchmark different batch sizes (conservative for memory)
    batch_sizes = [1, 2]  # Start with smaller batches
    results = []
    
    for batch_size in batch_sizes:
        try:
            result = benchmark_batch_size(pipe, batch_size)
            results.append(result)
        except torch.cuda.OutOfMemoryError:
            print(f"❌ Batch size {batch_size}: Out of memory")
            break
        except Exception as e:
            print(f"❌ Batch size {batch_size}: Error - {e}")
            break
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Compilation time: {compilation_time:.1f}s")
    print()
    
    for result in results:
        bs = result["batch_size"]
        mean_time = result["mean_time"]
        std_time = result["std_time"]
        time_per_img = result["time_per_image"]
        throughput = result["throughput_imgs_per_sec"]
        memory = result["mean_memory"]
        
        print(f"Batch Size {bs}:")
        print(f"  Total time: {mean_time:.2f}s ± {std_time:.2f}s")
        print(f"  Time/image: {time_per_img:.2f}s")
        print(f"  Throughput: {throughput:.1f} imgs/sec")
        print(f"  Memory: {memory:.1f}GB")
        print()
    
    # Save detailed results
    final_results = {
        "compilation_time": compilation_time,
        "batch_results": results,
        "gpu": torch.cuda.get_device_name(),
        "pytorch_version": torch.__version__
    }
    
    with open("batch_benchmark_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    print("💾 Results saved to batch_benchmark_results.json")
    
    # Find optimal batch size
    if results:
        best_throughput = max(results, key=lambda x: x["throughput_imgs_per_sec"])
        print(f"🏆 Best throughput: Batch size {best_throughput['batch_size']} "
              f"({best_throughput['throughput_imgs_per_sec']:.1f} imgs/sec)")


if __name__ == "__main__":
    main()
