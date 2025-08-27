#!/usr/bin/env python3
"""
Minimal FLUX Benchmark - Works with existing loaded model
Tests inference speed with multiple runs on different image sizes
"""

import torch
import time
import json
import gc
import os


def check_gpu_memory():
    """Check available GPU memory"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")
        return allocated < 70  # Safe if under 70GB
    return False


def simple_benchmark():
    """Simple benchmark without loading new models"""
    print("🚀 Minimal FLUX Benchmark")
    print("=" * 40)
    
    # Check if we can proceed
    if not check_gpu_memory():
        print("❌ GPU memory too high for safe testing")
        return
    
    try:
        from diffusers import FluxPipeline
        
        print("📥 Loading minimal FLUX pipeline...")
        
        # Try loading with maximum memory efficiency
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        # Enable all memory optimizations
        pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
        
        print("✅ Pipeline loaded with memory optimizations")
        
        # Simple benchmark - different image sizes
        test_configs = [
            {"size": 256, "steps": 1, "name": "Tiny"},
            {"size": 512, "steps": 2, "name": "Small"}, 
            {"size": 512, "steps": 4, "name": "Normal"}
        ]
        
        results = []
        prompt = "A mountain landscape"
        
        for config in test_configs:
            print(f"\n📊 Testing {config['name']} ({config['size']}px, {config['steps']} steps)")
            
            times = []
            for run in range(3):
                torch.cuda.empty_cache()
                gc.collect()
                
                torch.cuda.synchronize()
                start_time = time.time()
                
                image = pipe(
                    prompt,
                    height=config["size"],
                    width=config["size"],
                    num_inference_steps=config["steps"],
                    guidance_scale=0.0
                ).images[0]
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                inference_time = end_time - start_time
                times.append(inference_time)
                
                print(f"  Run {run+1}: {inference_time:.2f}s")
                
                if run == 0:
                    image.save(f"sample_{config['name'].lower()}.png")
                
                del image
                torch.cuda.empty_cache()
            
            mean_time = sum(times) / len(times)
            results.append({
                "config": config["name"],
                "size": config["size"],
                "steps": config["steps"],
                "times": times,
                "mean_time": mean_time
            })
        
        # Summary
        print("\n" + "=" * 40)
        print("📊 RESULTS")
        print("=" * 40)
        
        for result in results:
            print(f"{result['config']}: {result['mean_time']:.2f}s avg")
        
        # Save results
        with open("minimal_benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("💾 Results saved to minimal_benchmark_results.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try cleaning up GPU memory first:")
        print("   pkill -f python")
        print("   nvidia-smi")


if __name__ == "__main__":
    simple_benchmark()
