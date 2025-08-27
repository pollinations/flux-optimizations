#!/usr/bin/env python3
"""
Simple torch.compile Benchmark
Tests torch.compile performance without heavy models
"""

import torch
import time
import json
import os


def setup_inductor():
    """Setup torch inductor config"""
    cache_dir = "/home/ionet_baremetal/torch_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    
    config = torch._inductor.config
    config.conv_1x1_as_mm = True
    config.coordinate_descent_tuning = True
    config.epilogue_fusion = False
    print("✅ Inductor configured")


class SimpleModel(torch.nn.Module):
    """Simple model for testing torch.compile"""
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 3, 3, padding=1)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def benchmark_torch_compile():
    """Benchmark torch.compile with different batch sizes"""
    print("🚀 torch.compile Benchmark")
    print("=" * 40)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    setup_inductor()
    
    # Create model
    model = SimpleModel().cuda().bfloat16()
    
    # Test configurations
    configs = [
        {"batch": 1, "size": 512, "name": "Single 512px"},
        {"batch": 2, "size": 512, "name": "Batch 2 512px"},
        {"batch": 1, "size": 1024, "name": "Single 1024px"},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n📊 Testing {config['name']}")
        
        batch_size = config["batch"]
        img_size = config["size"]
        
        # Create test input
        x = torch.randn(batch_size, 3, img_size, img_size, 
                       device="cuda", dtype=torch.bfloat16)
        
        # Test eager mode
        model_eager = model
        times_eager = []
        
        for run in range(5):
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                _ = model_eager(x)
            torch.cuda.synchronize()
            times_eager.append(time.time() - start)
        
        # Test compiled mode
        model_compiled = torch.compile(model, mode="max-autotune")
        
        # Warmup
        print("  Compiling...")
        with torch.no_grad():
            _ = model_compiled(x)
        
        times_compiled = []
        for run in range(5):
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                _ = model_compiled(x)
            torch.cuda.synchronize()
            times_compiled.append(time.time() - start)
        
        # Calculate results
        eager_mean = sum(times_eager) / len(times_eager)
        compiled_mean = sum(times_compiled) / len(times_compiled)
        speedup = eager_mean / compiled_mean
        
        result = {
            "config": config["name"],
            "batch_size": batch_size,
            "image_size": img_size,
            "eager_time": eager_mean,
            "compiled_time": compiled_mean,
            "speedup": speedup
        }
        results.append(result)
        
        print(f"  Eager: {eager_mean*1000:.1f}ms")
        print(f"  Compiled: {compiled_mean*1000:.1f}ms")
        print(f"  Speedup: {speedup:.1f}x")
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 SUMMARY")
    print("=" * 40)
    
    for result in results:
        print(f"{result['config']}: {result['speedup']:.1f}x speedup")
    
    # Save results
    with open("torch_compile_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n💾 Results saved to torch_compile_benchmark.json")
    print("✅ torch.compile working correctly!")


if __name__ == "__main__":
    benchmark_torch_compile()
