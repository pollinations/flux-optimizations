#!/usr/bin/env python3
"""
Simple FP8 Weight-Only Quantization Test with TorchAO
Uses a simple model to demonstrate the quantization approach
"""

import torch
import time
import gc
from torchao.quantization import quantize_, Float8WeightOnlyConfig


class SimpleTransformerBlock(torch.nn.Module):
    """Simple transformer block to test quantization"""
    def __init__(self, dim=1024, hidden_dim=4096):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.attn = torch.nn.MultiheadAttention(dim, num_heads=16, batch_first=True)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x):
        # Self attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        return x


class SimpleModel(torch.nn.Module):
    """Simple model with multiple transformer blocks"""
    def __init__(self, num_layers=12, dim=1024):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            SimpleTransformerBlock(dim) for _ in range(num_layers)
        ])
        self.output_proj = torch.nn.Linear(dim, dim)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)


def get_gpu_memory():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def benchmark_model(model, input_tensor, num_runs=5):
    """Benchmark model inference time"""
    times = []
    
    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = model(input_tensor)
    
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    
    # Actual benchmarking
    for i in range(num_runs):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            output = model(input_tensor)
        
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
        
        print(f"Run {i+1}: {times[-1]:.4f}s, GPU Memory: {get_gpu_memory():.2f}GB")
    
    return times, output


def main():
    print("=== TorchAO FP8 Weight-Only Quantization Test ===")
    
    # Check GPU capability
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        print(f"CUDA Compute Capability: {capability}")
        if capability[0] < 8 or (capability[0] == 8 and capability[1] < 9):
            print("WARNING: FP8 quantization requires compute capability 8.9+")
        else:
            print("✅ GPU supports FP8 quantization")
    else:
        print("ERROR: CUDA not available")
        return
    
    # Create model and test input
    print("\nCreating test model...")
    model = SimpleModel(num_layers=12, dim=1024).to(torch.bfloat16).to("cuda")
    input_tensor = torch.randn(4, 512, 1024, dtype=torch.bfloat16, device="cuda")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Initial GPU Memory: {get_gpu_memory():.2f}GB")
    
    # Baseline benchmark
    print("\n--- BASELINE (No Quantization) ---")
    baseline_times, baseline_output = benchmark_model(model, input_tensor)
    baseline_avg = sum(baseline_times) / len(baseline_times)
    baseline_memory = get_gpu_memory()
    
    print(f"Baseline Average Time: {baseline_avg:.4f}s")
    print(f"Baseline Memory Usage: {baseline_memory:.2f}GB")
    
    # Apply FP8 Weight-Only Quantization (SIMPLEST APPROACH)
    print("\n--- Applying FP8 Weight-Only Quantization ---")
    print("Using: quantize_(model, Float8WeightOnlyConfig())")
    
    # This is the one-line quantization from TorchAO docs
    quantize_(model, Float8WeightOnlyConfig())
    
    print("✅ Quantization complete!")
    print(f"Post-quantization GPU Memory: {get_gpu_memory():.2f}GB")
    
    # Compile for better performance
    print("Compiling model...")
    model = torch.compile(model, mode='max-autotune')
    
    # Quantized benchmark
    print("\n--- QUANTIZED (FP8 Weight-Only) ---")
    quantized_times, quantized_output = benchmark_model(model, input_tensor)
    quantized_avg = sum(quantized_times) / len(quantized_times)
    quantized_memory = get_gpu_memory()
    
    print(f"Quantized Average Time: {quantized_avg:.4f}s")
    print(f"Quantized Memory Usage: {quantized_memory:.2f}GB")
    
    # Verify output similarity
    output_diff = torch.mean(torch.abs(baseline_output - quantized_output)).item()
    print(f"Output difference (MAE): {output_diff:.6f}")
    
    # Results summary
    print("\n=== RESULTS SUMMARY ===")
    speedup = baseline_avg / quantized_avg
    memory_reduction = (baseline_memory - quantized_memory) / baseline_memory * 100
    
    print(f"Baseline Time: {baseline_avg:.4f}s")
    print(f"Quantized Time: {quantized_avg:.4f}s")
    print(f"Speedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
    print(f"Memory Reduction: {memory_reduction:.1f}%")
    print(f"Output Accuracy: MAE = {output_diff:.6f}")
    
    # Target comparison
    target_speedup = 1.54  # 53.88% faster
    print(f"\nTarget Speedup: {target_speedup:.2f}x (53.88% faster)")
    
    if speedup >= target_speedup:
        print("✅ TARGET ACHIEVED!")
    else:
        print(f"📊 Current progress: {speedup/target_speedup*100:.1f}% of target")
    
    # Save results
    with open("fp8_quantization_test_results.txt", "w") as f:
        f.write("TorchAO FP8 Weight-Only Quantization Test Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"GPU: {torch.cuda.get_device_name()}\n")
        f.write(f"Compute Capability: {torch.cuda.get_device_capability()}\n")
        f.write(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")
        f.write(f"Baseline Time: {baseline_avg:.4f}s\n")
        f.write(f"Quantized Time: {quantized_avg:.4f}s\n")
        f.write(f"Speedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)\n")
        f.write(f"Memory Reduction: {memory_reduction:.1f}%\n")
        f.write(f"Output Accuracy: MAE = {output_diff:.6f}\n")
        f.write(f"Target Achieved: {'Yes' if speedup >= target_speedup else 'No'}\n")
    
    print("\n📁 Results saved to fp8_quantization_test_results.txt")
    print("\n🎯 Next step: Apply this same approach to FLUX model")


if __name__ == "__main__":
    main()
