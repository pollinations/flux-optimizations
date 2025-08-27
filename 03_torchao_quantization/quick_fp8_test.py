#!/usr/bin/env python3
"""
Quick FP8 Weight-Only Quantization Test
Minimal test to verify TorchAO FP8 quantization works on H100
"""

import torch
import time
from torchao.quantization import quantize_, Float8WeightOnlyConfig

def main():
    print("=== Quick FP8 Quantization Test ===")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    capability = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Compute Capability: {capability}")
    
    if capability[0] < 8 or (capability[0] == 8 and capability[1] < 9):
        print("❌ FP8 requires compute capability 8.9+")
        return
    
    print("✅ GPU supports FP8 quantization")
    
    # Create simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 4096),
        torch.nn.GELU(),
        torch.nn.Linear(4096, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 1024)
    ).to(torch.bfloat16).to("cuda")
    
    input_tensor = torch.randn(8, 1024, dtype=torch.bfloat16, device="cuda")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Baseline
    print("\n--- Baseline ---")
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        baseline_out = model(input_tensor)
    torch.cuda.synchronize()
    baseline_time = time.time() - start
    print(f"Baseline time: {baseline_time:.4f}s")
    
    # Apply FP8 Weight-Only Quantization
    print("\n--- Applying FP8 Quantization ---")
    quantize_(model, Float8WeightOnlyConfig())
    print("✅ Quantization applied successfully!")
    
    # Test quantized model
    print("\n--- Quantized ---")
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        quantized_out = model(input_tensor)
    torch.cuda.synchronize()
    quantized_time = time.time() - start
    print(f"Quantized time: {quantized_time:.4f}s")
    
    # Results
    speedup = baseline_time / quantized_time
    diff = torch.mean(torch.abs(baseline_out - quantized_out)).item()
    
    print(f"\n=== Results ===")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Output difference: {diff:.6f}")
    print(f"✅ FP8 Weight-Only quantization working!")

if __name__ == "__main__":
    main()
