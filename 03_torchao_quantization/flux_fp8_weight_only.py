#!/usr/bin/env python3
"""
FLUX Schnell FP8 Weight-Only Quantization with TorchAO
Simplest approach using Float8WeightOnlyConfig
"""

import torch
import time
import gc
import psutil
import os
from diffusers import FluxPipeline
from torchao.quantization import quantize_, Float8WeightOnlyConfig


def get_gpu_memory():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def benchmark_inference(pipe, prompt, num_runs=3, num_inference_steps=4):
    """Benchmark inference time and memory usage"""
    times = []
    
    # Warmup
    with torch.no_grad():
        _ = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=num_inference_steps,
            max_sequence_length=256,
            guidance_scale=0.0
        )
    
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    
    # Actual benchmarking
    for i in range(num_runs):
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                num_inference_steps=num_inference_steps,
                max_sequence_length=256,
                guidance_scale=0.0
            ).images[0]
        
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
        
        print(f"Run {i+1}: {times[-1]:.3f}s, GPU Memory: {get_gpu_memory():.2f}GB")
    
    return times, image


def main():
    print("=== FLUX Schnell FP8 Weight-Only Quantization Test ===")
    
    # Check GPU capability
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        print(f"CUDA Compute Capability: {capability}")
        if capability[0] < 8 or (capability[0] == 8 and capability[1] < 9):
            print("WARNING: FP8 quantization requires compute capability 8.9+")
            print("This may not work optimally on your GPU")
    else:
        print("ERROR: CUDA not available")
        return
    
    # Load model
    print("\nLoading FLUX Schnell model...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16
    )
    pipe = pipe.to("cuda")
    
    print(f"Initial GPU Memory: {get_gpu_memory():.2f}GB")
    
    # Test prompt
    prompt = "A beautiful sunset over mountains with vibrant colors"
    
    # Baseline benchmark
    print("\n--- BASELINE (No Quantization) ---")
    baseline_times, baseline_image = benchmark_inference(pipe, prompt)
    baseline_avg = sum(baseline_times) / len(baseline_times)
    baseline_memory = get_gpu_memory()
    
    print(f"Baseline Average Time: {baseline_avg:.3f}s")
    print(f"Baseline Memory Usage: {baseline_memory:.2f}GB")
    
    # Save baseline image
    baseline_image.save("baseline_output.png")
    print("Saved baseline_output.png")
    
    # Apply FP8 Weight-Only Quantization
    print("\n--- Applying FP8 Weight-Only Quantization ---")
    
    # Quantize the transformer (main compute component)
    print("Quantizing transformer...")
    quantize_(pipe.transformer, Float8WeightOnlyConfig())
    
    # Optional: Also quantize text encoders if memory is a concern
    print("Quantizing text encoders...")
    if hasattr(pipe, 'text_encoder'):
        quantize_(pipe.text_encoder, Float8WeightOnlyConfig())
    if hasattr(pipe, 'text_encoder_2'):
        quantize_(pipe.text_encoder_2, Float8WeightOnlyConfig())
    
    print("Quantization complete!")
    print(f"Post-quantization GPU Memory: {get_gpu_memory():.2f}GB")
    
    # Compile for better performance
    print("Compiling model...")
    pipe.transformer = torch.compile(pipe.transformer, mode='max-autotune')
    
    # Quantized benchmark
    print("\n--- QUANTIZED (FP8 Weight-Only) ---")
    quantized_times, quantized_image = benchmark_inference(pipe, prompt)
    quantized_avg = sum(quantized_times) / len(quantized_times)
    quantized_memory = get_gpu_memory()
    
    print(f"Quantized Average Time: {quantized_avg:.3f}s")
    print(f"Quantized Memory Usage: {quantized_memory:.2f}GB")
    
    # Save quantized image
    quantized_image.save("quantized_fp8_output.png")
    print("Saved quantized_fp8_output.png")
    
    # Results summary
    print("\n=== RESULTS SUMMARY ===")
    speedup = baseline_avg / quantized_avg
    memory_reduction = (baseline_memory - quantized_memory) / baseline_memory * 100
    
    print(f"Baseline Time: {baseline_avg:.3f}s")
    print(f"Quantized Time: {quantized_avg:.3f}s")
    print(f"Speedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
    print(f"Memory Reduction: {memory_reduction:.1f}%")
    print(f"Target Speedup: 1.54x (53.88% faster)")
    
    if speedup >= 1.54:
        print("✅ TARGET ACHIEVED!")
    else:
        print(f"❌ Target not met. Need {1.54/speedup:.2f}x more improvement")
    
    # Save results to file
    with open("fp8_weight_only_results.txt", "w") as f:
        f.write("FLUX Schnell FP8 Weight-Only Quantization Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Baseline Time: {baseline_avg:.3f}s\n")
        f.write(f"Quantized Time: {quantized_avg:.3f}s\n")
        f.write(f"Speedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)\n")
        f.write(f"Memory Reduction: {memory_reduction:.1f}%\n")
        f.write(f"Target Speedup: 1.54x (53.88% faster)\n")
        f.write(f"Target Achieved: {'Yes' if speedup >= 1.54 else 'No'}\n")
    
    print("\nResults saved to fp8_weight_only_results.txt")


if __name__ == "__main__":
    main()
