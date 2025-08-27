#!/usr/bin/env python3
"""
FLUX Schnell FP8 Weight-Only Quantization with Authentication
Handles Hugging Face authentication programmatically
"""

import torch
import time
import gc
import os
from diffusers import FluxPipeline
from torchao.quantization import quantize_, Float8WeightOnlyConfig
from huggingface_hub import login


def authenticate_huggingface():
    """Handle Hugging Face authentication with multiple methods"""
    print("=== Hugging Face Authentication ===")
    
    # Method 1: Check environment variable
    if "HF_TOKEN" in os.environ:
        print("✅ Using HF_TOKEN environment variable")
        return True
    
    # Method 2: Check if already logged in
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Already logged in as: {user_info['name']}")
        return True
    except Exception:
        pass
    
    # Method 3: Interactive login
    print("🔐 Please provide your Hugging Face token")
    print("Get your token from: https://huggingface.co/settings/tokens")
    print("Make sure you have access to FLUX.1-schnell model")
    
    try:
        login()
        print("✅ Authentication successful!")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False


def get_gpu_memory():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0


def benchmark_inference(pipe, prompt, num_runs=3, num_inference_steps=4):
    """Benchmark inference time and memory usage"""
    times = []
    
    # Warmup
    print("Warming up...")
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
    print(f"Running {num_runs} benchmark iterations...")
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
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Compute Capability: {capability}")
        if capability[0] < 8 or (capability[0] == 8 and capability[1] < 9):
            print("WARNING: FP8 quantization requires compute capability 8.9+")
        else:
            print("✅ GPU supports FP8 quantization")
    else:
        print("ERROR: CUDA not available")
        return
    
    # Authenticate with Hugging Face
    if not authenticate_huggingface():
        print("❌ Cannot proceed without authentication")
        return
    
    # Load FLUX model
    print("\n=== Loading FLUX Schnell Model ===")
    try:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16
        )
        pipe = pipe.to("cuda")
        print("✅ FLUX model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load FLUX model: {e}")
        return
    
    print(f"Initial GPU Memory: {get_gpu_memory():.2f}GB")
    
    # Test prompt
    prompt = "A beautiful sunset over mountains with vibrant colors, photorealistic"
    
    # Baseline benchmark
    print("\n=== BASELINE (No Quantization) ===")
    baseline_times, baseline_image = benchmark_inference(pipe, prompt)
    baseline_avg = sum(baseline_times) / len(baseline_times)
    baseline_memory = get_gpu_memory()
    
    print(f"Baseline Average Time: {baseline_avg:.3f}s")
    print(f"Baseline Memory Usage: {baseline_memory:.2f}GB")
    
    # Save baseline image
    baseline_image.save("baseline_flux_output.png")
    print("💾 Saved baseline_flux_output.png")
    
    # Apply FP8 Weight-Only Quantization (SIMPLEST APPROACH)
    print("\n=== Applying FP8 Weight-Only Quantization ===")
    print("Using: quantize_(model, Float8WeightOnlyConfig())")
    
    # Quantize the transformer (main compute component)
    print("Quantizing transformer...")
    quantize_(pipe.transformer, Float8WeightOnlyConfig())
    
    # Optional: Also quantize text encoders for additional memory savings
    print("Quantizing text encoders...")
    if hasattr(pipe, 'text_encoder'):
        quantize_(pipe.text_encoder, Float8WeightOnlyConfig())
    if hasattr(pipe, 'text_encoder_2'):
        quantize_(pipe.text_encoder_2, Float8WeightOnlyConfig())
    
    print("✅ Quantization complete!")
    print(f"Post-quantization GPU Memory: {get_gpu_memory():.2f}GB")
    
    # Compile for better performance
    print("Compiling model for optimization...")
    pipe.transformer = torch.compile(pipe.transformer, mode='max-autotune')
    
    # Quantized benchmark
    print("\n=== QUANTIZED (FP8 Weight-Only) ===")
    quantized_times, quantized_image = benchmark_inference(pipe, prompt)
    quantized_avg = sum(quantized_times) / len(quantized_times)
    quantized_memory = get_gpu_memory()
    
    print(f"Quantized Average Time: {quantized_avg:.3f}s")
    print(f"Quantized Memory Usage: {quantized_memory:.2f}GB")
    
    # Save quantized image
    quantized_image.save("quantized_flux_fp8_output.png")
    print("💾 Saved quantized_flux_fp8_output.png")
    
    # Results summary
    print("\n=== RESULTS SUMMARY ===")
    speedup = baseline_avg / quantized_avg
    memory_reduction = (baseline_memory - quantized_memory) / baseline_memory * 100
    target_speedup = 1.54  # 53.88% faster
    
    print(f"🚀 Performance Results:")
    print(f"   Baseline Time: {baseline_avg:.3f}s")
    print(f"   Quantized Time: {quantized_avg:.3f}s")
    print(f"   Speedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
    
    print(f"💾 Memory Results:")
    print(f"   Memory Reduction: {memory_reduction:.1f}%")
    
    print(f"🎯 Target Analysis:")
    print(f"   Target Speedup: {target_speedup:.2f}x (53.88% faster)")
    
    if speedup >= target_speedup:
        print("   ✅ TARGET ACHIEVED!")
    else:
        progress = speedup / target_speedup * 100
        print(f"   📊 Progress: {progress:.1f}% of target ({target_speedup/speedup:.2f}x more needed)")
    
    # Save detailed results
    with open("flux_fp8_weight_only_results.txt", "w") as f:
        f.write("FLUX Schnell FP8 Weight-Only Quantization Results\n")
        f.write("=" * 55 + "\n")
        f.write(f"GPU: {torch.cuda.get_device_name()}\n")
        f.write(f"Compute Capability: {torch.cuda.get_device_capability()}\n")
        f.write(f"Model: black-forest-labs/FLUX.1-schnell\n")
        f.write(f"Quantization: Float8WeightOnlyConfig()\n")
        f.write(f"Image Size: 1024x1024\n")
        f.write(f"Inference Steps: 4\n")
        f.write("\nPerformance Results:\n")
        f.write(f"Baseline Time: {baseline_avg:.3f}s\n")
        f.write(f"Quantized Time: {quantized_avg:.3f}s\n")
        f.write(f"Speedup: {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)\n")
        f.write(f"Memory Reduction: {memory_reduction:.1f}%\n")
        f.write(f"Target Speedup: {target_speedup:.2f}x (53.88% faster)\n")
        f.write(f"Target Achieved: {'Yes' if speedup >= target_speedup else 'No'}\n")
        f.write(f"\nDetailed Times (seconds):\n")
        f.write(f"Baseline runs: {', '.join([f'{t:.3f}' for t in baseline_times])}\n")
        f.write(f"Quantized runs: {', '.join([f'{t:.3f}' for t in quantized_times])}\n")
    
    print("\n📁 Detailed results saved to flux_fp8_weight_only_results.txt")
    print("\n🎉 FLUX FP8 Weight-Only Quantization Test Complete!")


if __name__ == "__main__":
    main()
