#!/usr/bin/env python3
"""
Simple single GPU test for xDiT Flux
"""

import torch
import time
import os
from diffusers import FluxPipeline

# Set environment for single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test_single_gpu_baseline():
    print("🚀 Testing Single GPU Baseline with Flux")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    
    # Load model
    print("📦 Loading Flux model...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    prompt = "A beautiful landscape with mountains and lakes"
    
    # Warmup
    print("🔥 Warmup run...")
    start_time = time.time()
    _ = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=4,
        guidance_scale=0.0,
        output_type="latent"
    )
    torch.cuda.synchronize()
    warmup_time = time.time() - start_time
    print(f"Warmup time: {warmup_time:.2f}s")
    
    # Benchmark runs
    print("⏱️ Running benchmark...")
    times = []
    for i in range(3):
        start_time = time.time()
        images = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=torch.Generator(device="cuda").manual_seed(42 + i)
        ).images
        torch.cuda.synchronize()
        end_time = time.time()
        
        inference_time = end_time - start_time
        times.append(inference_time)
        print(f"Run {i+1}: {inference_time:.2f}s")
        
        # Save first image
        if i == 0:
            images[0].save("single_gpu_baseline.png")
    
    avg_time = sum(times) / len(times)
    print(f"\n📊 Single GPU Baseline Results:")
    print(f"Average time: {avg_time:.2f}s")
    print(f"Min time: {min(times):.2f}s")
    print(f"Max time: {max(times):.2f}s")
    print(f"💾 Image saved as: single_gpu_baseline.png")
    
    return avg_time

if __name__ == "__main__":
    test_single_gpu_baseline()
