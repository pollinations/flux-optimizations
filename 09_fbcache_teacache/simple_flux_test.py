#!/usr/bin/env python3
"""
Simple FLUX.1-schnell test without FBCache to verify basic functionality
"""

import torch
import time
from diffusers import FluxPipeline

# Configuration
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEVICE = "cuda"
DTYPE = torch.bfloat16

def main():
    print("Simple FLUX.1-schnell test")
    print(f"Device: {DEVICE}")
    
    # Load pipeline
    print("Loading FLUX.1-schnell pipeline...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    
    # Test prompt
    prompt = "A red apple on a white table"
    
    # Baseline test
    print("\nRunning baseline test...")
    start_time = time.time()
    image = pipe(
        prompt,
        num_inference_steps=4,
        height=1024,
        width=1024,
    ).images[0]
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.3f}s")
    
    # Save image
    image.save("simple_flux_test.png")
    print(f"Image saved: simple_flux_test.png")

if __name__ == "__main__":
    main()
