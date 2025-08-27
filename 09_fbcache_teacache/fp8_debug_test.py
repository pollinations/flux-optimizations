#!/usr/bin/env python3
"""
Debug fp8 quantization for FLUX.1-schnell
Testing different quantization approaches
"""

import torch
import time
from diffusers import FluxPipeline
import torchao
from torchao.quantization import quantize_, float8_weight_only
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

# Configuration
MODEL_ID = "black-forest-labs/FLUX.1-schnell"
DEVICE = "cuda"
DTYPE = torch.bfloat16

def test_fp8_quantization():
    print("Testing FP8 quantization approaches for FLUX.1-schnell")
    
    # Load pipeline
    print("Loading pipeline...")
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    
    prompt = "A red apple on a white table"
    
    # Baseline test
    print("\n1. Baseline test...")
    start_time = time.time()
    _ = pipe(prompt, num_inference_steps=4, height=512, width=512)
    torch.cuda.synchronize()
    baseline_time = time.time() - start_time
    print(f"Baseline: {baseline_time:.3f}s")
    
    # Test FP8 weight-only quantization
    print("\n2. Testing FP8 weight-only quantization...")
    try:
        quantize_(pipe.transformer, float8_weight_only())
        
        start_time = time.time()
        _ = pipe(prompt, num_inference_steps=4, height=512, width=512)
        torch.cuda.synchronize()
        fp8_time = time.time() - start_time
        print(f"FP8 weight-only: {fp8_time:.3f}s ({baseline_time/fp8_time:.2f}x)")
        
    except Exception as e:
        print(f"FP8 weight-only failed: {e}")
    
    # Test with torch.compile
    print("\n3. Testing torch.compile...")
    try:
        # Fresh pipeline for torch.compile
        pipe_compile = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
        ).to(DEVICE)
        
        # Compile the transformer
        pipe_compile.transformer = torch.compile(pipe_compile.transformer, mode="max-autotune-no-cudagraphs")
        
        # Warmup
        _ = pipe_compile(prompt, num_inference_steps=4, height=512, width=512)
        torch.cuda.synchronize()
        
        start_time = time.time()
        _ = pipe_compile(prompt, num_inference_steps=4, height=512, width=512)
        torch.cuda.synchronize()
        compile_time = time.time() - start_time
        print(f"torch.compile: {compile_time:.3f}s ({baseline_time/compile_time:.2f}x)")
        
    except Exception as e:
        print(f"torch.compile failed: {e}")
    
    # Test FBCache with proper threshold
    print("\n4. Testing FBCache with threshold 0.08...")
    try:
        # Fresh pipeline for FBCache
        pipe_fbcache = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
        ).to(DEVICE)
        
        apply_cache_on_pipe(pipe_fbcache, residual_diff_threshold=0.08)
        
        # Warmup
        _ = pipe_fbcache(prompt, num_inference_steps=4, height=512, width=512)
        torch.cuda.synchronize()
        
        start_time = time.time()
        _ = pipe_fbcache(prompt, num_inference_steps=4, height=512, width=512)
        torch.cuda.synchronize()
        fbcache_time = time.time() - start_time
        print(f"FBCache (0.08): {fbcache_time:.3f}s ({baseline_time/fbcache_time:.2f}x)")
        
    except Exception as e:
        print(f"FBCache failed: {e}")

if __name__ == "__main__":
    test_fp8_quantization()
