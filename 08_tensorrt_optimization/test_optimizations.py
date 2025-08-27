#!/usr/bin/env python3
"""
Test script to verify FLUX optimization layers work incrementally
"""
import torch
import gc
import os
from diffusers import FluxPipeline

def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

def test_basic_load():
    """Test basic model loading with bfloat16"""
    print("Testing basic model loading...")
    try:
        # Use a smaller GPU if available
        device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        print(f"✓ Model loaded successfully on {device}")
        print(f"Memory usage: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
        
        # Test basic inference with minimal steps
        prompt = "A simple test image"
        with torch.no_grad():
            image = pipeline(prompt, num_inference_steps=1, height=512, width=512).images[0]
        
        print("✓ Basic inference successful")
        image.save("test_basic.png")
        
        del pipeline
        clear_memory()
        return True
        
    except Exception as e:
        print(f"✗ Basic loading failed: {e}")
        clear_memory()
        return False

def test_optimizations():
    """Test optimization layers incrementally"""
    print("\nTesting optimization layers...")
    
    try:
        device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        
        # Load with optimizations
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        
        print("✓ Base model loaded")
        
        # Apply channels_last memory format
        pipeline.vae.to(memory_format=torch.channels_last)
        print("✓ Channels last applied")
        
        # Fuse QKV projections
        pipeline.transformer.fuse_qkv_projections()
        pipeline.vae.fuse_qkv_projections()
        print("✓ QKV fusion applied")
        
        # Test inference
        prompt = "A cat with optimizations"
        with torch.no_grad():
            image = pipeline(prompt, num_inference_steps=1, height=512, width=512).images[0]
        
        print("✓ Optimized inference successful")
        image.save("test_optimized.png")
        
        del pipeline
        clear_memory()
        return True
        
    except Exception as e:
        print(f"✗ Optimization test failed: {e}")
        clear_memory()
        return False

def main():
    print("FLUX Optimization Test")
    print("=" * 50)
    
    # Check available GPUs
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1e9
        print(f"GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    
    # Test basic functionality
    if test_basic_load():
        print("\n✓ Basic test passed")
        
        # Test optimizations
        if test_optimizations():
            print("\n✓ All optimization tests passed!")
        else:
            print("\n✗ Optimization tests failed")
    else:
        print("\n✗ Basic test failed - cannot proceed with optimizations")

if __name__ == "__main__":
    main()
