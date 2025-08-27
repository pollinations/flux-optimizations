#!/usr/bin/env python3
"""
Debug xDiT setup and test basic functionality
"""

import torch
import os
import sys

def test_basic_xdit():
    """Test basic xDiT imports and functionality"""
    print("🔍 Testing xDiT basic functionality...")
    
    try:
        from xfuser import xFuserArgs
        from xfuser.config import FlexibleArgumentParser
        print("✅ xFuser imports successful")
    except Exception as e:
        print(f"❌ xFuser import failed: {e}")
        return False
    
    try:
        from xfuser.core.distributed import get_world_group
        print("✅ Distributed imports successful")
    except Exception as e:
        print(f"❌ Distributed import failed: {e}")
        return False
    
    # Test argument parsing
    try:
        parser = FlexibleArgumentParser()
        args = xFuserArgs.add_cli_args(parser)
        print("✅ Argument parsing successful")
    except Exception as e:
        print(f"❌ Argument parsing failed: {e}")
        return False
    
    return True

def test_flux_compatibility():
    """Test Flux model compatibility with xDiT"""
    print("\n🔍 Testing Flux compatibility...")
    
    try:
        from diffusers import FluxPipeline
        print("✅ FluxPipeline import successful")
        
        # Try loading model metadata only
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.bfloat16,
            device_map=None  # Don't load to GPU yet
        )
        print("✅ Flux model metadata loaded")
        
        # Check model components
        print(f"   Transformer: {type(pipe.transformer).__name__}")
        print(f"   VAE: {type(pipe.vae).__name__}")
        print(f"   Text Encoder: {type(pipe.text_encoder).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Flux compatibility test failed: {e}")
        return False

def test_multi_gpu_detection():
    """Test multi-GPU detection and setup"""
    print("\n🔍 Testing multi-GPU detection...")
    
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
    
    # Test NCCL availability
    try:
        if torch.distributed.is_nccl_available():
            print("✅ NCCL available for multi-GPU communication")
        else:
            print("❌ NCCL not available")
    except:
        print("❌ Could not check NCCL availability")
    
    return gpu_count > 1

def test_simple_distributed():
    """Test simple distributed setup"""
    print("\n🔍 Testing simple distributed setup...")
    
    # Set minimal environment
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    
    try:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                rank=0,
                world_size=1
            )
            print("✅ Distributed process group initialized")
        else:
            print("✅ Distributed already initialized")
        
        return True
    except Exception as e:
        print(f"❌ Distributed setup failed: {e}")
        return False

def main():
    print("🚀 xDiT Debug and Compatibility Test")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    # Run tests
    tests = [
        ("Basic xDiT", test_basic_xdit),
        ("Flux Compatibility", test_flux_compatibility),
        ("Multi-GPU Detection", test_multi_gpu_detection),
        ("Simple Distributed", test_simple_distributed),
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\n{'='*50}")
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results[name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Summary:")
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! xDiT should work correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the issues above.")
    
    return all_passed

if __name__ == "__main__":
    main()
