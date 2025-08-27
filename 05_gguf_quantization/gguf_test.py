#!/usr/bin/env python3
"""
GGUF Quantization Test Script for FLUX Schnell
Tests different quantization levels and measures performance
"""
import time
import gc
import psutil
import os

def get_gpu_memory():
    """Get GPU memory usage if torch is available"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3  # GB
    except ImportError:
        pass
    return None

def test_model_loading(model_url, quant_type):
    """Test loading a GGUF quantized model"""
    print(f"\n=== Testing {quant_type} Quantization ===")
    print(f"Model: {model_url}")
    
    start_time = time.time()
    initial_gpu_mem = get_gpu_memory()
    
    try:
        # Import required libraries
        from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
        import torch
        
        print("✓ Libraries imported successfully")
        
        # Configure quantization
        quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
        
        # Load model
        print("Loading model...")
        transformer = FluxTransformer2DModel.from_single_file(
            model_url,
            quantization_config=quantization_config
        )
        
        load_time = time.time() - start_time
        final_gpu_mem = get_gpu_memory()
        
        print(f"✓ Model loaded successfully in {load_time:.2f}s")
        if initial_gpu_mem and final_gpu_mem:
            mem_used = final_gpu_mem - initial_gpu_mem
            print(f"GPU Memory used: {mem_used:.2f} GB")
        
        # Test basic functionality
        print("Testing model properties...")
        print(f"Model config: {type(transformer.config)}")
        print(f"Device: {next(transformer.parameters()).device}")
        
        # Cleanup
        del transformer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True, load_time, final_gpu_mem - initial_gpu_mem if initial_gpu_mem and final_gpu_mem else 0
        
    except Exception as e:
        print(f"✗ Error loading {quant_type}: {e}")
        return False, 0, 0

def main():
    """Main test function"""
    print("=== GGUF Quantization Test ===")
    
    # Check prerequisites
    try:
        import torch
        import diffusers
        print(f"✓ PyTorch {torch.__version__}")
        print(f"✓ Diffusers {diffusers.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name()}")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Run: python3 install_minimal.py")
        return
    
    # Model URLs for testing
    models = {
        "Q8_0": "https://huggingface.co/city96/FLUX.1-schnell-gguf/blob/main/flux1-schnell-Q8_0.gguf",
        "Q4_K": "https://huggingface.co/city96/FLUX.1-schnell-gguf/blob/main/flux1-schnell-Q4_K.gguf", 
        "Q2_K": "https://huggingface.co/city96/FLUX.1-schnell-gguf/blob/main/flux1-schnell-Q2_K.gguf"
    }
    
    results = {}
    
    # Test each quantization level
    for quant_type, model_url in models.items():
        success, load_time, memory_used = test_model_loading(model_url, quant_type)
        results[quant_type] = {
            'success': success,
            'load_time': load_time,
            'memory_used': memory_used
        }
        
        # Small delay between tests
        time.sleep(2)
    
    # Print summary
    print("\n=== Test Summary ===")
    for quant_type, result in results.items():
        status = "✓" if result['success'] else "✗"
        print(f"{status} {quant_type}: {result['load_time']:.2f}s, {result['memory_used']:.2f}GB")
    
    # Recommendations
    print("\n=== Recommendations ===")
    successful = [q for q, r in results.items() if r['success']]
    if successful:
        print(f"Successfully tested: {', '.join(successful)}")
        if 'Q4_K' in successful:
            print("✓ Q4_K recommended for balanced performance/quality")
        elif 'Q8_0' in successful:
            print("✓ Q8_0 recommended for best quality")
        elif 'Q2_K' in successful:
            print("✓ Q2_K available for maximum memory savings")
    else:
        print("No models loaded successfully. Check network and dependencies.")

if __name__ == "__main__":
    main()
