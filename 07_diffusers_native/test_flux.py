#!/usr/bin/env python3
"""
Simple test script for Flux Schnell with diffusers native optimizations.
"""

from flux_schnell_diffusers import FluxSchnellOptimized
import torch

def test_basic_generation():
    """Test basic image generation with default settings."""
    print("Testing Flux Schnell with vanilla diffusers...")
    
    # Create instance
    flux = FluxSchnellOptimized()
    
    # Load with basic optimizations (SDPA enabled by default)
    flux.load_pipeline(
        vae_slicing=True,  # Enable for memory efficiency
        attention_slicing=True  # Enable for memory efficiency
    )
    
    # Generate a test image
    image = flux.generate_image(
        prompt="A beautiful sunset over mountains, digital art",
        width=1024,
        height=1024,
        num_inference_steps=4,
        seed=42,
        output_path="test_output.png"
    )
    
    print("Test completed successfully!")
    return image

def test_memory_optimizations():
    """Test with maximum memory optimizations."""
    print("\nTesting with memory optimizations...")
    
    flux = FluxSchnellOptimized()
    
    # Load with memory-focused optimizations
    flux.load_pipeline(
        sequential_offload=True,  # Maximum memory savings
        vae_slicing=True,
        vae_tiling=True,
        attention_slicing=True
    )
    
    # Generate with higher resolution to test memory efficiency
    image = flux.generate_image(
        prompt="A detailed cyberpunk cityscape at night",
        width=1024,
        height=1024,
        num_inference_steps=4,
        seed=123,
        output_path="test_memory_optimized.png"
    )
    
    print("Memory optimization test completed!")
    return image

if __name__ == "__main__":
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Run basic test
    test_basic_generation()
    
    # Run memory optimization test
    test_memory_optimizations()
