#!/usr/bin/env python3
"""
Simple example to run Flux Schnell with optimizations
"""

from flux_schnell_diffusers import FluxSchnellOptimized
import torch

def main():
    print("=== Flux Schnell with Vanilla Diffusers ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create Flux instance
    flux = FluxSchnellOptimized()
    
    # Load with memory-efficient optimizations
    print("\nLoading pipeline with optimizations...")
    flux.load_pipeline(
        vae_slicing=True,      # Memory efficient VAE
        attention_slicing=True  # Memory efficient attention
    )
    
    # Generate image
    print("\nGenerating image...")
    image = flux.generate_image(
        prompt="A serene mountain landscape at sunset with a crystal clear lake reflecting the orange sky",
        width=1024,
        height=1024,
        num_inference_steps=4,
        seed=42,
        output_path="flux_example_output.png"
    )
    
    print("\n✅ Generation completed successfully!")
    print("Check 'flux_example_output.png' for the generated image.")

if __name__ == "__main__":
    main()
