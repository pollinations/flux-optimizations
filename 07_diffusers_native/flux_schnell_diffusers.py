#!/usr/bin/env python3
"""
Flux Schnell with Vanilla Diffusers - Native Optimizations
Test implementation with memory-efficient attention, model offloading, and VAE optimizations.
"""

import torch
import time
import gc
import psutil
import os
from diffusers import FluxPipeline
from PIL import Image
import argparse
from typing import Optional, Dict, Any


class FluxSchnellOptimized:
    """Flux Schnell with comprehensive diffusers native optimizations."""
    
    def __init__(self, model_id: str = "black-forest-labs/FLUX.1-schnell"):
        self.model_id = model_id
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimization_config = {
            'sdpa': True,  # Scaled Dot-Product Attention (automatic in PyTorch 2.0+)
            'xformers': False,  # Will try to enable if available
            'cpu_offload': False,
            'sequential_offload': False,
            'vae_slicing': False,
            'vae_tiling': False,
            'attention_slicing': False,
            'channels_last': False,
            'compile': False
        }
        
    def load_pipeline(self, **optimization_kwargs):
        """Load the Flux pipeline with specified optimizations."""
        print(f"Loading Flux Schnell from {self.model_id}...")
        print(f"Device: {self.device}")
        
        # Update optimization config
        self.optimization_config.update(optimization_kwargs)
        
        # Load pipeline
        self.pipeline = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto" if self.optimization_config['cpu_offload'] else None
        )
        
        # Move to device if not using offloading
        if not self.optimization_config['cpu_offload'] and not self.optimization_config['sequential_offload']:
            self.pipeline = self.pipeline.to(self.device)
        
        self._apply_optimizations()
        print("Pipeline loaded and optimized!")
        
    def _apply_optimizations(self):
        """Apply all enabled optimizations."""
        print("\nApplying optimizations:")
        
        # Memory-efficient attention
        if self.optimization_config['xformers']:
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
                print("✓ xFormers memory-efficient attention enabled")
            except Exception as e:
                print(f"✗ xFormers not available: {e}")
                print("✓ Using SDPA (Scaled Dot-Product Attention) instead")
        else:
            print("✓ Using SDPA (Scaled Dot-Product Attention) - default in PyTorch 2.0+")
        
        # Model offloading
        if self.optimization_config['sequential_offload']:
            self.pipeline.enable_sequential_cpu_offload()
            print("✓ Sequential CPU offloading enabled (maximum memory savings)")
        elif self.optimization_config['cpu_offload']:
            self.pipeline.enable_model_cpu_offload()
            print("✓ Model CPU offloading enabled")
        
        # VAE optimizations
        if self.optimization_config['vae_slicing']:
            self.pipeline.enable_vae_slicing()
            print("✓ VAE slicing enabled")
            
        if self.optimization_config['vae_tiling']:
            self.pipeline.enable_vae_tiling()
            print("✓ VAE tiling enabled")
        
        # Attention slicing
        if self.optimization_config['attention_slicing']:
            self.pipeline.enable_attention_slicing()
            print("✓ Attention slicing enabled")
        
        # Memory format optimization
        if self.optimization_config['channels_last']:
            try:
                if hasattr(self.pipeline, 'transformer'):
                    self.pipeline.transformer.to(memory_format=torch.channels_last)
                if hasattr(self.pipeline, 'vae'):
                    self.pipeline.vae.to(memory_format=torch.channels_last)
                print("✓ Channels last memory format enabled")
            except Exception as e:
                print(f"✗ Channels last format failed: {e}")
        
        # Torch compile
        if self.optimization_config['compile']:
            try:
                if hasattr(self.pipeline, 'transformer'):
                    self.pipeline.transformer = torch.compile(self.pipeline.transformer)
                print("✓ Torch compile enabled")
            except Exception as e:
                print(f"✗ Torch compile failed: {e}")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_stats = {}
        
        if torch.cuda.is_available():
            memory_stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_stats['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3    # GB
            memory_stats['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        process = psutil.Process()
        memory_stats['cpu_memory'] = process.memory_info().rss / 1024**3  # GB
        
        return memory_stats
    
    def generate_image(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        seed: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> Image.Image:
        """Generate an image with the given parameters."""
        if self.pipeline is None:
            raise ValueError("Pipeline not loaded. Call load_pipeline() first.")
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        print(f"\nGenerating image:")
        print(f"Prompt: {prompt}")
        print(f"Size: {width}x{height}")
        print(f"Steps: {num_inference_steps}")
        print(f"Guidance scale: {guidance_scale}")
        print(f"Seed: {seed}")
        
        # Clear cache and get initial memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        initial_memory = self.get_memory_usage()
        print(f"Initial GPU memory: {initial_memory.get('gpu_allocated', 0):.2f} GB")
        
        # Generate image
        start_time = time.time()
        
        with torch.inference_mode():
            image = self.pipeline(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(seed) if seed else None
            ).images[0]
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Get final memory usage
        final_memory = self.get_memory_usage()
        peak_memory = self.get_memory_usage()
        
        print(f"\nGeneration completed!")
        print(f"Time: {generation_time:.2f} seconds")
        print(f"Peak GPU memory: {peak_memory.get('gpu_max_allocated', 0):.2f} GB")
        print(f"Final GPU memory: {final_memory.get('gpu_allocated', 0):.2f} GB")
        
        # Save image if path provided
        if output_path:
            image.save(output_path)
            print(f"Image saved to: {output_path}")
        
        return image
    
    def benchmark(self, prompts: list, **generation_kwargs):
        """Run benchmark with multiple prompts."""
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n{'='*60}")
            print(f"Benchmark {i+1}/{len(prompts)}")
            print(f"{'='*60}")
            
            # Clear cache before each generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            gc.collect()
            
            start_time = time.time()
            image = self.generate_image(prompt, **generation_kwargs)
            end_time = time.time()
            
            memory_stats = self.get_memory_usage()
            
            result = {
                'prompt': prompt,
                'generation_time': end_time - start_time,
                'memory_stats': memory_stats,
                'image_size': (image.width, image.height)
            }
            results.append(result)
        
        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        
        avg_time = sum(r['generation_time'] for r in results) / len(results)
        max_memory = max(r['memory_stats'].get('gpu_max_allocated', 0) for r in results)
        
        print(f"Average generation time: {avg_time:.2f} seconds")
        print(f"Peak GPU memory usage: {max_memory:.2f} GB")
        print(f"Optimizations: {self.optimization_config}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Flux Schnell with Diffusers Native Optimizations")
    parser.add_argument("--prompt", type=str, default="A beautiful landscape with mountains and a lake at sunset", help="Text prompt")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--steps", type=int, default=4, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="flux_schnell_output.png", help="Output image path")
    
    # Optimization flags
    parser.add_argument("--xformers", action="store_true", help="Enable xFormers memory-efficient attention")
    parser.add_argument("--cpu-offload", action="store_true", help="Enable CPU offloading")
    parser.add_argument("--sequential-offload", action="store_true", help="Enable sequential CPU offloading")
    parser.add_argument("--vae-slicing", action="store_true", help="Enable VAE slicing")
    parser.add_argument("--vae-tiling", action="store_true", help="Enable VAE tiling")
    parser.add_argument("--attention-slicing", action="store_true", help="Enable attention slicing")
    parser.add_argument("--channels-last", action="store_true", help="Enable channels last memory format")
    parser.add_argument("--compile", action="store_true", help="Enable torch compile")
    
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark with multiple prompts")
    
    args = parser.parse_args()
    
    # Create Flux instance
    flux = FluxSchnellOptimized()
    
    # Load pipeline with optimizations
    optimizations = {
        'xformers': args.xformers,
        'cpu_offload': args.cpu_offload,
        'sequential_offload': args.sequential_offload,
        'vae_slicing': args.vae_slicing,
        'vae_tiling': args.vae_tiling,
        'attention_slicing': args.attention_slicing,
        'channels_last': args.channels_last,
        'compile': args.compile
    }
    
    flux.load_pipeline(**optimizations)
    
    if args.benchmark:
        # Benchmark prompts
        prompts = [
            "A beautiful landscape with mountains and a lake at sunset",
            "A futuristic city with flying cars and neon lights",
            "A portrait of a wise old wizard with a long beard",
            "A serene forest with sunlight filtering through the trees"
        ]
        
        flux.benchmark(
            prompts,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            seed=args.seed
        )
    else:
        # Single generation
        image = flux.generate_image(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            seed=args.seed,
            output_path=args.output
        )


if __name__ == "__main__":
    main()
