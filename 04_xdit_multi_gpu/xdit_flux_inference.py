#!/usr/bin/env python3
"""
xDiT Multi-GPU Flux Inference Script
Supports various parallel strategies for optimal performance
"""

import torch
import time
import argparse
from pathlib import Path
from diffusers import FluxPipeline
from xfuser import xFuserArgs, xDiTParallel
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import get_world_group


def setup_args():
    """Setup command line arguments"""
    parser = FlexibleArgumentParser(description="xDiT Multi-GPU Flux Inference")
    
    # Add xFuser arguments
    args = xFuserArgs.add_cli_args(parser)
    
    # Add custom arguments
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-schnell",
                       help="Model name or path")
    parser.add_argument("--prompt", type=str, nargs="+", 
                       default=["A beautiful landscape with mountains and lakes"],
                       help="Text prompts for generation")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--num_inference_steps", type=int, default=4, 
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=0.0,
                       help="Guidance scale (0.0 for FLUX.1-schnell)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory for generated images")
    parser.add_argument("--warmup_runs", type=int, default=1,
                       help="Number of warmup runs")
    parser.add_argument("--benchmark_runs", type=int, default=3,
                       help="Number of benchmark runs")
    
    return parser.parse_args()


def print_gpu_info():
    """Print GPU information"""
    print(f"🔍 GPU Information:")
    print(f"   Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
    print()


def print_parallel_config(engine_config):
    """Print parallel configuration"""
    print(f"⚡ Parallel Configuration:")
    print(f"   Data Parallel Degree: {engine_config.parallel_config.dp_degree}")
    print(f"   Ulysses Degree: {engine_config.parallel_config.ulysses_degree}")
    print(f"   Ring Degree: {engine_config.parallel_config.ring_degree}")
    print(f"   PipeFusion Degree: {engine_config.parallel_config.pipefusion_parallel_degree}")
    print(f"   CFG Parallel: {engine_config.parallel_config.cfg_degree > 1}")
    print()


def load_model(model_name, local_rank):
    """Load Flux model"""
    print(f"📦 Loading model: {model_name}")
    
    # Determine guidance scale based on model
    guidance_scale = 0.0 if "schnell" in model_name.lower() else 3.5
    
    pipe = FluxPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None  # Let xDiT handle device placement
    ).to(f"cuda:{local_rank}")
    
    return pipe, guidance_scale


def generate_images(pipe, prompts, args, guidance_scale):
    """Generate images with timing"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Warmup runs
    if args.warmup_runs > 0:
        print(f"🔥 Running {args.warmup_runs} warmup iterations...")
        for i in range(args.warmup_runs):
            _ = pipe(
                prompt=prompts[0],
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
                output_type="latent"  # Skip VAE decode for warmup
            )
        torch.cuda.synchronize()
        print("✅ Warmup complete")
    
    # Benchmark runs
    print(f"⏱️ Running {args.benchmark_runs} benchmark iterations...")
    times = []
    
    for run in range(args.benchmark_runs):
        start_time = time.time()
        
        images = pipe(
            prompt=prompts,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(args.seed + run),
            output_type="pil"
        ).images
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        inference_time = end_time - start_time
        times.append(inference_time)
        
        print(f"   Run {run + 1}: {inference_time:.2f}s")
        
        # Save images from first run
        if run == 0:
            for i, image in enumerate(images):
                image.save(output_dir / f"image_{i}.png")
    
    return times


def print_results(times, prompts, args, world_size):
    """Print benchmark results"""
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n📊 Benchmark Results:")
    print(f"   Model: {args.model}")
    print(f"   Resolution: {args.width}x{args.height}")
    print(f"   Steps: {args.num_inference_steps}")
    print(f"   Prompts: {len(prompts)}")
    print(f"   GPUs: {world_size}")
    print(f"   Average Time: {avg_time:.2f}s")
    print(f"   Min Time: {min_time:.2f}s")
    print(f"   Max Time: {max_time:.2f}s")
    print(f"   Images/Second: {len(prompts) / avg_time:.2f}")
    print(f"   Time per Image: {avg_time / len(prompts):.2f}s")


def main():
    args = setup_args()
    
    # Create engine configuration
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    
    # Get distributed info
    world_group = get_world_group()
    local_rank = world_group.local_rank
    world_size = world_group.world_size
    
    if local_rank == 0:
        print("🚀 xDiT Multi-GPU Flux Inference")
        print_gpu_info()
        print_parallel_config(engine_config)
    
    # Load model
    pipe, auto_guidance_scale = load_model(args.model, local_rank)
    
    # Use auto-detected guidance scale if not specified
    guidance_scale = args.guidance_scale if args.guidance_scale > 0 else auto_guidance_scale
    
    # Apply xDiT parallelization
    if local_rank == 0:
        print("⚡ Applying xDiT parallelization...")
    
    pipe = xDiTParallel(pipe, engine_config, input_config)
    
    # Generate images
    if local_rank == 0:
        print(f"🎨 Generating images for prompts: {args.prompt}")
    
    times = generate_images(pipe, args.prompt, args, guidance_scale)
    
    # Print results (only on rank 0)
    if local_rank == 0:
        print_results(times, args.prompt, args, world_size)
        print(f"💾 Images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
