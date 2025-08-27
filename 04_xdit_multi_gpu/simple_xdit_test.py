
import torch
import time
import os
from diffusers import FluxPipeline
from xfuser import xFuserArgs, xDiTParallel
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import get_world_group, init_distributed_environment

def main():
    # Initialize distributed environment
    init_distributed_environment()
    world_group = get_world_group()
    local_rank = world_group.local_rank
    world_size = world_group.world_size
    
    if local_rank == 0:
        print(f"🔍 Running on {world_size} GPUs")
    
    # Parse arguments
    parser = FlexibleArgumentParser()
    args = xFuserArgs.add_cli_args(parser).parse_args()
    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    
    # Load model
    if local_rank == 0:
        print("📦 Loading Flux model...")
    
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16
    ).to(f"cuda:{local_rank}")
    
    # Apply xDiT parallelization
    pipe = xDiTParallel(pipe, engine_config, input_config)
    
    # Generate image
    prompt = "A beautiful landscape with mountains and lakes"
    
    if local_rank == 0:
        print("🎨 Generating image...")
    
    start_time = time.time()
    images = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=4,
        guidance_scale=0.0,
        generator=torch.Generator(device="cuda").manual_seed(42)
    ).images
    torch.cuda.synchronize()
    end_time = time.time()
    
    inference_time = end_time - start_time
    
    if local_rank == 0:
        print(f"Average time: {inference_time:.2f}s")
        images[0].save(f"multi_gpu_{world_size}gpu.png")
        print(f"💾 Image saved as: multi_gpu_{world_size}gpu.png")

if __name__ == "__main__":
    main()
