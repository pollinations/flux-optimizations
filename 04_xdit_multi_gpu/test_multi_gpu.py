#!/usr/bin/env python3
"""
Multi-GPU test for xDiT Flux using different parallel strategies
"""

import torch
import time
import os
import subprocess
import sys
from pathlib import Path

def run_multi_gpu_test(gpus, strategy_name, strategy_args):
    """Run a multi-GPU test with specific strategy"""
    print(f"\n🚀 Testing {strategy_name} with {gpus} GPUs")
    
    # Create command
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={gpus}",
        "--standalone",
        "simple_xdit_test.py"
    ] + strategy_args
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            # Parse timing from output
            lines = result.stdout.split('\n')
            for line in lines:
                if "Average time:" in line:
                    time_str = line.split(":")[1].strip().replace("s", "")
                    return float(time_str)
            print(f"✅ {strategy_name} completed but couldn't parse timing")
            print("Output:", result.stdout[-500:])  # Last 500 chars
            return None
        else:
            print(f"❌ {strategy_name} failed:")
            print("Error:", result.stderr[-500:])  # Last 500 chars
            return None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {strategy_name} timed out")
        return None
    except Exception as e:
        print(f"❌ {strategy_name} error: {e}")
        return None

def create_simple_xdit_test():
    """Create a simple xDiT test script"""
    script_content = '''
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
'''
    
    with open("simple_xdit_test.py", "w") as f:
        f.write(script_content)

def main():
    print("🚀 xDiT Multi-GPU Testing Suite")
    
    # Check available GPUs
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    
    if available_gpus < 2:
        print("❌ Need at least 2 GPUs for multi-GPU testing")
        return
    
    # Create the simple test script
    create_simple_xdit_test()
    
    # Test configurations
    configs = [
        (2, "2-GPU Ring Parallel", ["--ring_degree", "2"]),
        (2, "2-GPU Data Parallel", ["--data_parallel_degree", "2"]),
    ]
    
    if available_gpus >= 4:
        configs.extend([
            (4, "4-GPU Ulysses Parallel", ["--ulysses_degree", "4"]),
            (4, "4-GPU Hybrid (2x2)", ["--ulysses_degree", "2", "--ring_degree", "2"]),
        ])
    
    if available_gpus >= 8:
        configs.extend([
            (8, "8-GPU Ulysses Parallel", ["--ulysses_degree", "8"]),
        ])
    
    # Run tests
    results = {}
    baseline_time = 1.61  # From single GPU test
    
    print(f"\n📊 Baseline (1 GPU): {baseline_time:.2f}s")
    
    for gpus, name, args in configs:
        if gpus <= available_gpus:
            result_time = run_multi_gpu_test(gpus, name, args)
            if result_time:
                speedup = baseline_time / result_time
                results[name] = {"time": result_time, "speedup": speedup}
                print(f"✅ {name}: {result_time:.2f}s ({speedup:.1f}x speedup)")
            else:
                results[name] = {"time": None, "speedup": None}
    
    # Summary
    print(f"\n📈 Multi-GPU Performance Summary:")
    print(f"{'Configuration':<25} {'Time':<10} {'Speedup':<10}")
    print("-" * 50)
    print(f"{'1-GPU Baseline':<25} {baseline_time:<10.2f} {'1.0x':<10}")
    
    for name, data in results.items():
        if data["time"]:
            print(f"{name:<25} {data['time']:<10.2f} {data['speedup']:<10.1f}x")
        else:
            print(f"{name:<25} {'Failed':<10} {'N/A':<10}")

if __name__ == "__main__":
    main()
