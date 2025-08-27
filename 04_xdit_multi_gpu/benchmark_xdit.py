#!/usr/bin/env python3
"""
xDiT Multi-GPU Benchmarking Script
Comprehensive performance testing across different parallel strategies
"""

import torch
import time
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def run_benchmark(config):
    """Run a single benchmark configuration"""
    cmd = [
        "torchrun", f"--nproc_per_node={config['gpus']}", 
        "xdit_flux_inference.py",
        "--model", config["model"],
        "--prompt", config["prompt"],
        "--height", str(config["height"]),
        "--width", str(config["width"]),
        "--num_inference_steps", str(config["steps"]),
        "--benchmark_runs", str(config["runs"]),
        "--warmup_runs", str(config["warmup"])
    ]
    
    # Add parallel configuration
    if config.get("data_parallel_degree"):
        cmd.extend(["--data_parallel_degree", str(config["data_parallel_degree"])])
    if config.get("ulysses_degree"):
        cmd.extend(["--ulysses_degree", str(config["ulysses_degree"])])
    if config.get("ring_degree"):
        cmd.extend(["--ring_degree", str(config["ring_degree"])])
    if config.get("pipefusion_parallel_degree"):
        cmd.extend(["--pipefusion_parallel_degree", str(config["pipefusion_parallel_degree"])])
    if config.get("use_cfg_parallel"):
        cmd.append("--use_cfg_parallel")
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Timeout", -1


def parse_results(stdout):
    """Parse benchmark results from stdout"""
    lines = stdout.split('\n')
    results = {}
    
    for line in lines:
        if "Average Time:" in line:
            results["avg_time"] = float(line.split(":")[1].strip().replace("s", ""))
        elif "Min Time:" in line:
            results["min_time"] = float(line.split(":")[1].strip().replace("s", ""))
        elif "Max Time:" in line:
            results["max_time"] = float(line.split(":")[1].strip().replace("s", ""))
        elif "Images/Second:" in line:
            results["images_per_sec"] = float(line.split(":")[1].strip())
        elif "Time per Image:" in line:
            results["time_per_image"] = float(line.split(":")[1].strip().replace("s", ""))
    
    return results


def get_benchmark_configs():
    """Define benchmark configurations"""
    base_config = {
        "model": "black-forest-labs/FLUX.1-schnell",
        "prompt": "A beautiful landscape with mountains and lakes",
        "height": 1024,
        "width": 1024,
        "steps": 4,
        "runs": 3,
        "warmup": 1
    }
    
    configs = []
    
    # Single GPU baseline
    configs.append({
        **base_config,
        "name": "Single GPU Baseline",
        "gpus": 1,
        "data_parallel_degree": 1
    })
    
    # 2 GPU configurations
    configs.append({
        **base_config,
        "name": "2 GPU Ring Parallel",
        "gpus": 2,
        "ring_degree": 2
    })
    
    configs.append({
        **base_config,
        "name": "2 GPU Data Parallel",
        "gpus": 2,
        "data_parallel_degree": 2
    })
    
    # 4 GPU configurations
    configs.append({
        **base_config,
        "name": "4 GPU Ulysses Parallel",
        "gpus": 4,
        "ulysses_degree": 4
    })
    
    configs.append({
        **base_config,
        "name": "4 GPU Hybrid (Ulysses-2 x Ring-2)",
        "gpus": 4,
        "ulysses_degree": 2,
        "ring_degree": 2
    })
    
    configs.append({
        **base_config,
        "name": "4 GPU Data Parallel",
        "gpus": 4,
        "data_parallel_degree": 4
    })
    
    # 8 GPU configurations (if available)
    configs.append({
        **base_config,
        "name": "8 GPU Ulysses Parallel",
        "gpus": 8,
        "ulysses_degree": 8
    })
    
    configs.append({
        **base_config,
        "name": "8 GPU Hybrid (Ulysses-4 x Ring-2)",
        "gpus": 8,
        "ulysses_degree": 4,
        "ring_degree": 2
    })
    
    return configs


def main():
    parser = argparse.ArgumentParser(description="xDiT Multi-GPU Benchmark Suite")
    parser.add_argument("--max_gpus", type=int, default=8, help="Maximum GPUs to test")
    parser.add_argument("--output", type=str, default="benchmark_results.json", 
                       help="Output file for results")
    args = parser.parse_args()
    
    # Check available GPUs
    available_gpus = torch.cuda.device_count()
    print(f"🔍 Available GPUs: {available_gpus}")
    
    if available_gpus == 0:
        print("❌ No GPUs available!")
        return
    
    # Get benchmark configurations
    configs = get_benchmark_configs()
    
    # Filter configs based on available GPUs
    valid_configs = [c for c in configs if c["gpus"] <= min(available_gpus, args.max_gpus)]
    
    print(f"📊 Running {len(valid_configs)} benchmark configurations...")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "available_gpus": available_gpus,
        "max_gpus_tested": min(available_gpus, args.max_gpus),
        "benchmarks": []
    }
    
    for i, config in enumerate(valid_configs):
        print(f"\n{'='*60}")
        print(f"📈 Benchmark {i+1}/{len(valid_configs)}: {config['name']}")
        print(f"{'='*60}")
        
        stdout, stderr, returncode = run_benchmark(config)
        
        if returncode == 0:
            parsed_results = parse_results(stdout)
            if parsed_results:
                benchmark_result = {
                    "name": config["name"],
                    "config": config,
                    "results": parsed_results,
                    "status": "success"
                }
                print(f"✅ Success: {parsed_results.get('avg_time', 'N/A')}s avg")
            else:
                benchmark_result = {
                    "name": config["name"],
                    "config": config,
                    "status": "failed",
                    "error": "Could not parse results"
                }
                print(f"❌ Failed: Could not parse results")
        else:
            benchmark_result = {
                "name": config["name"],
                "config": config,
                "status": "failed",
                "error": stderr
            }
            print(f"❌ Failed: {stderr}")
        
        results["benchmarks"].append(benchmark_result)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output}")
    
    # Print summary
    print(f"\n📊 Benchmark Summary:")
    print(f"{'Configuration':<35} {'Status':<10} {'Avg Time':<10} {'Speedup':<10}")
    print("-" * 70)
    
    baseline_time = None
    for benchmark in results["benchmarks"]:
        if benchmark["status"] == "success":
            avg_time = benchmark["results"].get("avg_time", 0)
            if "Single GPU" in benchmark["name"]:
                baseline_time = avg_time
                speedup = "1.0x"
            elif baseline_time:
                speedup = f"{baseline_time/avg_time:.1f}x"
            else:
                speedup = "N/A"
            
            print(f"{benchmark['name']:<35} {'✅':<10} {avg_time:<10.2f} {speedup:<10}")
        else:
            print(f"{benchmark['name']:<35} {'❌':<10} {'N/A':<10} {'N/A':<10}")


if __name__ == "__main__":
    main()
