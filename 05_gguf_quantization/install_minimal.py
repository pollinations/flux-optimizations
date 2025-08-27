#!/usr/bin/env python3
"""
Minimal installation script for GGUF quantization testing
Handles disk space constraints by installing only essential packages
"""
import subprocess
import sys
import os

def run_command(cmd):
    """Run command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ {cmd}")
            return True
        else:
            print(f"✗ {cmd}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ {cmd} - Exception: {e}")
        return False

def check_space():
    """Check available disk space"""
    result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    if len(lines) > 1:
        space_info = lines[1].split()
        available = space_info[3]
        usage = space_info[4]
        print(f"Available space: {available}, Usage: {usage}")
        return available, usage
    return None, None

def install_packages():
    """Install packages with minimal footprint"""
    print("=== Checking disk space ===")
    available, usage = check_space()
    
    if usage and int(usage.rstrip('%')) > 95:
        print("WARNING: Disk usage > 95%. Installation may fail.")
        print("Consider freeing up space or using a different environment.")
        return False
    
    print("\n=== Installing minimal packages ===")
    
    # Try installing with --no-cache-dir to save space
    packages = [
        "pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121",
        "pip install --no-cache-dir diffusers",
        "pip install --no-cache-dir transformers",
        "pip install --no-cache-dir gguf",
        "pip install --no-cache-dir huggingface_hub",
        "pip install --no-cache-dir accelerate"
    ]
    
    success_count = 0
    for package in packages:
        if run_command(package):
            success_count += 1
        else:
            print(f"Failed to install: {package}")
    
    print(f"\nInstalled {success_count}/{len(packages)} packages")
    return success_count > 0

if __name__ == "__main__":
    print("=== Minimal GGUF Installation ===")
    success = install_packages()
    
    if success:
        print("\n=== Testing installation ===")
        run_command("python3 test_environment.py")
    else:
        print("\nInstallation failed. Check disk space and try manual installation.")
