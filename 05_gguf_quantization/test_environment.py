#!/usr/bin/env python3
"""
Environment check script for GGUF quantization testing
"""
import sys
import subprocess

def check_python():
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

def check_gpu():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("NVIDIA GPU detected:")
            print(result.stdout.split('\n')[8:12])  # GPU info lines
        else:
            print("nvidia-smi not available")
    except FileNotFoundError:
        print("nvidia-smi not found")

def check_packages():
    packages = ['torch', 'diffusers', 'transformers', 'gguf', 'huggingface_hub']
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} available")
        except ImportError:
            print(f"✗ {package} not available")

def check_disk_space():
    try:
        result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
        print("Disk space:")
        print(result.stdout)
    except:
        print("Could not check disk space")

if __name__ == "__main__":
    print("=== Environment Check ===")
    check_python()
    print("\n=== GPU Check ===")
    check_gpu()
    print("\n=== Package Check ===")
    check_packages()
    print("\n=== Disk Space ===")
    check_disk_space()
