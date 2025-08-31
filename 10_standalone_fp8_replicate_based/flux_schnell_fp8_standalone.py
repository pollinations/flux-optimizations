#!/usr/bin/env python3
"""
Standalone Flux Schnell FP8 implementation extracted from cog-flux
Runs Flux Schnell with FP8 quantization optimizations without Cog dependency
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Tuple
import subprocess
import tarfile
import requests
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image
from safetensors.torch import load_file as load_sft

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fp8.flux_pipeline import FluxPipeline
from fp8.util import (
    LoadedModels,
    ModelSpec,
    load_config_from_path,
    load_models_from_config,
)
from fp8.modules.conditioner import HFEmbedder
from fp8.util import load_autoencoder, load_text_encoders

# Model URLs from Replicate
MODEL_URLS = {
    "t5": "https://weights.replicate.delivery/default/official-models/flux/t5/t5-v1_1-xxl.tar",
    "clip": "https://weights.replicate.delivery/default/official-models/flux/clip/clip-vit-large-patch14.tar",
    "ae": "https://weights.replicate.delivery/default/official-models/flux/ae/ae.sft",
    "schnell_fp8": "https://weights.replicate.delivery/default/official-models/flux/schnell/schnell-fp8.safetensors",
}

# Cache directories
CACHE_BASE = Path("./model-cache")
MODEL_CACHES = {
    "t5": CACHE_BASE / "t5",
    "clip": CACHE_BASE / "clip", 
    "ae": CACHE_BASE / "ae" / "ae.sft",
    "schnell_fp8": CACHE_BASE / "schnell-fp8" / "schnell-fp8.safetensors",
}

# Aspect ratios from the original implementation
ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}


def download_file(url: str, dest_path: Path, desc: str = None) -> None:
    """Download a file with progress bar"""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists():
        print(f"✓ {desc or dest_path.name} already exists, skipping download")
        return
        
    print(f"📥 Downloading {desc or dest_path.name}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=desc or dest_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_tar(tar_path: Path, extract_to: Path) -> None:
    """Extract tar file"""
    extract_to.mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(extract_to)
    
    # Remove the tar file after extraction
    tar_path.unlink()


def download_models() -> None:
    """Download all required models"""
    print("🚀 Starting model downloads...")
    
    # Download T5 text encoder
    t5_tar = CACHE_BASE / "t5.tar"
    download_file(MODEL_URLS["t5"], t5_tar, "T5 Text Encoder")
    if t5_tar.exists():
        print("📦 Extracting T5...")
        extract_tar(t5_tar, MODEL_CACHES["t5"])
    
    # Download CLIP text encoder  
    clip_tar = CACHE_BASE / "clip.tar"
    download_file(MODEL_URLS["clip"], clip_tar, "CLIP Text Encoder")
    if clip_tar.exists():
        print("📦 Extracting CLIP...")
        extract_tar(clip_tar, MODEL_CACHES["clip"])
    
    # Download AutoEncoder
    download_file(MODEL_URLS["ae"], MODEL_CACHES["ae"], "AutoEncoder")
    
    # Download Flux Schnell FP8 model
    download_file(MODEL_URLS["schnell_fp8"], MODEL_CACHES["schnell_fp8"], "Flux Schnell FP8")
    
    print("✅ All models downloaded successfully!")


class FluxSchnellFP8:
    """Standalone Flux Schnell FP8 implementation"""
    
    def __init__(self, device: str = "cuda", gpu_id: int = 0, compile_model: bool = False):
        self.device = device
        self.gpu_id = gpu_id
        self.cuda_device = f"cuda:{gpu_id}" if device == "cuda" else device
        self.compile_model = compile_model
        self.pipeline = None
        
        # Set GPU and torch optimizations
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
    def load_models(self) -> None:
        """Load all required models"""
        print("🔄 Loading models...")
        
        # Load config for Flux Schnell FP8
        config_path = os.path.join(os.path.dirname(__file__), "fp8/configs/config-1-flux-schnell-fp8-h100.json")
        config = load_config_from_path(config_path)
        
        # Update paths to use our cache locations
        config.ckpt_path = str(MODEL_CACHES["schnell_fp8"])
        config.ae_path = str(MODEL_CACHES["ae"])
        config.text_enc_path = str(MODEL_CACHES["t5"])
        
        # Disable compilation if requested
        if not self.compile_model:
            config.compile_whole_model = False
            config.compile_extras = False
            config.compile_blocks = False
        
        # Disable T5 quantization to avoid inference tensor issues
        config.text_enc_quantization_dtype = None
        
        # Ensure all models are on the same device
        config.text_enc_device = self.cuda_device
        config.ae_device = self.cuda_device
        config.flux_device = self.cuda_device
        
        # Load models from config
        loaded_models = load_models_from_config(config)
        
        # Create pipeline with explicit GPU device assignment
        with torch.cuda.device(self.gpu_id):
            self.pipeline = FluxPipeline(
                name="flux-schnell-fp8",
                offload=False,  # Keep models in VRAM for better performance
                clip=loaded_models.clip,
                t5=loaded_models.t5,
                model=loaded_models.flow,
                ae=loaded_models.ae,
                dtype=torch.bfloat16,
                config=config,
                # Force all models to use the specified GPU
                flux_device=self.cuda_device,
                ae_device=self.cuda_device,
                clip_device=self.cuda_device, 
                t5_device=self.cuda_device,
            )
        
        print("✅ Models loaded successfully!")
    
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_steps: int = 4,
        guidance: float = 0.0,  # Schnell doesn't use guidance
        seed: int = None,
        num_images: int = 1,
    ) -> List[Image.Image]:
        """Generate images using Flux Schnell FP8"""
        
        if self.pipeline is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        
        print(f"🎨 Generating {num_images} image(s) with prompt: '{prompt}'")
        print(f"   Size: {width}x{height}, Steps: {num_steps}, Seed: {seed}")
        
        start_time = time.time()
        
        # Generate images
        pil_images, np_images = self.pipeline.generate(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
            num_images=num_images,
        )
        
        # Use only PIL images
        images = pil_images
        
        end_time = time.time()
        print(f"⚡ Generation completed in {end_time - start_time:.2f}s")
        
        return images
    
    def save_images(self, images: List[Image.Image], output_dir: str = "./outputs") -> List[str]:
        """Save generated images"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_paths = []
        timestamp = int(time.time())
        
        for i, img in enumerate(images):
            filename = f"flux_schnell_fp8_{timestamp}_{i:03d}.png"
            filepath = output_path / filename
            # Handle case where img might be a list or nested structure
            if isinstance(img, list):
                img = img[0] if img else None
            if img is not None:
                # Convert numpy array to PIL Image if needed
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img.astype(np.uint8))
                img.save(filepath, "PNG")
                saved_paths.append(str(filepath))
                print(f"💾 Saved: {filepath}")
        
        return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Flux Schnell FP8 Standalone Generator")
    parser.add_argument("--prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("--width", type=int, default=1024, help="Image width (default: 1024)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (default: 1024)")
    parser.add_argument("--aspect-ratio", type=str, choices=list(ASPECT_RATIOS.keys()), 
                       help="Aspect ratio (overrides width/height)")
    parser.add_argument("--steps", type=int, default=4, help="Number of inference steps (default: 4)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible generation")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--download-models", action="store_true", help="Download models and exit")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile for faster startup")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (default: cuda)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("❌ CUDA not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "cuda" and args.gpu >= torch.cuda.device_count():
        print(f"❌ GPU {args.gpu} not available, only {torch.cuda.device_count()} GPUs found")
        print(f"   Using GPU 0 instead")
        args.gpu = 0
    
    # Download models if requested
    if args.download_models:
        download_models()
        return
    
    # Check if prompt is provided for generation
    if not args.prompt:
        print("❌ Prompt is required for image generation")
        print("Use --prompt 'your prompt here' or --download-models to download models")
        return
    
    # Check if models exist
    missing_models = []
    for name, path in MODEL_CACHES.items():
        if not path.exists():
            missing_models.append(name)
    
    if missing_models:
        print(f"❌ Missing models: {', '.join(missing_models)}")
        print("Run with --download-models to download required models")
        return
    
    # Set dimensions from aspect ratio if provided
    if args.aspect_ratio:
        args.width, args.height = ASPECT_RATIOS[args.aspect_ratio]
    
    # Initialize and run generator
    generator = FluxSchnellFP8(device=args.device, gpu_id=args.gpu, compile_model=not args.no_compile)
    
    try:
        generator.load_models()
        images = generator.generate(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            num_steps=args.steps,
            seed=args.seed,
            num_images=args.num_images,
        )
        generator.save_images(images, args.output_dir)
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        raise


if __name__ == "__main__":
    main()
