#!/usr/bin/env python3
"""
Pollinations-compatible Flux Schnell FP8 server
Replicates the nunchaku service interface with heartbeat registration
"""

import os
import time
import uuid
import sys
import asyncio
import aiohttp
import requests
import logging
import io
import base64
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our Flux Schnell FP8 implementation
from flux_schnell_fp8_standalone import FluxSchnellFP8

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageRequest(BaseModel):
    prompts: List[str] = ["a photo of an astronaut riding a horse on mars"]
    width: int = 1024
    height: int = 1024
    steps: int = 4
    seed: int | None = None
    safety_checker_adj: float = 0.5  # Controls sensitivity of NSFW detection

# Global variables
flux_generator = None

# Function to get public IP address
def get_public_ip():
    # Check if external IP is provided via environment variable
    external_ip = os.getenv("EXTERNAL_IP")
    if external_ip:
        return external_ip
    
    # Fallback to detecting public IP
    try:
        response = requests.get('https://api.ipify.org')
        return response.text
    except:
        return None

# Heartbeat function
async def send_heartbeat():
    public_ip = await asyncio.get_event_loop().run_in_executor(None, get_public_ip)
    if public_ip:
        try:
            port = int(os.getenv("PORT", "8765"))
            url = f"http://{public_ip}:{port}"
            service_type = os.getenv("SERVICE_TYPE", "flux")  # Get service type from environment variable
            
            logger.info(f"Attempting to register heartbeat with URL: {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post('https://image.pollinations.ai/register', json={'url': url, 'type': service_type}) as response:
                    if response.status == 200:
                        logger.info(f"Heartbeat sent successfully. URL: {url}")
                    else:
                        logger.error(f"Failed to send heartbeat. Status code: {response.status}")
                        # Log response content for debugging
                        try:
                            response_text = await response.text()
                            logger.error(f"Response content: {response_text}")
                        except:
                            pass
        except Exception as e:
            logger.error(f"Error sending heartbeat: {str(e)}")

# Periodic heartbeat function
async def periodic_heartbeat():
    while True:
        try:
            await send_heartbeat()
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
        except asyncio.CancelledError:
            logger.info("Heartbeat task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in periodic heartbeat: {str(e)}")
            await asyncio.sleep(5)  # Wait a bit before retrying

def scale_to_max_pixels(width: int, height: int, max_pixels: int = 512 * 512) -> tuple[int, int]:
    """Scale dimensions down proportionally if they exceed max_pixels while maintaining aspect ratio."""
    current_pixels = width * height
    
    if current_pixels <= max_pixels:
        return width, height
    
    # Calculate scaling factor to fit within max_pixels
    scale_factor = (max_pixels / current_pixels) ** 0.5
    
    # Apply scaling and round to integers
    scaled_width = int(width * scale_factor)
    scaled_height = int(height * scale_factor)
    
    return scaled_width, scaled_height

# Import the real safety checker
from safety_checker.censor import check_safety

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global flux_generator
    heartbeat_task = None
    try:
        print("Loading Flux Schnell FP8 pipeline...")
        
        # Get GPU ID from environment variable
        gpu_id = int(os.getenv("GPU_ID", "0"))
        
        # Initialize our Flux Schnell FP8 generator
        flux_generator = FluxSchnellFP8(
            device="cuda", 
            gpu_id=gpu_id, 
            compile_model=False  # Disable compilation for faster startup
        )
        flux_generator.load_models()
        print("Flux Schnell FP8 pipeline loaded successfully")
        
        # Send initial heartbeat and start periodic task
        try:
            await send_heartbeat()
            logger.info("Initial heartbeat sent successfully")
            # Store the task in app.state to prevent garbage collection
            heartbeat_task = asyncio.create_task(periodic_heartbeat())
            app.state.heartbeat_task = heartbeat_task
            logger.info("Periodic heartbeat task started")
        except Exception as e:
            logger.error(f"Error in heartbeat initialization: {str(e)}")
            if heartbeat_task:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
            raise
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        raise

    try:
        yield  # Server is running
    finally:
        # Shutdown
        if hasattr(app.state, "heartbeat_task"):
            app.state.heartbeat_task.cancel()
            try:
                await app.state.heartbeat_task
            except asyncio.CancelledError:
                pass

app = FastAPI(title="Flux Schnell FP8 Image Generation API", lifespan=lifespan)

@app.post("/generate")
async def generate(request: ImageRequest):
    print(f"Request: {request}")
    if flux_generator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    seed = request.seed if request.seed is not None else int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")
    
    # Scale down if needed to fit within 512x512 pixel limit
    width, height = scale_to_max_pixels(request.width, request.height)
    original_pixels = request.width * request.height
    final_pixels = width * height
    print(f"Original dimensions: {request.width}x{request.height} ({original_pixels:,} pixels)")
    print(f"Final dimensions: {width}x{height} ({final_pixels:,} pixels)")
    if original_pixels > 512 * 512:
        print(f"Scaled down from {original_pixels:,} to {final_pixels:,} pixels (max: {512*512:,})")

    try:
        # Generate image using our Flux Schnell FP8 implementation
        images = flux_generator.generate(
            prompt=request.prompts[0],
            width=width,
            height=height,
            num_steps=request.steps,
            seed=seed,
            num_images=1,
        )
        
        image = images[0]  # Get the first (and only) image
        
        # Check for NSFW content
        concepts, has_nsfw = check_safety([image], request.safety_checker_adj)
        
        # Convert image to base64
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=95)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        
        response_content = [{
            "image": img_base64,
            "has_nsfw_concept": has_nsfw[0],
            "concept": concepts[0],
            "width": width,
            "height": height,
            "seed": seed,
            "prompt": request.prompts[0]
        }]
        
        # Send heartbeat after successful generation
        await send_heartbeat()
        return JSONResponse(content=response_content)
        
    except Exception as e:
        logger.error(f"Error during image generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": flux_generator is not None}

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "Flux Schnell FP8 Image Generation",
        "version": "1.0.0",
        "endpoints": ["/generate", "/health"],
        "compatible_with": "pollinations/nunchaku"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8765"))
    uvicorn.run(app, host="0.0.0.0", port=port)
