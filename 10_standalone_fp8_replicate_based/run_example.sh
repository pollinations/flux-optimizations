#!/bin/bash
# Example script to run Flux Schnell FP8 standalone implementation

echo "🚀 Running Flux Schnell FP8 Standalone Example"
echo "=============================================="

# Set single GPU usage (important for cluster environments)
export CUDA_VISIBLE_DEVICES=0

# Run with example prompt
python3 flux_schnell_fp8_standalone.py \
    --prompt "a majestic cat wearing a wizard hat, sitting on ancient books, magical atmosphere, detailed digital art" \
    --width 1024 \
    --height 1024 \
    --steps 4 \
    --num-images 1 \
    --output-dir ./outputs \
    --no-compile

echo "✅ Generation complete! Check ./outputs/ for results."
