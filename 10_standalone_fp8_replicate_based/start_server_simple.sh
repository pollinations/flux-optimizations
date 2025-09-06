#!/bin/bash
# Simple server start script without SSH tunnel management
# Usage: ./start_server_simple.sh [PORT] [GPU_ID]

# Parse command line arguments
if [ $# -ge 1 ]; then
    PORT_ARG=$1
else
    PORT_ARG=${PORT:-15400}
fi

if [ $# -ge 2 ]; then
    GPU_ARG=$2
else
    GPU_ARG=${GPU_ID:-0}
fi

# Set environment variables
export PORT=$PORT_ARG
export GPU_ID=$GPU_ARG
export SERVICE_TYPE=${SERVICE_TYPE:-flux}
export EXTERNAL_IP=${EXTERNAL_IP:-163.172.149.24}

echo "Starting Flux Schnell FP8 server..."
echo "Port: $PORT"
echo "GPU ID: $GPU_ID"
echo "Service Type: $SERVICE_TYPE"
echo "External IP: $EXTERNAL_IP"

# Start the server
echo "Starting server..."
python3 server.py
