#!/bin/bash
# Start the Pollinations-compatible Flux Schnell FP8 server with persistent autossh tunnel
# Usage: ./start_server_autossh.sh [PORT] [GPU_ID]
# Example: ./start_server_autossh.sh 8766 1

# Parse command line arguments
if [ $# -ge 1 ]; then
    PORT_ARG=$1
else
    PORT_ARG=${PORT:-8765}
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
export TUNNEL_HOST=${TUNNEL_HOST:-163.172.149.24}
export SSH_KEY=${SSH_KEY:-/home/ionet_baremetal/thomashkey}

echo "Starting Flux Schnell FP8 server with autossh tunnel..."
echo "Port: $PORT"
echo "GPU ID: $GPU_ID"
echo "Service Type: $SERVICE_TYPE"
echo "Tunnel Host: $TUNNEL_HOST"

# Kill any existing SSH tunnels
pkill -f "ssh.*$TUNNEL_HOST"
pkill -f "autossh.*$TUNNEL_HOST"

# Check if autossh is available, if not use enhanced SSH
if command -v autossh >/dev/null 2>&1; then
    echo "Using autossh for persistent tunnel to $TUNNEL_HOST:$PORT..."
    autossh -M 0 \
        -o StrictHostKeyChecking=no \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        -o TCPKeepAlive=yes \
        -i "$SSH_KEY" \
        -R 0.0.0.0:$PORT:localhost:$PORT \
        -N ubuntu@$TUNNEL_HOST &
    SSH_PID=$!
    echo "Autossh tunnel started with PID: $SSH_PID"
else
    echo "Autossh not available, using enhanced SSH with keepalive..."
    ssh -o StrictHostKeyChecking=no \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        -o TCPKeepAlive=yes \
        -i "$SSH_KEY" \
        -R 0.0.0.0:$PORT:localhost:$PORT \
        -N ubuntu@$TUNNEL_HOST &
    SSH_PID=$!
    echo "Enhanced SSH tunnel started with PID: $SSH_PID"
fi

# Wait a moment for tunnel to establish
sleep 3

# Set the external IP for heartbeat
export EXTERNAL_IP=$TUNNEL_HOST

# Start the server
echo "Starting server..."
python3 server.py

# Cleanup: kill SSH tunnel when server stops
echo "Cleaning up SSH tunnel..."
kill $SSH_PID 2>/dev/null
