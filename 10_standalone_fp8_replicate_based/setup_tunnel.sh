#!/bin/bash
# Setup persistent SSH tunnel for Flux server
# Usage: ./setup_tunnel.sh [PORT] [GPU_ID]

# Parse command line arguments
if [ $# -ge 1 ]; then
    PORT_ARG=$1
else
    PORT_ARG=${PORT:-14400}
fi

if [ $# -ge 2 ]; then
    GPU_ARG=$2
else
    GPU_ARG=${GPU_ID:-0}
fi

# Set environment variables
export PORT=$PORT_ARG
export GPU_ID=$GPU_ARG
export TUNNEL_HOST=${TUNNEL_HOST:-163.172.149.24}
export SSH_KEY=${SSH_KEY:-$HOME/.ssh/thomashkey}

echo "Setting up persistent SSH tunnel..."
echo "Port: $PORT"
echo "Tunnel Host: $TUNNEL_HOST"

# Kill any existing SSH tunnels for this port
pkill -f "ssh.*$TUNNEL_HOST.*$PORT" 2>/dev/null || true

# Start persistent SSH tunnel
echo "Creating SSH tunnel to $TUNNEL_HOST:$PORT..."
ssh -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -o TCPKeepAlive=yes \
    -f \
    -N \
    -i "$SSH_KEY" \
    -R 0.0.0.0:$PORT:localhost:$PORT \
    ubuntu@$TUNNEL_HOST

if [ $? -eq 0 ]; then
    echo "✅ SSH tunnel established successfully"
    echo "🌐 Server will be accessible at: http://$TUNNEL_HOST:$PORT"
else
    echo "❌ Failed to establish SSH tunnel"
    exit 1
fi
