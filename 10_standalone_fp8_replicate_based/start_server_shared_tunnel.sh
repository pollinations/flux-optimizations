#!/bin/bash
# Start server using existing shared SSH tunnel
# Usage: ./start_server_shared_tunnel.sh [PORT] [GPU_ID]
# Note: Run start_multi_tunnels.sh first to establish tunnels

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

echo "Starting Flux Schnell FP8 server (using shared tunnel)..."
echo "Port: $PORT"
echo "GPU ID: $GPU_ID"
echo "Service Type: $SERVICE_TYPE"
echo "Tunnel Host: $TUNNEL_HOST"

# Check if tunnel exists for this port
if ! ss -tln | grep ":$PORT " > /dev/null; then
    echo "❌ No tunnel found for port $PORT"
    echo "Run: ./start_multi_tunnels.sh first to establish tunnels"
    exit 1
fi

echo "✅ Using existing tunnel on port $PORT"

# Set the external IP for heartbeat
export EXTERNAL_IP=$TUNNEL_HOST

# Start the server (no tunnel management needed)
echo "Starting server..."
python3 server.py
