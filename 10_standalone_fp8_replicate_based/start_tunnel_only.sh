#!/bin/bash
# Start only the SSH tunnel in a completely detached way
# Usage: ./start_tunnel_only.sh [PORT]

PORT=${1:-8765}
TUNNEL_HOST=${TUNNEL_HOST:-163.172.149.24}
SSH_KEY=${SSH_KEY:-/home/ionet_baremetal/thomashkey}

echo "Starting detached SSH tunnel for port $PORT..."

# Kill any existing tunnel for this port
pkill -f "ssh.*$TUNNEL_HOST.*:$PORT:"

# Start completely detached SSH tunnel
nohup ssh -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -o TCPKeepAlive=yes \
    -o BatchMode=yes \
    -f \
    -i "$SSH_KEY" \
    -R 0.0.0.0:$PORT:localhost:$PORT \
    -N ubuntu@$TUNNEL_HOST > /tmp/ssh_tunnel_$PORT.log 2>&1 &

# Give it a moment to establish
sleep 2

# Check if tunnel is working
if ss -tln | grep ":$PORT " > /dev/null; then
    echo "✅ SSH tunnel established on port $PORT"
    echo "📋 Log file: /tmp/ssh_tunnel_$PORT.log"
else
    echo "❌ Failed to establish tunnel on port $PORT"
    echo "📋 Check log: /tmp/ssh_tunnel_$PORT.log"
fi
