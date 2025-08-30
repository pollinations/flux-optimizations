#!/bin/bash
# Start multiple SSH tunnels using a single SSH connection
# Usage: ./start_multi_tunnels.sh [BASE_PORT] [NUM_PORTS]
# Example: ./start_multi_tunnels.sh 24106 4

BASE_PORT=${1:-24106}
NUM_PORTS=${2:-4}
TUNNEL_HOST=${TUNNEL_HOST:-163.172.149.24}
SSH_KEY=${SSH_KEY:-/home/ionet_baremetal/thomashkey}

echo "Starting multi-tunnel SSH connection to $TUNNEL_HOST..."
echo "Base port: $BASE_PORT, Number of ports: $NUM_PORTS"

# Kill any existing SSH tunnels
pkill -f "ssh.*$TUNNEL_HOST"

# Build the port forwarding arguments
PORT_FORWARDS=""
for ((i=0; i<NUM_PORTS; i++)); do
    PORT=$((BASE_PORT + i))
    PORT_FORWARDS="$PORT_FORWARDS -R 0.0.0.0:$PORT:localhost:$PORT"
    echo "  Port $PORT -> localhost:$PORT"
done

# Start single SSH connection with multiple port forwards
echo "Establishing SSH connection with multiple tunnels..."
nohup ssh -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -o TCPKeepAlive=yes \
    -o BatchMode=yes \
    -f \
    -i "$SSH_KEY" \
    $PORT_FORWARDS \
    -N ubuntu@$TUNNEL_HOST > /tmp/ssh_multi_tunnel.log 2>&1 &

SSH_PID=$!
echo "Multi-tunnel SSH connection started with PID: $SSH_PID"
echo "Log file: /tmp/ssh_multi_tunnel.log"

# Wait for tunnels to establish
sleep 3

# Check which tunnels are working
echo "Checking tunnel status..."
for ((i=0; i<NUM_PORTS; i++)); do
    PORT=$((BASE_PORT + i))
    if ss -tln | grep ":$PORT " > /dev/null; then
        echo "  ✅ Port $PORT: Active"
    else
        echo "  ❌ Port $PORT: Failed"
    fi
done

echo "Multi-tunnel setup complete!"
