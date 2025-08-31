#!/bin/bash
# Start multiple Pollinations-compatible Flux Schnell FP8 servers for 8 GPUs
# Each server runs on a different port with SSH tunnel
# Usage: ./start_servers.sh [BASE_PORT]
# Example: ./start_servers.sh 14400

# Parse command line arguments
if [ $# -ge 1 ]; then
    BASE_PORT=$1
else
    BASE_PORT=${BASE_PORT:-14400}
fi

# Configuration
TUNNEL_HOST=${TUNNEL_HOST:-163.172.149.24}
SSH_KEY=${SSH_KEY:-/home/ionet_baremetal/thomashkey}
SERVICE_TYPE=${SERVICE_TYPE:-flux}
NUM_GPUS=8

# Arrays to store process IDs
declare -a SERVER_PIDS
declare -a SSH_PIDS

echo "Starting $NUM_GPUS Flux Schnell FP8 servers..."
echo "Base port: $BASE_PORT"
echo "Tunnel host: $TUNNEL_HOST"

# Function to cleanup all processes
cleanup() {
    echo "Cleaning up all servers and SSH tunnels..."
    
    # Kill all server processes
    for pid in "${SERVER_PIDS[@]}"; do
        if [ ! -z "$pid" ]; then
            echo "Stopping server PID: $pid"
            kill $pid 2>/dev/null
        fi
    done
    
    # Kill all SSH tunnel processes
    for pid in "${SSH_PIDS[@]}"; do
        if [ ! -z "$pid" ]; then
            echo "Stopping SSH tunnel PID: $pid"
            kill $pid 2>/dev/null
        fi
    done
    
    # Wait for processes to terminate
    sleep 2
    
    # Force kill if still running
    for pid in "${SERVER_PIDS[@]}" "${SSH_PIDS[@]}"; do
        if [ ! -z "$pid" ]; then
            kill -9 $pid 2>/dev/null
        fi
    done
    
    echo "Cleanup complete."
    exit 0
}

# Trap signals for cleanup
trap cleanup SIGINT SIGTERM

# Start servers for each GPU
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    port=$((BASE_PORT + gpu_id))
    
    echo "Starting server for GPU $gpu_id on port $port..."
    
    # Start SSH tunnel in background
    nohup ssh -o StrictHostKeyChecking=no \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        -o TCPKeepAlive=yes \
        -f \
        -i "$SSH_KEY" \
        -R 0.0.0.0:$port:localhost:$port \
        -N ubuntu@$TUNNEL_HOST > /tmp/ssh_tunnel_$port.log 2>&1 &
    ssh_pid=$!
    SSH_PIDS[$gpu_id]=$ssh_pid
    echo "  SSH tunnel PID: $ssh_pid (log: /tmp/ssh_tunnel_$port.log)"
    
    # Wait for tunnel to establish
    sleep 2
    
    # Start server in background
    PORT=$port GPU_ID=$gpu_id SERVICE_TYPE=$SERVICE_TYPE EXTERNAL_IP=$TUNNEL_HOST \
        nohup python3 server.py > /tmp/flux_server_gpu${gpu_id}_port${port}.log 2>&1 &
    server_pid=$!
    SERVER_PIDS[$gpu_id]=$server_pid
    echo "  Server PID: $server_pid (log: /tmp/flux_server_gpu${gpu_id}_port${port}.log)"
    
    # Brief pause between server starts to avoid resource conflicts
    sleep 5
done

echo ""
echo "All servers started successfully!"
echo "Server details:"
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    port=$((BASE_PORT + gpu_id))
    echo "  GPU $gpu_id: http://$TUNNEL_HOST:$port (local: http://localhost:$port)"
    echo "    Server PID: ${SERVER_PIDS[$gpu_id]}"
    echo "    SSH PID: ${SSH_PIDS[$gpu_id]}"
    echo "    Logs: /tmp/flux_server_gpu${gpu_id}_port${port}.log"
done

echo ""
echo "To monitor logs: tail -f /tmp/flux_server_gpu*_port*.log"
echo "To stop all servers: Press Ctrl+C or run: kill ${SERVER_PIDS[*]} ${SSH_PIDS[*]}"
echo ""
echo "Waiting for servers (Press Ctrl+C to stop all)..."

# Wait for all background processes
wait
