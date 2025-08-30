#!/bin/bash
# Start multiple Flux Schnell FP8 servers on different GPUs and ports
# Usage: ./start_multi_gpu_servers.sh [BASE_PORT] [NUM_GPUS]
# Example: ./start_multi_gpu_servers.sh 8765 4

# Parse command line arguments
BASE_PORT=${1:-8765}
NUM_GPUS=${2:-4}

echo "Starting $NUM_GPUS Flux servers starting from port $BASE_PORT..."

# Array to store background process IDs
declare -a SERVER_PIDS=()

# Function to cleanup on exit
cleanup() {
    echo "Stopping all servers..."
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping server with PID: $pid"
            kill "$pid"
        fi
    done
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start servers on different GPUs and ports
for ((i=0; i<NUM_GPUS; i++)); do
    PORT=$((BASE_PORT + i))
    GPU_ID=$i
    
    echo "Starting server on GPU $GPU_ID, port $PORT..."
    
    # Start server in background
    ./start_server.sh $PORT $GPU_ID &
    SERVER_PID=$!
    SERVER_PIDS+=($SERVER_PID)
    
    echo "Server started with PID: $SERVER_PID (GPU: $GPU_ID, Port: $PORT)"
    
    # Wait a bit between starts to avoid conflicts
    sleep 5
done

echo "All $NUM_GPUS servers started!"
echo "Ports: $BASE_PORT to $((BASE_PORT + NUM_GPUS - 1))"
echo "GPUs: 0 to $((NUM_GPUS - 1))"
echo "Press Ctrl+C to stop all servers"

# Wait for all background processes
wait
