#!/bin/bash
# Script to start 2 Flux servers per machine (one per GPU) on all io4090 instances
# Total: 4 machines × 2 GPUs = 8 servers
# Usage: ./start_all_servers_dual_gpu.sh

set -e  # Exit on any error

echo "🚀 Starting 2 Flux servers per machine on all io4090 instances..."

# Define instances and port ranges
INSTANCES=("io4090-1" "io4090-2" "io4090-3" "io4090-4")
BASE_PORTS=(15400 15402 15404 15406)  # Each machine gets 2 consecutive ports

# Function to kill existing servers and tunnels on an instance
cleanup_instance() {
    local instance=$1
    echo "🧹 Cleaning up existing servers on $instance..."
    
    # Kill screen sessions
    ssh "$instance" 'screen -S flux-server-gpu0 -X quit' 2>/dev/null || true
    ssh "$instance" 'screen -S flux-server-gpu1 -X quit' 2>/dev/null || true
    
    # Kill SSH tunnels
    ssh "$instance" 'pkill -f "ssh.*1540[0-7]"' 2>/dev/null || true
    
    # Brief pause
    sleep 1
}

# Start servers on each instance
for i in "${!INSTANCES[@]}"; do
    instance="${INSTANCES[$i]}"
    base_port="${BASE_PORTS[$i]}"
    port_gpu0=$base_port
    port_gpu1=$((base_port + 1))
    
    echo "🖥️  Setting up $instance with ports $port_gpu0 (GPU0) and $port_gpu1 (GPU1)..."
    
    # Cleanup existing processes
    cleanup_instance "$instance"
    
    # Setup SSH tunnels for both GPUs
    echo "  🔗 Setting up SSH tunnels..."
    ssh "$instance" "cd flux-optimizations/10_standalone_fp8_replicate_based && ./setup_tunnel.sh $port_gpu0 0" &
    ssh "$instance" "cd flux-optimizations/10_standalone_fp8_replicate_based && ./setup_tunnel.sh $port_gpu1 1" &
    
    # Wait for tunnels to establish
    wait
    sleep 2
    
    # Start server for GPU 0
    echo "  🎯 Starting server for GPU 0 (port $port_gpu0)..."
    ssh "$instance" "screen -dmS flux-server-gpu0 bash -c 'cd flux-optimizations/10_standalone_fp8_replicate_based && source venv/bin/activate && export PORT=$port_gpu0 && export GPU_ID=0 && export SERVICE_TYPE=flux && export EXTERNAL_IP=163.172.149.24 && python3 server.py; exec bash'"
    
    # Start server for GPU 1
    echo "  🎯 Starting server for GPU 1 (port $port_gpu1)..."
    ssh "$instance" "screen -dmS flux-server-gpu1 bash -c 'cd flux-optimizations/10_standalone_fp8_replicate_based && source venv/bin/activate && export PORT=$port_gpu1 && export GPU_ID=1 && export SERVICE_TYPE=flux && export EXTERNAL_IP=163.172.149.24 && python3 server.py; exec bash'"
    
    echo "  ✅ Servers started on $instance"
    echo "    📺 GPU 0: ssh $instance 'screen -r flux-server-gpu0'"
    echo "    📺 GPU 1: ssh $instance 'screen -r flux-server-gpu1'"
    echo "    🌐 URLs: http://163.172.149.24:$port_gpu0 | http://163.172.149.24:$port_gpu1"
    echo ""
    
    # Brief pause between instances
    sleep 3
done

echo "🎉 All servers started!"
echo ""
echo "📋 Server URLs (8 total servers):"
for i in "${!INSTANCES[@]}"; do
    instance="${INSTANCES[$i]}"
    base_port="${BASE_PORTS[$i]}"
    port_gpu0=$base_port
    port_gpu1=$((base_port + 1))
    echo "- $instance GPU 0: http://163.172.149.24:$port_gpu0"
    echo "- $instance GPU 1: http://163.172.149.24:$port_gpu1"
done
echo ""
echo "💡 Management commands:"
echo "- Check screen sessions: ssh <instance> 'screen -list'"
echo "- Attach to GPU 0 server: ssh <instance> 'screen -r flux-server-gpu0'"
echo "- Attach to GPU 1 server: ssh <instance> 'screen -r flux-server-gpu1'"
echo "- Detach from screen: Ctrl+A, then D"
echo "- Kill GPU 0 server: ssh <instance> 'screen -S flux-server-gpu0 -X quit'"
echo "- Kill GPU 1 server: ssh <instance> 'screen -S flux-server-gpu1 -X quit'"
echo ""
echo "🔍 Health check all servers:"
echo "for port in 15400 15401 15402 15403 15404 15405 15406 15407; do echo \"Port \$port:\"; curl -s http://163.172.149.24:\$port/health; done"
