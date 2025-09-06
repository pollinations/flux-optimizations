#!/bin/bash
# Script to start Flux servers in screen sessions on all io4090 instances
# Usage: ./start_all_servers.sh

set -e  # Exit on any error

echo "🚀 Starting Flux servers on all io4090 instances..."

# Define instances and ports (using different ports to avoid conflicts)
INSTANCES=("io4090-1" "io4090-2" "io4090-3" "io4090-4")
PORTS=("15400" "15401" "15402" "15403")

# Start servers on each instance
for i in "${!INSTANCES[@]}"; do
    instance="${INSTANCES[$i]}"
    port="${PORTS[$i]}"
    echo "🖥️  Starting server on $instance (port $port)..."
    
    # Check if screen session already exists
    if ssh "$instance" "screen -list | grep -q flux-server" 2>/dev/null; then
        echo "  ⚠️  Screen session 'flux-server' already exists on $instance"
        echo "  To kill existing session: ssh $instance 'screen -S flux-server -X quit'"
        continue
    fi
    
    # Setup SSH tunnel first
    ssh "$instance" "cd flux-optimizations/10_standalone_fp8_replicate_based && ./setup_tunnel.sh $port 0"
    
    # Start server in detached screen session with explicit environment variables
    ssh "$instance" "screen -dmS flux-server bash -c 'cd flux-optimizations/10_standalone_fp8_replicate_based && source venv/bin/activate && export PORT=$port && export GPU_ID=0 && export SERVICE_TYPE=flux && export EXTERNAL_IP=163.172.149.24 && python3 server.py; exec bash'"
    
    echo "  ✅ Server started on $instance (port $port)"
    echo "  📺 To attach: ssh $instance 'screen -r flux-server'"
    echo "  🌐 URL: http://163.172.149.24:$port"
    echo ""
    
    # Brief pause between starts
    sleep 2
done

echo "🎉 All servers started!"
echo ""
echo "📋 Server URLs:"
for i in "${!INSTANCES[@]}"; do
    instance="${INSTANCES[$i]}"
    port="${PORTS[$i]}"
    echo "- $instance: http://163.172.149.24:$port"
done
echo ""
echo "💡 Management commands:"
echo "- Check screen sessions: ssh <instance> 'screen -list'"
echo "- Attach to server: ssh <instance> 'screen -r flux-server'"
echo "- Detach from screen: Ctrl+A, then D"
echo "- Kill server: ssh <instance> 'screen -S flux-server -X quit'"
echo "- Check server logs: ssh <instance> 'tail -f /tmp/flux_server.log'"
