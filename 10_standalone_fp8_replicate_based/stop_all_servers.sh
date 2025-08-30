#!/bin/bash
# Stop all Flux servers and SSH tunnels

echo "Stopping all Flux servers and SSH tunnels..."

# Kill all Python server processes
echo "Killing Python server processes..."
pkill -f "python3 server.py"

# Kill all SSH tunnels to the tunnel host
echo "Killing SSH tunnels..."
pkill -f "ssh.*163.172.149.24"

# Wait a moment for cleanup
sleep 2

echo "All servers and tunnels stopped."
