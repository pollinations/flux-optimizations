#!/bin/bash
# Script to copy SSH keys to all io4090 instances
# Usage: ./copy_ssh_keys.sh

set -e  # Exit on any error

echo "🔑 Copying SSH keys to all io4090 instances..."

# Define instances
INSTANCES=("io4090-1" "io4090-2" "io4090-3" "io4090-4")
SSH_KEY_PATH="$HOME/.ssh/thomashkey"

# Check if SSH key exists
if [ ! -f "$SSH_KEY_PATH" ]; then
    echo "❌ SSH key not found at $SSH_KEY_PATH"
    exit 1
fi

echo "📋 Found SSH key at: $SSH_KEY_PATH"

# Copy SSH key to each instance
for instance in "${INSTANCES[@]}"; do
    echo "📤 Processing $instance..."
    
    # Copy key to tmp first
    echo "  Copying key to /tmp/thomashkey..."
    scp "$SSH_KEY_PATH" "$instance:/tmp/thomashkey"
    
    # Move to final location with correct permissions
    echo "  Setting up key at ~/.ssh/thomashkey..."
    ssh "$instance" "mkdir -p ~/.ssh && \
                     mv /tmp/thomashkey ~/.ssh/thomashkey && \
                     chmod 600 ~/.ssh/thomashkey"
    
    # Test SSH connection to tunnel host
    echo "  Testing SSH tunnel connectivity..."
    if ssh "$instance" "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i ~/.ssh/thomashkey ubuntu@163.172.149.24 'echo SSH connection successful'" > /dev/null 2>&1; then
        echo "  ✅ $instance: SSH key installed and tunnel connectivity verified"
    else
        echo "  ⚠️  $instance: SSH key installed but tunnel connectivity failed"
    fi
    
    echo ""
done

echo "🎉 SSH key deployment complete!"
echo ""
echo "💡 Next steps:"
echo "1. Install screen on any instances that need it: ssh <instance> 'sudo apt install -y screen'"
echo "2. Start servers in screen sessions: ssh <instance> 'screen -dmS flux-server bash -c \"cd flux-optimizations/10_standalone_fp8_replicate_based && source venv/bin/activate && ./start_server.sh <port> <gpu_id>\"'"
echo ""
echo "📋 Suggested port assignments:"
echo "- io4090-1: port 14400, GPU 0"
echo "- io4090-2: port 14401, GPU 0" 
echo "- io4090-3: port 14402, GPU 0"
echo "- io4090-4: port 14403, GPU 0"
