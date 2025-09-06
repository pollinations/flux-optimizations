#!/bin/bash
# Easy server monitoring script for all 8 Flux servers
# Usage: ./check_servers.sh [logs|health|status|all]

COMMAND=${1:-all}

echo "🔍 Flux Server Monitor - $(date)"
echo "=================================="

check_health() {
    echo "🏥 Health Check - All 8 Servers:"
    echo "--------------------------------"
    for port in 15400 15401 15402 15403 15404 15405 15406 15407; do
        printf "Port %s: " "$port"
        curl -s http://163.172.149.24:$port/health | jq -r '.status + " - model_loaded: " + (.model_loaded|tostring)' 2>/dev/null || echo "❌ Failed to connect"
    done
    echo ""
}

check_status() {
    echo "📺 Screen Sessions Status:"
    echo "-------------------------"
    for instance in io4090-1 io4090-2 io4090-3 io4090-4; do
        echo "=== $instance ==="
        ssh "$instance" 'screen -list' 2>/dev/null || echo "No screen sessions"
        echo ""
    done
}

check_logs() {
    echo "📋 Recent Server Logs (last 5 lines each):"
    echo "------------------------------------------"
    for instance in io4090-1 io4090-2 io4090-3 io4090-4; do
        echo "=== $instance GPU 0 ==="
        ssh "$instance" 'tail -5 /tmp/flux_server.log 2>/dev/null' || echo "No logs found"
        echo ""
        echo "=== $instance GPU 1 ==="
        ssh "$instance" 'tail -5 /tmp/flux_server.log 2>/dev/null' || echo "No logs found"
        echo ""
    done
}

show_log_commands() {
    echo "💡 Log Monitoring Commands:"
    echo "--------------------------"
    echo "# Watch logs in real-time for specific instance:"
    echo "ssh io4090-1 'tail -f /tmp/flux_server.log'"
    echo ""
    echo "# Attach to specific server screen session:"
    echo "ssh -t io4090-1 'screen -r flux-server-gpu0'  # GPU 0"
    echo "ssh -t io4090-1 'screen -r flux-server-gpu1'  # GPU 1"
    echo ""
    echo "# Check server processes:"
    echo "ssh io4090-1 'ps aux | grep python3 | grep server.py'"
    echo ""
    echo "# Check SSH tunnels:"
    echo "ssh io4090-1 'ps aux | grep ssh | grep 1540'"
    echo ""
    echo "# Monitor all servers continuously:"
    echo "watch -n 5 './check_servers.sh health'"
    echo ""
}

case $COMMAND in
    "health")
        check_health
        ;;
    "status")
        check_status
        ;;
    "logs")
        check_logs
        ;;
    "commands")
        show_log_commands
        ;;
    "all")
        check_health
        check_status
        show_log_commands
        ;;
    *)
        echo "Usage: $0 [health|status|logs|commands|all]"
        echo ""
        echo "  health   - Check health endpoints of all 8 servers"
        echo "  status   - Show screen session status on all instances"
        echo "  logs     - Show recent logs from all servers"
        echo "  commands - Show useful log monitoring commands"
        echo "  all      - Show health, status, and commands (default)"
        ;;
esac
