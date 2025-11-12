#!/bin/bash

# Script name: kill_ports.sh
# Purpose: Terminate processes using ports 8001 and 8002
# Usage: ./kill_ports.sh

# Optional: ensure the script runs with sudo/root (kill -9 may need privileges)
# Uncomment the snippet below if you want to enforce the check
#
# if [[ $EUID -ne 0 ]]; then
#    echo "Warning: run with sudo to guarantee the target processes can be terminated."
# fi

# Target ports
PORTS=(8001 8002)

# Iterate over ports and terminate the owning processes
for port in "${PORTS[@]}"; do
    echo "Checking and terminating processes using port $port..."
    lsof -t -i:$port | xargs kill -9 2>/dev/null || echo "Port $port is free or already cleaned up."
done

echo "Done."
