#!/bin/bash

echo "🚀 Starting Computer Use Agent..."

# Load environment variables
if [ -f ".env" ]; then
    echo "📋 Loading environment variables..."
    set -a  # Export all variables automatically
    source .env
    set +a  # Stop auto export
    echo "✅ Environment variables loaded"
else
    echo "⚠️  .env file not found"
fi

# Define colors
BLUE=$(printf '\033[0;34m')
GREEN=$(printf '\033[0;32m')
NC=$(printf '\033[0m')

# Start backend service
echo "🔧 Launching backend service..."
python3 backend.py  &

# Give backend time to start
sleep 3

# Start static front-end server
echo "🎨 Launching static front-end server..."
cd static || { echo "❌ Unable to enter static directory"; exit 1; }
python3 -m http.server 8001 --bind 127.0.0.1 &

# Start Nginx (comment out locally if you don't need it)
echo "🌐 Starting Nginx..."
sudo nginx

echo "✅ Services running!"
echo "📱 Visit: http://localhost:7860"
echo ""

echo "Press Ctrl+C to stop all services..."

# Wait for user interrupt
trap "echo '🛑 Stopping services...'; sudo nginx -s stop; pkill -P $$; exit" INT
wait
