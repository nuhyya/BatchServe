#!/bin/bash
# scripts/stop-system.sh

echo "Stopping Load Balancer System..."

docker-compose down

echo "System stopped successfully!"

