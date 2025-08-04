#!/bin/bash
# scripts/start-system.sh

echo "Starting Load Balancer System..."

# Create data directory
mkdir -p data

# Start core services first (3 model servers always running)
echo "Starting core services..."
docker-compose up -d model-server-1 model-server-2 model-server-3 ml-predictor

# Wait for model servers to be ready
echo "Waiting for model servers to start..."
sleep 10

# Start management services
echo "Starting management services..."
docker-compose up -d server-manager

# Wait for server manager
sleep 5

# Start load balancer
echo "Starting load balancer..."
docker-compose up -d load-balancer

# Wait for load balancer
sleep 5

# Start API gateway
echo "Starting API gateway..."
docker-compose up -d api-gateway

# Wait for API gateway
sleep 5

# Start dashboard
echo "Starting dashboard..."
docker-compose up -d dashboard

echo ""
echo "🚀 System started successfully!"
echo ""
echo "Services available at:"
echo "📊 Dashboard: http://localhost:3000"
echo "🌐 API Gateway: http://localhost:8080"
echo "⚖️  Load Balancer: http://localhost:8082"
echo "🖥️  Server Manager: http://localhost:8083"
echo "🤖 ML Predictor: http://localhost:5000"
echo "🖥️  Model Server 1: http://localhost:8084"
echo "🖥️  Model Server 2: http://localhost:8085"
echo "🖥️  Model Server 3: http://localhost:8086"
echo ""
echo "To view logs: docker-compose logs -f [service-name]"
echo "To stop system: ./scripts/stop-system.sh"

