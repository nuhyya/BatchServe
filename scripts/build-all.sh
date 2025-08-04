#!/bin/bash
# scripts/build-all.sh

echo "Building all services..."

# Build Java services
echo "Building API Gateway..."
cd api-gateway
mvn clean package -DskipTests
cd ..

echo "Building Load Balancer..."
cd load-balancer
mvn clean package -DskipTests
cd ..

echo "Building Server Manager..."
cd server-manager
mvn clean package -DskipTests
cd ..

echo "Building Model Servers..."
cd servers
mvn clean package -DskipTests
cd ..

# Build React Dashboard
echo "Building Dashboard..."
cd dashboard
npm install
npm run build
cd ..

echo "All services built successfully!"

