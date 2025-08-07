# BatchServe

A **smart load balancer** that routes ML model requests to different servers based on their current load and predicted resource usage. This system demonstrates load balancing, auto-scaling, circuit breaker patterns, and ML-based prediction.

## System Architecture
<img width="1231" height="566" alt="Arch" src="https://github.com/user-attachments/assets/61808091-dd3d-4264-b571-c26bcb843c4f" />



## Project Structure

```
BatchServe/
├── docker-compose.yml          # Container orchestration
├── README.md                   # This file
├── dashboard/                  # React frontend
│   ├── Dockerfile
│   ├── package.json
│   └── src/
│       ├── App.css
│       ├── App.js
│       ├── index.js
│       ├── components/
│       │   ├── Dashboard.js
│       │   ├── Metrics.js
│       │   ├── Servers.js
│       │   └── Traffic.js
│       └── utils/
│           └── helpers.js
├── api-gateway/                # Request batching and routing
│   ├── Dockerfile
│   ├── pom.xml
│   └── src/main/java/
├── load-balancer/              # Smart load balancing logic
│   ├── Dockerfile
│   ├── pom.xml
│   └── src/main/java/
├── server-manager/             # Server monitoring and auto-scaling
│   ├── Dockerfile
│   ├── pom.xml
│   └── src/main/java/
├── servers/                    # Model server instances
│   ├── Dockerfile
│   ├── pom.xml
│   └── src/main/java/
├── load-predictor/             # ML prediction service
│   ├── Dockerfile
│   ├── app.py
│   └── requirements.txt
├── scripts/                    # Utility scripts
    ├── build-all.sh
    ├── start-system.sh
    ├── stop-system.sh
    └── scale-up.sh

```

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Java 17+ (for local development)
- Node.js 18+ (for dashboard development)
- Python 3.9+ (for ML predictor development)

### 1. Build All Services
```bash
chmod +x scripts/*.sh
./scripts/build-all.sh
```

### 2. Start the System
```bash
./scripts/start-system.sh
```

### 3. Access the Dashboard
Open http://localhost:3000 in your browser

## Key Features

### Smart Routing
- **Resource-Based**: Routes requests to servers with lowest CPU/GPU/Memory usage
- **Model Awareness**: Only routes to servers that have the required model loaded
- **ML Prediction**: Uses historical data to predict resource usage

### Fault Tolerance
- **Circuit Breaker**: Automatically stops sending requests to failed servers
- **Health Monitoring**: Continuous health checks every 5 seconds
- **Auto Recovery**: Failed servers automatically rejoin when healthy

### Resource Simulation
- **Realistic CPU Usage**: Actually burns CPU cycles based on token count
- **Memory Allocation**: Physically allocates memory when models are loaded
- **Processing Time**: Variable processing time based on model type and tokens

### Auto-Scaling
- **Demand-Based**: Automatically starts additional servers when request count exceeds threshold
- **Model-Specific**: Loads required models on scaled servers
- **Resource-Aware**: Considers current server resource usage

## Using the Dashboard

### Traffic Patterns
- **High Model A Traffic**: 70% Model A, 10% each for B, C, D
- **Equal Distribution**: 25% each model type
- **Burst Traffic**: Random high-volume requests

### Server Controls
- **Load Model**: Dynamically load models A, B, C, or D on any server
- **Unload Model**: Remove models to free up memory
- **Real-time Monitoring**: Live CPU, GPU, Memory usage graphs

### Metrics
- **Request Counts**: Total requests processed per model type
- **Response Times**: Average response time per server
- **Resource Usage**: Real-time resource utilization

## ML Predictor

The ML component uses scikit-learn to predict resource usage:

### Training Data
- **Synthetic Dataset**: 1000+ samples based on realistic model formulas
- **Features**: Request counts and token counts per model type
- **Targets**: CPU, GPU, Memory usage percentages

### Model Formulas (Simulated)
- **Model A**: 15% base CPU + 5% per 100 tokens
- **Model B**: 20% base CPU + 6% per 100 tokens  
- **Model C**: 25% base CPU + 7% per 100 tokens
- **Model D**: 18% base CPU + 5.5% per 100 tokens

### Prediction API
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "modelCounts": {"A": 3, "B": 2, "C": 1, "D": 0},
    "tokenCounts": {"A": 300, "B": 160, "C": 120, "D": 0}
  }'
```

## Monitoring and Debugging

### Health Endpoints
- API Gateway: http://localhost:8080/api/metrics
- Load Balancer: http://localhost:8082/api/health
- Server Manager: http://localhost:8083/api/servers/status
- ML Predictor: http://localhost:5000/health
- Model Servers: http://localhost:8084-8089/api/status

### Logs
```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f api-gateway
docker-compose logs -f load-balancer
docker-compose logs -f server-manager
```

### Data Files
Server status and metrics are persisted in `./data/servers.json`

## Development

### Local Development
Each service can be run locally for development:

```bash
# API Gateway
cd api-gateway && mvn spring-boot:run

# Load Balancer  
cd load-balancer && mvn spring-boot:run

# Server Manager
cd server-manager && mvn spring-boot:run

# Model Server
cd servers && SERVER_ID=server-1 SERVER_PORT=8084 mvn spring-boot:run

# ML Predictor
cd load-predictor && python app.py

# Dashboard
cd dashboard && npm start
```

### Scaling Additional Servers
```bash
./scripts/scale-up.sh
```

### Testing
```bash
# Quick system test
./scripts/quick-test.sh

# Generate test traffic
curl -X POST http://localhost:8080/api/test-traffic/high-a
curl -X POST http://localhost:8080/api/test-traffic/equal
curl -X POST http://localhost:8080/api/test-traffic/burst
```

## Configuration

### Environment Variables
- `SERVER_ID`: Unique identifier for model servers
- `SERVER_PORT`: Port for model servers
- `LOAD_BALANCER_URL`: Load balancer endpoint
- `SERVER_MANAGER_URL`: Server manager endpoint
- `ML_PREDICTOR_URL`: ML predictor endpoint


## System Flow

1. **Dashboard** - user submits requests
2. **API Gateway** - batches requests (200ms or 15 requests)
3. **Load Balancer** - gets server status and ML predictions
4. **Routing** - happens based on server load and model availability
5. **Model Servers** - process requests and update resource usage
6. **Server Manager** - monitors health and triggers auto-scaling
7. **Dashboard** - displays real-time metrics and status

## Learning Objectives

This project demonstrates:
- **Load Balancing Algorithms**: Weighted round-robin based on resource usage
- **Circuit Breaker Pattern**: Fault tolerance for distributed systems
- **Auto-scaling**: Dynamic resource allocation based on demand
- **ML Integration**: Using machine learning for system optimization
- **Microservices Architecture**: Service decomposition and communication
- **Real-time Monitoring**: Live dashboards and metrics collection
- **Container Orchestration**: Docker Compose for multi-service deployment

## 🏁 Stopping the System

```bash
./scripts/stop-system.sh
```

---

