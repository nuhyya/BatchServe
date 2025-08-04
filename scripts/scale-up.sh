#!/bin/bash
# scripts/scale-up.sh

echo "Scaling up additional model servers..."

# Start additional servers for auto-scaling
docker-compose --profile scaling up -d model-server-4 model-server-5 model-server-6

echo "Additional servers started!"
echo "🖥️  Model Server 4: http://localhost:8087"
echo "🖥️  Model Server 5: http://localhost:8088"
echo "🖥️  Model Server 6: http://localhost:8089"

