package loadbalancer.service;

import loadbalancer.model.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.*;
import java.util.stream.Collectors;

@Service
public class LoadBalancingService {

    private final RestTemplate restTemplate = new RestTemplate();
    private final CircuitBreakerService circuitBreakerService;

    private static final String SERVER_MANAGER_URL = "http://server-manager:8083";
    private static final String ML_PREDICTOR_URL = "http://ml-predictor:5000";

    // Define total capacity for normalization
    private static final double TOTAL_CPU_CAPACITY = 8000.0;    // Example: 8 vCPUs
    private static final double TOTAL_GPU_CAPACITY = 4000.0;    // Example: 4 GPUs
    private static final double TOTAL_MEMORY_CAPACITY = 16000.0; // Example: 16 GB

    @Autowired
    public LoadBalancingService(CircuitBreakerService circuitBreakerService) {
        this.circuitBreakerService = circuitBreakerService;
    }

    public void processBatch(RequestBatch batch) {
        try {
            List<ServerInfo> servers = getServerStatus();
            PredictionResponse prediction = getPrediction(batch);
            routeRequests(batch, servers, prediction);
        } catch (Exception e) {
            System.err.println("Error processing batch: " + e.getMessage());
        }
    }

    private List<ServerInfo> getServerStatus() {
        try {
            ServerInfo[] servers = restTemplate.getForObject(
                SERVER_MANAGER_URL + "/api/servers/status",
                ServerInfo[].class
            );
            return Arrays.asList(servers);
        } catch (Exception e) {
            System.err.println("Failed to get server status: " + e.getMessage());
            return new ArrayList<>();
        }
    }

    private PredictionResponse getPrediction(RequestBatch batch) {
        try {
            PredictionRequest predictionRequest = new PredictionRequest(
                batch.getModelCounts(),
                batch.getTokenCounts()
            );

            return restTemplate.postForObject(
                ML_PREDICTOR_URL + "/predict",
                predictionRequest,
                PredictionResponse.class
            );
        } catch (Exception e) {
            System.err.println("Failed to get ML prediction: " + e.getMessage());
            PredictionResponse defaultPrediction = new PredictionResponse();
            defaultPrediction.setPredictedCpuUsage(20.0);
            defaultPrediction.setPredictedGpuUsage(15.0);
            defaultPrediction.setPredictedMemoryUsage(10.0);
            return defaultPrediction;
        }
    }

    void routeRequests(RequestBatch batch, List<ServerInfo> servers, PredictionResponse prediction) {
        // Normalize predictions
        double predictedCpuPercent = (prediction.getPredictedCpuUsage() / TOTAL_CPU_CAPACITY) * 100.0;
        double predictedGpuPercent = (prediction.getPredictedGpuUsage() / TOTAL_GPU_CAPACITY) * 100.0;
        double predictedMemoryPercent = (prediction.getPredictedMemoryUsage() / TOTAL_MEMORY_CAPACITY) * 100.0;

        // Group requests by model
        Map<String, List<ModelRequest>> requestsByModel = batch.getRequests()
            .stream()
            .collect(Collectors.groupingBy(ModelRequest::getModelType));

        requestsByModel.forEach((modelType, requests) -> {
            ServerInfo bestServer = findBestServer(servers, modelType,
                predictedCpuPercent, predictedGpuPercent, predictedMemoryPercent);

            if (bestServer != null) {
                sendRequestsToServer(bestServer, requests);
            } else {
                System.err.println("No available server for model: " + modelType);
            }
        });
    }

    private ServerInfo findBestServer(
        List<ServerInfo> servers,
        String modelType,
        double predictedCpuPercent,
        double predictedGpuPercent,
        double predictedMemoryPercent
    ) {
        return servers.stream()
            .filter(circuitBreakerService::isServerAvailable)
            .filter(server -> server.getLoadedModels().contains(modelType))
            .filter(server ->
                (server.getCpuUsage() + predictedCpuPercent <= 100.0) &&
                (server.getGpuUsage() + predictedGpuPercent <= 100.0) &&
                (server.getMemoryUsage() + predictedMemoryPercent <= 100.0)
            )
            .min((s1, s2) -> {
                double score1 = s1.getCpuUsage() + s1.getGpuUsage() + s1.getMemoryUsage();
                double score2 = s2.getCpuUsage() + s2.getGpuUsage() + s2.getMemoryUsage();
                return Double.compare(score1, score2);
            })
            .orElse(null);
    }

    private void sendRequestsToServer(ServerInfo server, List<ModelRequest> requests) {
        try {
            String response = restTemplate.postForObject(
                server.getServerUrl() + "/api/process",
                requests,
                String.class
            );

            circuitBreakerService.recordSuccess(server.getServerId());
            System.out.println("Sent " + requests.size() + " requests to " + server.getServerId());

        } catch (Exception e) {
            circuitBreakerService.recordFailure(server.getServerId());
            System.err.println("Failed to send requests to " + server.getServerId() + ": " + e.getMessage());
        }
    }
}

