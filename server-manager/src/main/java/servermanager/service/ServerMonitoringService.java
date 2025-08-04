package servermanager.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import servermanager.model.ServerData;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class ServerMonitoringService {

    private final RestTemplate restTemplate = new RestTemplate();
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final Map<String, ServerData> servers = new ConcurrentHashMap<>();
    private final Map<String, Integer> modelRequestCounts = new ConcurrentHashMap<>();
    
    private static final String DATA_FILE = "/app/data/servers.json";
    private static final String[] MODEL_SERVERS = {
        "http://model-server-1:8084",
        "http://model-server-2:8085", 
        "http://model-server-3:8086",
        "http://model-server-4:8087",
        "http://model-server-5:8088",
        "http://model-server-6:8089"
    };

    public ServerMonitoringService() {
        initializeServers();
        loadServerData();
    }

    private void initializeServers() {
        for (int i = 0; i < MODEL_SERVERS.length; i++) {
            String serverId = "server-" + (i + 1);
            servers.put(serverId, new ServerData(serverId, MODEL_SERVERS[i]));
        }
        
        // Initialize model request counters
        modelRequestCounts.put("A", 0);
        modelRequestCounts.put("B", 0);
        modelRequestCounts.put("C", 0);
        modelRequestCounts.put("D", 0);
    }

    @Scheduled(fixedRate = 5000) // Every 5 seconds
    public void monitorServers() {
        for (ServerData server : servers.values()) {
            checkServerHealth(server);
        }
        saveServerData();
    }

    private void checkServerHealth(ServerData server) {
        try {
            Map<String, Object> status = restTemplate.getForObject(
                server.getServerUrl() + "/api/status", 
                Map.class
            );
            
            if (status != null) {
                server.setStatus("UP");
                server.setHealth("HEALTHY");
                server.setCpuUsage((Double) status.getOrDefault("cpuUsage", 0.0));
                server.setGpuUsage((Double) status.getOrDefault("gpuUsage", 0.0));
                server.setMemoryUsage((Double) status.getOrDefault("memoryUsage", 0.0));
                server.setLoadedModels((List<String>) status.getOrDefault("loadedModels", new ArrayList<>()));
                server.setTotalRequestsServed((Integer) status.getOrDefault("totalRequests", 0));
                server.setAvgResponseTime((Double) status.getOrDefault("avgResponseTime", 0.0));
                server.setLastUpdated(System.currentTimeMillis());
            }
            
        } catch (Exception e) {
            server.setStatus("DOWN");
            server.setHealth("UNHEALTHY");
            System.err.println("Server " + server.getServerId() + " is down: " + e.getMessage());
        }
    }

    public List<ServerData> getAllServers() {
        return new ArrayList<>(servers.values());
    }

    public ServerData getServer(String serverId) {
        return servers.get(serverId);
    }

    public void updateModelRequestCounts(Map<String, Integer> requestCounts) {
        requestCounts.forEach((model, count) -> 
            modelRequestCounts.merge(model, count, Integer::sum)
        );
        
        // Check if auto-scaling is needed
        checkAutoScaling();
    }

    private void checkAutoScaling() {
        // Simple auto-scaling logic: if any model has > 100 requests, try to scale
        modelRequestCounts.forEach((model, count) -> {
            if (count > 100) {
                scaleForModel(model);
            }
        });
    }

    private void scaleForModel(String modelType) {
        // Find a DOWN server to bring up
        Optional<ServerData> downServer = servers.values().stream()
            .filter(s -> "DOWN".equals(s.getStatus()))
            .findFirst();
            
        if (downServer.isPresent()) {
            ServerData server = downServer.get();
            try {
                // Try to start the server and load the required model
                restTemplate.postForObject(
                    server.getServerUrl() + "/api/load-model", 
                    Collections.singletonMap("modelType", modelType), 
                    String.class
                );
                System.out.println("Auto-scaled server " + server.getServerId() + " for model " + modelType);
            } catch (Exception e) {
                System.err.println("Failed to auto-scale server: " + e.getMessage());
            }
        }
    }

    private void saveServerData() {
        try {
            // Ensure data directory exists
            File dataDir = new File("/app/data");
            if (!dataDir.exists()) {
                dataDir.mkdirs();
            }
            
            Map<String, Object> data = new HashMap<>();
            data.put("servers", servers.values());
            data.put("lastUpdated", System.currentTimeMillis());
            data.put("modelRequestCounts", modelRequestCounts);
            
            objectMapper.writeValue(new File(DATA_FILE), data);
        } catch (IOException e) {
            System.err.println("Failed to save server data: " + e.getMessage());
        }
    }

    private void loadServerData() {
        try {
            File file = new File(DATA_FILE);
            if (file.exists()) {
                Map<String, Object> data = objectMapper.readValue(file, Map.class);
                System.out.println("Loaded existing server data from " + DATA_FILE);
            }
        } catch (IOException e) {
            System.out.println("No existing server data found, starting fresh");
        }
    }
}

