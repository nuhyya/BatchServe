package modelserver.service;

import modelserver.model.ModelRequest;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

@Service
public class ModelProcessingService {
    
    private final Set<String> loadedModels = ConcurrentHashMap.newKeySet();
    private final Map<String, byte[]> modelMemory = new ConcurrentHashMap<>(); // Simulate memory allocation
    private final AtomicInteger totalRequests = new AtomicInteger(0);
    private final List<Long> responseTimes = Collections.synchronizedList(new ArrayList<>());
    private final Random random = new Random();
    
    // Resource usage tracking
    private volatile double currentCpuUsage = 0.0;
    private volatile double currentGpuUsage = 0.0;
    private volatile double currentMemoryUsage = 0.0;

    public ModelProcessingService() {
        // Initialize with some default models
        loadModel("A");
        loadModel("B");
    }

    public void loadModel(String modelType) {
        if (loadedModels.size() >= 2) {
            throw new RuntimeException("Server can only hold 2 models maximum");
        }
        
        // Simulate memory allocation for model
        int modelSize = 100 * 1024 * 1024; // 100MB per model
        modelMemory.put(modelType, new byte[modelSize]);
        loadedModels.add(modelType);
        
        updateMemoryUsage();
        System.out.println("Loaded model " + modelType + ". Current models: " + loadedModels);
    }

    public void unloadModel(String modelType) {
        if (loadedModels.remove(modelType)) {
            modelMemory.remove(modelType);
            updateMemoryUsage();
            System.out.println("Unloaded model " + modelType + ". Current models: " + loadedModels);
        }
    }

    public List<String> getLoadedModels() {
        return new ArrayList<>(loadedModels);
    }

    public void processRequests(List<ModelRequest> requests) {
        long startTime = System.currentTimeMillis();
        
        for (ModelRequest request : requests) {
            processRequest(request);
        }
        
        long endTime = System.currentTimeMillis();
        long responseTime = endTime - startTime;
        responseTimes.add(responseTime);
        
        // Keep only last 100 response times for averaging
        if (responseTimes.size() > 100) {
            responseTimes.remove(0);
        }
    }

    private void processRequest(ModelRequest request) {
        if (!loadedModels.contains(request.getModelType())) {
            throw new RuntimeException("Model " + request.getModelType() + " not loaded on this server");
        }

        // Simulate processing based on model type and token count
        simulateProcessing(request.getModelType(), request.getTokenCount());
        totalRequests.incrementAndGet();
    }

    private void simulateProcessing(String modelType, int tokenCount) {
        // Different models have different resource requirements
        double baseCpu = getBaseCpuUsage(modelType);
        double baseGpu = getBaseGpuUsage(modelType);
        
        // Calculate processing time and resource usage
        long processingTime = getProcessingTime(modelType, tokenCount);
        double additionalCpu = (tokenCount / 100.0) * 5.0; // 5% CPU per 100 tokens
        double additionalGpu = (tokenCount / 100.0) * 3.0; // 3% GPU per 100 tokens
        
        // Update resource usage
        currentCpuUsage = Math.min(100.0, baseCpu + additionalCpu + random.nextGaussian() * 2);
        currentGpuUsage = Math.min(100.0, baseGpu + additionalGpu + random.nextGaussian() * 1.5);
        
        // Simulate actual CPU work
        burnCpu(processingTime);
        
        // Gradually reduce resource usage after processing
        new Thread(() -> {
            try {
                Thread.sleep(processingTime + 1000);
                currentCpuUsage = Math.max(0, currentCpuUsage - additionalCpu);
                currentGpuUsage = Math.max(0, currentGpuUsage - additionalGpu);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }).start();
    }

    private double getBaseCpuUsage(String modelType) {
        switch (modelType) {
            case "A": return 15.0;
            case "B": return 20.0;
            case "C": return 25.0;
            case "D": return 18.0;
            default: return 10.0;
        }
    }

    private double getBaseGpuUsage(String modelType) {
        switch (modelType) {
            case "A": return 8.0;
            case "B": return 12.0;
            case "C": return 15.0;
            case "D": return 10.0;
            default: return 5.0;
        }
    }

    private long getProcessingTime(String modelType, int tokenCount) {
        // Base processing time + token-based time + random variation
        long baseTime = switch (modelType) {
            case "A" -> 200;
            case "B" -> 250;
            case "C" -> 300;
            case "D" -> 220;
            default -> 200;
        };
        
        long tokenTime = tokenCount * 4; // 4ms per token
        long randomVariation = (long) (random.nextGaussian() * 50); // Network/disk delays
        
        return Math.max(50, baseTime + tokenTime + randomVariation);
    }

    private void burnCpu(long milliseconds) {
        // Actually use CPU cycles to simulate real processing
        long endTime = System.currentTimeMillis() + milliseconds;
        while (System.currentTimeMillis() < endTime) {
            Math.sqrt(random.nextDouble() * 1000000);
        }
    }

    private void updateMemoryUsage() {
        // Calculate memory usage based on loaded models
        double memoryPerModel = 15.0; // 15% memory per model
        currentMemoryUsage = loadedModels.size() * memoryPerModel;
    }

    public Map<String, Object> getServerStatus() {
        Map<String, Object> status = new HashMap<>();
        status.put("serverId", System.getenv("SERVER_ID"));
        status.put("cpuUsage", Math.max(0, Math.min(100, currentCpuUsage)));
        status.put("gpuUsage", Math.max(0, Math.min(100, currentGpuUsage)));
        status.put("memoryUsage", Math.max(0, Math.min(100, currentMemoryUsage)));
        status.put("loadedModels", new ArrayList<>(loadedModels));
        status.put("totalRequests", totalRequests.get());
        status.put("avgResponseTime", getAverageResponseTime());
        return status;
    }

    private double getAverageResponseTime() {
        if (responseTimes.isEmpty()) {
            return 0.0;
        }
        return responseTimes.stream()
            .mapToLong(Long::longValue)
            .average()
            .orElse(0.0);
    }
}

