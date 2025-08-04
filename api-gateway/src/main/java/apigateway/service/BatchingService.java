package apigateway.service;

import apigateway.model.ModelRequest;
import apigateway.model.RequestBatch;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;

@Service
public class BatchingService {
    
    private final Queue<ModelRequest> requestQueue = new ConcurrentLinkedQueue<>();
    private final RestTemplate restTemplate = new RestTemplate();
    private final MetricsService metricsService;
    
    private static final int BATCH_SIZE = 15;
    private static final String LOAD_BALANCER_URL = "http://load-balancer:8082";

    @Autowired
    public BatchingService(MetricsService metricsService) {
        this.metricsService = metricsService;
    }

    public void addRequest(ModelRequest request) {
        requestQueue.offer(request);
        
        // Check if we should batch immediately
        if (requestQueue.size() >= BATCH_SIZE) {
            processBatch();
        }
    }

    @Scheduled(fixedRate = 200) // Every 200ms
    public void scheduledBatch() {
        if (!requestQueue.isEmpty()) {
            processBatch();
        }
    }

    private synchronized void processBatch() {
        if (requestQueue.isEmpty()) return;

        List<ModelRequest> batchRequests = new ArrayList<>();
        Map<String, Integer> modelCounts = new HashMap<>();
        Map<String, Integer> tokenCounts = new HashMap<>();

        // Collect requests for batch
        int count = 0;
        while (!requestQueue.isEmpty() && count < BATCH_SIZE) {
            ModelRequest request = requestQueue.poll();
            batchRequests.add(request);
            
            // Update counts
            modelCounts.merge(request.getModelType(), 1, Integer::sum);
            tokenCounts.merge(request.getModelType(), request.getTokenCount(), Integer::sum);
            count++;
        }

        if (!batchRequests.isEmpty()) {
            RequestBatch batch = new RequestBatch(batchRequests, modelCounts, tokenCounts);
            
            // Update metrics
            metricsService.updateRequestCounts(modelCounts);
            
            // Send to load balancer
            sendBatchToLoadBalancer(batch);
        }
    }

    private void sendBatchToLoadBalancer(RequestBatch batch) {
        try {
            restTemplate.postForObject(
                LOAD_BALANCER_URL + "/api/process-batch", 
                batch, 
                String.class
            );
            System.out.println("Sent batch with " + batch.getRequests().size() + " requests");
        } catch (Exception e) {
            System.err.println("Failed to send batch to load balancer: " + e.getMessage());
        }
    }
}
