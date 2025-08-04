package apigateway.service;

import org.springframework.stereotype.Service;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

@Service
public class MetricsService {
    
    private final Map<String, AtomicInteger> totalRequestCounts = new ConcurrentHashMap<>();
    
    public MetricsService() {
        // Initialize counters for all model types
        totalRequestCounts.put("A", new AtomicInteger(0));
        totalRequestCounts.put("B", new AtomicInteger(0));
        totalRequestCounts.put("C", new AtomicInteger(0));
        totalRequestCounts.put("D", new AtomicInteger(0));
    }
    
    public void updateRequestCounts(Map<String, Integer> modelCounts) {
        modelCounts.forEach((model, count) -> 
            totalRequestCounts.get(model).addAndGet(count)
        );
    }
    
    public Map<String, Integer> getTotalRequestCounts() {
        Map<String, Integer> counts = new ConcurrentHashMap<>();
        totalRequestCounts.forEach((model, counter) -> 
            counts.put(model, counter.get())
        );
        return counts;
    }
}
