package loadbalancer.model;

import java.util.List;
import java.util.Map;

public class RequestBatch {
    private List<ModelRequest> requests;
    private Map<String, Integer> modelCounts; // A: 3, B: 2, etc.
    private Map<String, Integer> tokenCounts; // A: 300, B: 120, etc.
    private long batchId;
    private long timestamp;

    // Constructors
    public RequestBatch() {}

    public RequestBatch(List<ModelRequest> requests, Map<String, Integer> modelCounts, 
                       Map<String, Integer> tokenCounts) {
        this.requests = requests;
        this.modelCounts = modelCounts;
        this.tokenCounts = tokenCounts;
        this.batchId = System.currentTimeMillis();
        this.timestamp = System.currentTimeMillis();
    }

    // Getters and Setters
    public List<ModelRequest> getRequests() { return requests; }
    public void setRequests(List<ModelRequest> requests) { this.requests = requests; }
    
    public Map<String, Integer> getModelCounts() { return modelCounts; }
    public void setModelCounts(Map<String, Integer> modelCounts) { this.modelCounts = modelCounts; }
    
    public Map<String, Integer> getTokenCounts() { return tokenCounts; }
    public void setTokenCounts(Map<String, Integer> tokenCounts) { this.tokenCounts = tokenCounts; }
    
    public long getBatchId() { return batchId; }
    public void setBatchId(long batchId) { this.batchId = batchId; }
    
    public long getTimestamp() { return timestamp; }
    public void setTimestamp(long timestamp) { this.timestamp = timestamp; }
}

