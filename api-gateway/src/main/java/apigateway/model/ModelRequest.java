package apigateway.model;

public class ModelRequest {
    private String modelType; // A, B, C, D
    private int tokenCount;
    private String requestId;
    private long timestamp;

    // Constructors
    public ModelRequest() {}

    public ModelRequest(String modelType, int tokenCount) {
        this.modelType = modelType;
        this.tokenCount = tokenCount;
        this.requestId = java.util.UUID.randomUUID().toString();
        this.timestamp = System.currentTimeMillis();
    }

    // Getters and Setters
    public String getModelType() { return modelType; }
    public void setModelType(String modelType) { this.modelType = modelType; }
    
    public int getTokenCount() { return tokenCount; }
    public void setTokenCount(int tokenCount) { this.tokenCount = tokenCount; }
    
    public String getRequestId() { return requestId; }
    public void setRequestId(String requestId) { this.requestId = requestId; }
    
    public long getTimestamp() { return timestamp; }
    public void setTimestamp(long timestamp) { this.timestamp = timestamp; }
}

