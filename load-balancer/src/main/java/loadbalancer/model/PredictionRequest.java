package loadbalancer.model;

import java.util.Map;

public class PredictionRequest {
    private Map<String, Integer> modelCounts;
    private Map<String, Integer> tokenCounts;

    public PredictionRequest() {}

    public PredictionRequest(Map<String, Integer> modelCounts, Map<String, Integer> tokenCounts) {
        this.modelCounts = modelCounts;
        this.tokenCounts = tokenCounts;
    }

    public Map<String, Integer> getModelCounts() { return modelCounts; }
    public void setModelCounts(Map<String, Integer> modelCounts) { this.modelCounts = modelCounts; }
    
    public Map<String, Integer> getTokenCounts() { return tokenCounts; }
    public void setTokenCounts(Map<String, Integer> tokenCounts) { this.tokenCounts = tokenCounts; }
}
