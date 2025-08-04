package loadbalancer.model;

public class PredictionResponse {
    private double predictedCpuUsage;
    private double predictedGpuUsage;
    private double predictedMemoryUsage;

    public PredictionResponse() {}

    public double getPredictedCpuUsage() { return predictedCpuUsage; }
    public void setPredictedCpuUsage(double predictedCpuUsage) { this.predictedCpuUsage = predictedCpuUsage; }
    
    public double getPredictedGpuUsage() { return predictedGpuUsage; }
    public void setPredictedGpuUsage(double predictedGpuUsage) { this.predictedGpuUsage = predictedGpuUsage; }
    
    public double getPredictedMemoryUsage() { return predictedMemoryUsage; }
    public void setPredictedMemoryUsage(double predictedMemoryUsage) { this.predictedMemoryUsage = predictedMemoryUsage; }
}
