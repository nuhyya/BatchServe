package loadbalancer.model;

import java.util.List;

public class ServerInfo {
    private String serverId;
    private String status; // UP, DOWN, CIRCUIT_OPEN
    private double cpuUsage;
    private double gpuUsage;
    private double memoryUsage;
    private List<String> loadedModels;
    private String serverUrl;
    private int failureCount;
    private long lastFailureTime;

    // Constructors
    public ServerInfo() {}

    // Getters and Setters
    public String getServerId() { return serverId; }
    public void setServerId(String serverId) { this.serverId = serverId; }
    
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    
    public double getCpuUsage() { return cpuUsage; }
    public void setCpuUsage(double cpuUsage) { this.cpuUsage = cpuUsage; }
    
    public double getGpuUsage() { return gpuUsage; }
    public void setGpuUsage(double gpuUsage) { this.gpuUsage = gpuUsage; }
    
    public double getMemoryUsage() { return memoryUsage; }
    public void setMemoryUsage(double memoryUsage) { this.memoryUsage = memoryUsage; }
    
    public List<String> getLoadedModels() { return loadedModels; }
    public void setLoadedModels(List<String> loadedModels) { this.loadedModels = loadedModels; }
    
    public String getServerUrl() { return serverUrl; }
    public void setServerUrl(String serverUrl) { this.serverUrl = serverUrl; }
    
    public int getFailureCount() { return failureCount; }
    public void setFailureCount(int failureCount) { this.failureCount = failureCount; }
    
    public long getLastFailureTime() { return lastFailureTime; }
    public void setLastFailureTime(long lastFailureTime) { this.lastFailureTime = lastFailureTime; }
}

