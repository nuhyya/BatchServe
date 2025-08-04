package servermanager.model;

import java.util.List;

public class ServerData {
    private String serverId;
    private String status; // UP, DOWN, STARTING
    private String health; // HEALTHY, UNHEALTHY, UNKNOWN
    private double cpuUsage;
    private double gpuUsage;
    private double memoryUsage;
    private List<String> loadedModels;
    private String serverUrl;
    private int totalRequestsServed;
    private double avgResponseTime;
    private long lastUpdated;

    // Constructors
    public ServerData() {}

    public ServerData(String serverId, String serverUrl) {
        this.serverId = serverId;
        this.serverUrl = serverUrl;
        this.status = "DOWN";
        this.health = "UNKNOWN";
        this.cpuUsage = 0.0;
        this.gpuUsage = 0.0;
        this.memoryUsage = 0.0;
        this.totalRequestsServed = 0;
        this.avgResponseTime = 0.0;
        this.lastUpdated = System.currentTimeMillis();
    }

    // Getters and Setters
    public String getServerId() { return serverId; }
    public void setServerId(String serverId) { this.serverId = serverId; }
    
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
    
    public String getHealth() { return health; }
    public void setHealth(String health) { this.health = health; }
    
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
    
    public int getTotalRequestsServed() { return totalRequestsServed; }
    public void setTotalRequestsServed(int totalRequestsServed) { this.totalRequestsServed = totalRequestsServed; }
    
    public double getAvgResponseTime() { return avgResponseTime; }
    public void setAvgResponseTime(double avgResponseTime) { this.avgResponseTime = avgResponseTime; }
    
    public long getLastUpdated() { return lastUpdated; }
    public void setLastUpdated(long lastUpdated) { this.lastUpdated = lastUpdated; }
}
