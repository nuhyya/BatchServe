package loadbalancer.service;

import loadbalancer.model.ServerInfo;
import org.springframework.stereotype.Service;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class CircuitBreakerService {
    
    private static final int FAILURE_THRESHOLD = 3;
    private static final long RECOVERY_TIMEOUT = 30000; // 30 seconds
    
    private final ConcurrentHashMap<String, Integer> failureCounts = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Long> lastFailureTimes = new ConcurrentHashMap<>();

    public boolean isServerAvailable(ServerInfo server) {
        String serverId = server.getServerId();
        
        // Check if circuit is open
        if ("CIRCUIT_OPEN".equals(server.getStatus())) {
            long lastFailure = lastFailureTimes.getOrDefault(serverId, 0L);
            if (System.currentTimeMillis() - lastFailure > RECOVERY_TIMEOUT) {
                // Try to recover
                resetServer(serverId);
                return true;
            }
            return false;
        }
        
        return "UP".equals(server.getStatus());
    }

    public void recordSuccess(String serverId) {
        failureCounts.put(serverId, 0);
    }

    public void recordFailure(String serverId) {
        int failures = failureCounts.getOrDefault(serverId, 0) + 1;
        failureCounts.put(serverId, failures);
        lastFailureTimes.put(serverId, System.currentTimeMillis());
        
        if (failures >= FAILURE_THRESHOLD) {
            System.out.println("Circuit breaker opened for server: " + serverId);
        }
    }

    public boolean shouldOpenCircuit(String serverId) {
        return failureCounts.getOrDefault(serverId, 0) >= FAILURE_THRESHOLD;
    }

    private void resetServer(String serverId) {
        failureCounts.put(serverId, 0);
        System.out.println("Circuit breaker attempting recovery for server: " + serverId);
    }
}
