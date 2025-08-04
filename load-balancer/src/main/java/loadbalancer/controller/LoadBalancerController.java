package loadbalancer.controller;

import loadbalancer.model.RequestBatch;
import loadbalancer.service.LoadBalancingService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")
public class LoadBalancerController {

    private final LoadBalancingService loadBalancingService;

    @Autowired
    public LoadBalancerController(LoadBalancingService loadBalancingService) {
        this.loadBalancingService = loadBalancingService;
    }

    @PostMapping("/process-batch")
    public ResponseEntity<String> processBatch(@RequestBody RequestBatch batch) {
        try {
            loadBalancingService.processBatch(batch);
            return ResponseEntity.ok("Batch processed successfully");
        } catch (Exception e) {
            return ResponseEntity.internalServerError()
                .body("Failed to process batch: " + e.getMessage());
        }
    }

    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("Load Balancer is healthy");
    }
}
