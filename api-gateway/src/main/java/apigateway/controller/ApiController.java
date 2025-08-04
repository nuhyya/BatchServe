package apigateway.controller;

import apigateway.model.ModelRequest;
import apigateway.service.BatchingService;
import apigateway.service.MetricsService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;
import java.util.Random;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")
public class ApiController {

    private final BatchingService batchingService;
    private final MetricsService metricsService;
    private final Random random = new Random();

    @Autowired
    public ApiController(BatchingService batchingService, MetricsService metricsService) {
        this.batchingService = batchingService;
        this.metricsService = metricsService;
    }

    @PostMapping("/request")
    public ResponseEntity<String> submitRequest(@RequestBody ModelRequest request) {
        batchingService.addRequest(request);
        return ResponseEntity.ok("Request queued: " + request.getRequestId());
    }

    @PostMapping("/test-traffic/{pattern}")
    public ResponseEntity<String> generateTestTraffic(@PathVariable String pattern) {
        switch (pattern.toLowerCase()) {
            case "high-a":
                generateHighATraffic();
                break;
            case "equal":
                generateEqualTraffic();
                break;
            case "burst":
                generateBurstTraffic();
                break;
            default:
                return ResponseEntity.badRequest().body("Unknown pattern: " + pattern);
        }
        return ResponseEntity.ok("Generated " + pattern + " traffic pattern");
    }

    @GetMapping("/metrics")
    public ResponseEntity<Map<String, Integer>> getMetrics() {
        return ResponseEntity.ok(metricsService.getTotalRequestCounts());
    }

    private void generateHighATraffic() {
        // 70% Model A, 10% each for B, C, D
        for (int i = 0; i < 50; i++) {
            String model = (i < 35) ? "A" : (i < 40) ? "B" : (i < 45) ? "C" : "D";
            int tokens = 50 + random.nextInt(200);
            batchingService.addRequest(new ModelRequest(model, tokens));
        }
    }

    private void generateEqualTraffic() {
        // 25% each model
        for (int i = 0; i < 40; i++) {
            String model = (i < 10) ? "A" : (i < 20) ? "B" : (i < 30) ? "C" : "D";
            int tokens = 50 + random.nextInt(200);
            batchingService.addRequest(new ModelRequest(model, tokens));
        }
    }

    private void generateBurstTraffic() {
        // Quick burst of random requests
        for (int i = 0; i < 100; i++) {
            String[] models = {"A", "B", "C", "D"};
            String model = models[random.nextInt(4)];
            int tokens = 100 + random.nextInt(400);
            batchingService.addRequest(new ModelRequest(model, tokens));
        }
    }
}
