package modelserver.controller;

import modelserver.model.ModelRequest;
import modelserver.service.ModelProcessingService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")
public class ModelServerController {

    private final ModelProcessingService processingService;

    @Autowired
    public ModelServerController(ModelProcessingService processingService) {
        this.processingService = processingService;
    }

    @PostMapping("/process")
    public ResponseEntity<String> processRequests(@RequestBody List<ModelRequest> requests) {
        try {
            processingService.processRequests(requests);
            return ResponseEntity.ok("Processed " + requests.size() + " requests");
        } catch (Exception e) {
            return ResponseEntity.internalServerError()
                .body("Failed to process requests: " + e.getMessage());
        }
    }

    @PostMapping("/load-model")
    public ResponseEntity<String> loadModel(@RequestBody Map<String, String> request) {
        try {
            String modelType = request.get("modelType");
            processingService.loadModel(modelType);
            return ResponseEntity.ok("Model " + modelType + " loaded successfully");
        } catch (Exception e) {
            return ResponseEntity.badRequest()
                .body("Failed to load model: " + e.getMessage());
        }
    }

    @PostMapping("/unload-model")
    public ResponseEntity<String> unloadModel(@RequestBody Map<String, String> request) {
        try {
            String modelType = request.get("modelType");
            processingService.unloadModel(modelType);
            return ResponseEntity.ok("Model " + modelType + " unloaded successfully");
        } catch (Exception e) {
            return ResponseEntity.badRequest()
                .body("Failed to unload model: " + e.getMessage());
        }
    }

    @GetMapping("/status")
    public ResponseEntity<Map<String, Object>> getStatus() {
        return ResponseEntity.ok(processingService.getServerStatus());
    }

    @GetMapping("/models")
    public ResponseEntity<List<String>> getLoadedModels() {
        return ResponseEntity.ok(processingService.getLoadedModels());
    }

    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("Model Server is healthy");
    }
}
