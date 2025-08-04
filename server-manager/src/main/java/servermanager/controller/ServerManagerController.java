package servermanager.controller;

import servermanager.model.ServerData;
import servermanager.service.ServerMonitoringService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")
public class ServerManagerController {

    private final ServerMonitoringService monitoringService;

    @Autowired
    public ServerManagerController(ServerMonitoringService monitoringService) {
        this.monitoringService = monitoringService;
    }

    @GetMapping("/servers/status")
    public ResponseEntity<List<ServerData>> getServerStatus() {
        return ResponseEntity.ok(monitoringService.getAllServers());
    }

    @GetMapping("/servers/{serverId}")
    public ResponseEntity<ServerData> getServer(@PathVariable String serverId) {
        ServerData server = monitoringService.getServer(serverId);
        if (server != null) {
            return ResponseEntity.ok(server);
        }
        return ResponseEntity.notFound().build();
    }

    @PostMapping("/servers/request-counts")
    public ResponseEntity<String> updateRequestCounts(@RequestBody Map<String, Integer> requestCounts) {
        monitoringService.updateModelRequestCounts(requestCounts);
        return ResponseEntity.ok("Request counts updated");
    }

    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("Server Manager is healthy");
    }
}
