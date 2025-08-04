import React from 'react';
import { getStatusColor, getUsageColor } from '../utils/helpers';

const Servers = ({ servers, handleModelAction }) => {
  return (
    <div className="content-section">
      <div className="section-header">
        <h2>Server Management</h2>
        <span className="server-count">{servers.length} servers</span>
      </div>
      
      <div className="servers-grid">
        {servers.map((server) => (
          <div key={server.serverId} className="server-card">
            <div className="server-header">
              <h3>{server.serverId}</h3>
              <span 
                className="status-badge"
                style={{ backgroundColor: getStatusColor(server.status) }}
              >
                {server.status}
              </span>
            </div>

            <div className="server-metrics">
              <div className="usage-row">
                <span className="usage-label">CPU Usage</span>
                <div className="usage-bar">
                  <div 
                    className="usage-fill"
                    style={{
                      width: `${server.cpuUsage}%`,
                      backgroundColor: getUsageColor(server.cpuUsage)
                    }}
                  />
                </div>
                <span className="usage-value">{server.cpuUsage?.toFixed(1)}%</span>
              </div>

              <div className="usage-row">
                <span className="usage-label">GPU Usage</span>
                <div className="usage-bar">
                  <div 
                    className="usage-fill"
                    style={{
                      width: `${server.gpuUsage}%`,
                      backgroundColor: getUsageColor(server.gpuUsage)
                    }}
                  />
                </div>
                <span className="usage-value">{server.gpuUsage?.toFixed(1)}%</span>
              </div>

              <div className="usage-row">
                <span className="usage-label">Memory Usage</span>
                <div className="usage-bar">
                  <div 
                    className="usage-fill"
                    style={{
                      width: `${server.memoryUsage}%`,
                      backgroundColor: getUsageColor(server.memoryUsage)
                    }}
                  />
                </div>
                <span className="usage-value">{server.memoryUsage?.toFixed(1)}%</span>
              </div>
            </div>

            <div className="server-info">
              <div className="info-row">
                <span className="info-label">Loaded Models:</span>
                <span className="info-value">{server.loadedModels?.join(', ') || 'None'}</span>
              </div>
              <div className="info-row">
                <span className="info-label">Requests Served:</span>
                <span className="info-value">{server.totalRequestsServed || 0}</span>
              </div>
              <div className="info-row">
                <span className="info-label">Avg Response Time:</span>
                <span className="info-value">{server.avgResponseTime?.toFixed(0) || 0}ms</span>
              </div>
            </div>

            <div className="model-controls">
              <h4>Model Controls</h4>
              <div className="model-grid">
                {['A', 'B', 'C', 'D'].map(model => (
                  <div key={model} className="model-control">
                    <span className="model-label">Model {model}</span>
                    <div className="model-buttons">
                      <button
                        onClick={() => handleModelAction(server.serverId, 'load', model)}
                        disabled={server.status !== 'UP'}
                        className="btn btn-primary btn-sm"
                      >
                        Load
                      </button>
                      <button
                        onClick={() => handleModelAction(server.serverId, 'unload', model)}
                        disabled={server.status !== 'UP'}
                        className="btn btn-secondary btn-sm"
                      >
                        Unload
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Servers;
