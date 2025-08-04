import React from 'react';
import { getStatusColor } from '../utils/helpers';

const Metrics = ({ requestCounts, servers }) => {
  return (
    <div className="content-section">
      <div className="section-header">
        <h2>Request Metrics</h2>
        <span className="metrics-total">
          Total: {Object.values(requestCounts).reduce((a, b) => a + b, 0)} requests
        </span>
      </div>
      
      <div className="metrics-grid">
        {Object.entries(requestCounts).map(([model, count]) => (
          <div key={model} className="metric-card">
            <div className="metric-header">
              <h3>Model {model}</h3>
            </div>
            <div className="metric-value">{count}</div>
            <div className="metric-label">Total Requests</div>
            <div className="metric-percentage">
              {Object.values(requestCounts).reduce((a, b) => a + b, 0) > 0 
                ? ((count / Object.values(requestCounts).reduce((a, b) => a + b, 0)) * 100).toFixed(1)
                : 0}%
            </div>
          </div>
        ))}
      </div>

      <div className="table-container">
        <h3>Request Distribution by Server</h3>
        <table className="data-table">
          <thead>
            <tr>
              <th>Server ID</th>
              <th>Status</th>
              <th>Requests Served</th>
              <th>Avg Response Time</th>
              <th>Loaded Models</th>
            </tr>
          </thead>
          <tbody>
            {servers.map((server) => (
              <tr key={server.serverId}>
                <td className="server-id">{server.serverId}</td>
                <td>
                  <span 
                    className="status-badge"
                    style={{ backgroundColor: getStatusColor(server.status) }}
                  >
                    {server.status}
                  </span>
                </td>
                <td>{server.totalRequestsServed || 0}</td>
                <td>{server.avgResponseTime?.toFixed(0) || 0}ms</td>
                <td>{server.loadedModels?.join(', ') || 'None'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Metrics;
