import React from 'react';
import { getStatusColor } from '../utils/helpers';

const Dashboard = ({ servers, requestCounts, lastUpdate }) => {
  return (
    <div className="content-section">
      <div className="section-header">
        <h2>Dashboard Overview</h2>
        <span className="last-update">Last updated: {lastUpdate.toLocaleTimeString()}</span>
      </div>
      
      <div className="cards-grid">
        <div className="summary-card">
          <h3>Active Servers</h3>
          <div className="summary-value">{servers.filter(s => s.status === 'UP').length}</div>
          <div className="summary-total">of {servers.length} total</div>
        </div>
        
        <div className="summary-card">
          <h3>Total Requests</h3>
          <div className="summary-value">
            {Object.values(requestCounts).reduce((a, b) => a + b, 0)}
          </div>
          <div className="summary-total">across all models</div>
        </div>
        
        <div className="summary-card">
          <h3>Loaded Models</h3>
          <div className="summary-value">
            {servers.reduce((total, server) => total + (server.loadedModels?.length || 0), 0)}
          </div>
          <div className="summary-total">total instances</div>
        </div>
        
        <div className="summary-card">
          <h3>Avg Response Time</h3>
          <div className="summary-value">
            {servers.length > 0 
              ? Math.round(servers.reduce((sum, s) => sum + (s.avgResponseTime || 0), 0) / servers.length)
              : 0}ms
          </div>
          <div className="summary-total">across all servers</div>
        </div>
      </div>

      <div className="table-container">
        <h3>Server Status Overview</h3>
        <table className="data-table">
          <thead>
            <tr>
              <th>Server ID</th>
              <th>Status</th>
              <th>CPU Usage</th>
              <th>Memory Usage</th>
              <th>Loaded Models</th>
              <th>Requests Served</th>
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
                <td>{server.cpuUsage?.toFixed(1)}%</td>
                <td>{server.memoryUsage?.toFixed(1)}%</td>
                <td>{server.loadedModels?.join(', ') || 'None'}</td>
                <td>{server.totalRequestsServed || 0}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Dashboard;
