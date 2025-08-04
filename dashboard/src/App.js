import React, { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import Servers from './components/Servers';
import Traffic from './components/Traffic';
import Metrics from './components/Metrics';
import './App.css';

const API_GATEWAY_URL = process.env.REACT_APP_API_GATEWAY_URL || 'http://localhost:8080';
const SERVER_MANAGER_URL = process.env.REACT_APP_SERVER_MANAGER_URL || 'http://localhost:8083';

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [servers, setServers] = useState([]);
  const [requestCounts, setRequestCounts] = useState({ A: 0, B: 0, C: 0, D: 0 });
  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState(new Date());

  // Fetch server status
  const fetchServerStatus = async () => {
    try {
      const response = await fetch(`${SERVER_MANAGER_URL}/api/servers/status`);
      const data = await response.json();
      setServers(data);
    } catch (error) {
      console.error('Failed to fetch server status:', error);
    }
  };

  // Fetch request metrics
  const fetchRequestMetrics = async () => {
    try {
      const response = await fetch(`${API_GATEWAY_URL}/api/metrics`);
      const data = await response.json();
      setRequestCounts(data);
    } catch (error) {
      console.error('Failed to fetch request metrics:', error);
    }
  };

  // Auto-refresh data every 3 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      fetchServerStatus();
      fetchRequestMetrics();
      setLastUpdate(new Date());
    }, 3000);

    // Initial fetch
    fetchServerStatus();
    fetchRequestMetrics();

    return () => clearInterval(interval);
  }, []);

  // Generate traffic patterns
  const generateTraffic = async (pattern) => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_GATEWAY_URL}/api/test-traffic/${pattern}`, {
        method: 'POST'
      });
      const result = await response.text();
      console.log('Traffic generation result:', result);
    } catch (error) {
      console.error('Failed to generate traffic:', error);
    }
    setIsLoading(false);
  };

  // Load/Unload models
  const handleModelAction = async (serverId, action, modelType) => {
    try {
      const serverUrl = getServerUrl(serverId);
      const response = await fetch(`${serverUrl}/api/${action}-model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ modelType })
      });
      const result = await response.text();
      console.log(`${action} model result:`, result);
      
      // Refresh server status
      setTimeout(fetchServerStatus, 1000);
    } catch (error) {
      console.error(`Failed to ${action} model:`, error);
    }
  };

  const getServerUrl = (serverId) => {
    const serverPorts = {
      'server-1': 8084,
      'server-2': 8085,
      'server-3': 8086,
      'server-4': 8087,
      'server-5': 8088,
      'server-6': 8089
    };
    return `http://localhost:${serverPorts[serverId]}`;
  };

  const navItems = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'servers', label: 'Servers' },
    { id: 'traffic', label: 'Traffic' },
    { id: 'metrics', label: 'Metrics' }
  ];

  const renderContent = () => {
    const commonProps = {
      servers,
      requestCounts,
      lastUpdate,
      isLoading,
      generateTraffic,
      handleModelAction
    };

    switch (currentPage) {
      case 'dashboard':
        return <Dashboard {...commonProps} />;
      case 'servers':
        return <Servers {...commonProps} />;
      case 'traffic':
        return <Traffic {...commonProps} />;
      case 'metrics':
        return <Metrics {...commonProps} />;
      default:
        return <Dashboard {...commonProps} />;
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">BatchServe</div>
          <nav className="nav">
            {navItems.map((item) => (
              <div
                key={item.id}
                className={`nav-item ${currentPage === item.id ? 'active' : ''}`}
                onClick={() => setCurrentPage(item.id)}
              >
                {item.label}
              </div>
            ))}
          </nav>
        </div>
      </header>

      <main className="main">
        {renderContent()}
      </main>
    </div>
  );
}

export default App;
