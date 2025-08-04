import React from 'react';

const Traffic = ({ isLoading, generateTraffic }) => {
  return (
    <div className="content-section">
      <div className="section-header">
        <h2>Traffic Generation</h2>
        <span className="status-text">Generate load testing patterns</span>
      </div>
      
      <div className="traffic-container">
        <div className="traffic-card">
          <h3>Load Testing Patterns</h3>
          <p>Generate different traffic patterns to test load balancing behavior</p>
          
          <div className="traffic-buttons">
            <button 
              onClick={() => generateTraffic('high-a')}
              disabled={isLoading}
              className="btn btn-primary traffic-btn"
            >
              High Model A Traffic
              <span className="btn-description">Focus traffic on Model A endpoints</span>
            </button>
            
            <button 
              onClick={() => generateTraffic('equal')}
              disabled={isLoading}
              className="btn btn-primary traffic-btn"
            >
              Equal Distribution
              <span className="btn-description">Distribute traffic evenly across all models</span>
            </button>
            
            <button 
              onClick={() => generateTraffic('burst')}
              disabled={isLoading}
              className="btn btn-primary traffic-btn"
            >
              Burst Traffic
              <span className="btn-description">Generate sudden traffic spikes</span>
            </button>
          </div>
          
          {isLoading && (
            <div className="loading-indicator">
              <span>Generating traffic...</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Traffic;
