# load-predictor/app.py
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from datetime import datetime

app = Flask(__name__)

class ModelPredictor:
    def __init__(self):
        self.cpu_model = LinearRegression()
        self.gpu_model = LinearRegression()
        self.memory_model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Load existing model if available
        self.load_model()
        
        # If no model exists, create and train with synthetic data
        if not self.is_trained:
            self.create_training_data()
            self.train_models()

    def create_training_data(self):
        """Create synthetic training data based on the model formulas"""
        print("Creating synthetic training data...")
        
        np.random.seed(42)
        n_samples = 1000
        
        # Generate random request patterns
        data = []
        for _ in range(n_samples):
            # Random number of requests for each model
            model_a_reqs = np.random.randint(0, 10)
            model_b_reqs = np.random.randint(0, 8)
            model_c_reqs = np.random.randint(0, 6)
            model_d_reqs = np.random.randint(0, 7)
            
            # Random token counts
            model_a_tokens = model_a_reqs * np.random.randint(50, 300) if model_a_reqs > 0 else 0
            model_b_tokens = model_b_reqs * np.random.randint(80, 250) if model_b_reqs > 0 else 0
            model_c_tokens = model_c_reqs * np.random.randint(100, 400) if model_c_reqs > 0 else 0
            model_d_tokens = model_d_reqs * np.random.randint(60, 200) if model_d_reqs > 0 else 0
            
            # Calculate resource usage based on model formulas
            # Model A: 200 base + 4 * tokens + random
            # Model B: 250 base + 5 * tokens + random
            # Model C: 300 base + 6 * tokens + random
            # Model D: 220 base + 4.5 * tokens + random
            
            cpu_usage = 0
            gpu_usage = 0
            memory_usage = 0
            
            if model_a_reqs > 0:
                cpu_usage += (15.0 + (model_a_tokens / 100.0) * 5.0) * model_a_reqs
                gpu_usage += (8.0 + (model_a_tokens / 100.0) * 3.0) * model_a_reqs
                memory_usage += 15.0 * min(2, model_a_reqs)  # Max 2 models per server
                
            if model_b_reqs > 0:
                cpu_usage += (20.0 + (model_b_tokens / 100.0) * 6.0) * model_b_reqs
                gpu_usage += (12.0 + (model_b_tokens / 100.0) * 4.0) * model_b_reqs
                memory_usage += 15.0 * min(2, model_b_reqs)
                
            if model_c_reqs > 0:
                cpu_usage += (25.0 + (model_c_tokens / 100.0) * 7.0) * model_c_reqs
                gpu_usage += (15.0 + (model_c_tokens / 100.0) * 5.0) * model_c_reqs
                memory_usage += 15.0 * min(2, model_c_reqs)
                
            if model_d_reqs > 0:
                cpu_usage += (18.0 + (model_d_tokens / 100.0) * 5.5) * model_d_reqs
                gpu_usage += (10.0 + (model_d_tokens / 100.0) * 3.5) * model_d_reqs
                memory_usage += 15.0 * min(2, model_d_reqs)
            
            # Add some random variation and network effects
            cpu_usage += np.random.normal(0, 3)
            gpu_usage += np.random.normal(0, 2)
            memory_usage += np.random.normal(0, 1)
            
            # Ensure values are within reasonable bounds
            cpu_usage = max(0, min(100, cpu_usage))
            gpu_usage = max(0, min(100, gpu_usage))
            memory_usage = max(0, min(100, memory_usage))
            
            data.append({
                'model_a_reqs': model_a_reqs,
                'model_b_reqs': model_b_reqs,
                'model_c_reqs': model_c_reqs,
                'model_d_reqs': model_d_reqs,
                'model_a_tokens': model_a_tokens,
                'model_b_tokens': model_b_tokens,
                'model_c_tokens': model_c_tokens,
                'model_d_tokens': model_d_tokens,
                'cpu_usage': cpu_usage,
                'gpu_usage': gpu_usage,
                'memory_usage': memory_usage
            })
        
        self.training_data = pd.DataFrame(data)
        print(f"Created {len(self.training_data)} training samples")

    def train_models(self):
        """Train the prediction models"""
        print("Training ML models...")
        
        # Prepare features
        features = [
            'model_a_reqs', 'model_b_reqs', 'model_c_reqs', 'model_d_reqs',
            'model_a_tokens', 'model_b_tokens', 'model_c_tokens', 'model_d_tokens'
        ]
        
        X = self.training_data[features].values
        y_cpu = self.training_data['cpu_usage'].values
        y_gpu = self.training_data['gpu_usage'].values
        y_memory = self.training_data['memory_usage'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        self.cpu_model.fit(X_scaled, y_cpu)
        self.gpu_model.fit(X_scaled, y_gpu)
        self.memory_model.fit(X_scaled, y_memory)
        
        self.is_trained = True
        
        # Save models
        self.save_model()
        
        print("Models trained successfully!")
        print(f"CPU Model R² Score: {self.cpu_model.score(X_scaled, y_cpu):.3f}")
        print(f"GPU Model R² Score: {self.gpu_model.score(X_scaled, y_gpu):.3f}")
        print(f"Memory Model R² Score: {self.memory_model.score(X_scaled, y_memory):.3f}")

    def predict(self, model_counts, token_counts):
        """Make predictions for resource usage"""
        if not self.is_trained:
            return {
                'predicted_cpu_usage': 20.0,
                'predicted_gpu_usage': 15.0,
                'predicted_memory_usage': 10.0
            }
        
        # Prepare features
        features = [
            model_counts.get('A', 0),
            model_counts.get('B', 0),
            model_counts.get('C', 0),
            model_counts.get('D', 0),
            token_counts.get('A', 0),
            token_counts.get('B', 0),
            token_counts.get('C', 0),
            token_counts.get('D', 0)
        ]
        
        # Scale features
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        cpu_pred = max(0, min(100, self.cpu_model.predict(X_scaled)[0]))
        gpu_pred = max(0, min(100, self.gpu_model.predict(X_scaled)[0]))
        memory_pred = max(0, min(100, self.memory_model.predict(X_scaled)[0]))
        
        return {
            'predicted_cpu_usage': float(cpu_pred),
            'predicted_gpu_usage': float(gpu_pred),
            'predicted_memory_usage': float(memory_pred)
        }

    def save_model(self):
        """Save trained models to disk"""
        try:
            os.makedirs('/app/data', exist_ok=True)
            joblib.dump(self.cpu_model, '/app/data/cpu_model.pkl')
            joblib.dump(self.gpu_model, '/app/data/gpu_model.pkl')
            joblib.dump(self.memory_model, '/app/data/memory_model.pkl')
            joblib.dump(self.scaler, '/app/data/scaler.pkl')
            print("Models saved successfully!")
        except Exception as e:
            print(f"Failed to save models: {e}")

    def load_model(self):
        """Load trained models from disk"""
        try:
            if (os.path.exists('/app/data/cpu_model.pkl') and 
                os.path.exists('/app/data/gpu_model.pkl') and
                os.path.exists('/app/data/memory_model.pkl') and
                os.path.exists('/app/data/scaler.pkl')):
                
                self.cpu_model = joblib.load('/app/data/cpu_model.pkl')
                self.gpu_model = joblib.load('/app/data/gpu_model.pkl')
                self.memory_model = joblib.load('/app/data/memory_model.pkl')
                self.scaler = joblib.load('/app/data/scaler.pkl')
                self.is_trained = True
                print("Models loaded successfully!")
                return True
        except Exception as e:
            print(f"Failed to load models: {e}")
        return False

# Initialize predictor
predictor = ModelPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_counts = data.get('modelCounts', {})
        token_counts = data.get('tokenCounts', {})
        
        prediction = predictor.predict(model_counts, token_counts)
        
        # Log prediction for debugging
        print(f"Prediction request: {model_counts}, {token_counts}")
        print(f"Prediction result: {prediction}")
        
        return jsonify(prediction)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'predicted_cpu_usage': 20.0,
            'predicted_gpu_usage': 15.0,
            'predicted_memory_usage': 10.0,
            'error': str(e)
        }), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        predictor.create_training_data()
        predictor.train_models()
        return jsonify({'status': 'success', 'message': 'Models retrained successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_trained': predictor.is_trained,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    if not predictor.is_trained:
        return jsonify({'error': 'Models not trained yet'}), 400
    
    return jsonify({
        'cpu_model_coef': predictor.cpu_model.coef_.tolist(),
        'gpu_model_coef': predictor.gpu_model.coef_.tolist(),
        'memory_model_coef': predictor.memory_model.coef_.tolist(),
        'feature_names': [
            'model_a_reqs', 'model_b_reqs', 'model_c_reqs', 'model_d_reqs',
            'model_a_tokens', 'model_b_tokens', 'model_c_tokens', 'model_d_tokens'
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
