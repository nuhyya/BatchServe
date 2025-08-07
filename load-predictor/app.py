from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import json
from datetime import datetime
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import TensorFlow, fallback to sklearn if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tf.get_logger().setLevel('ERROR')  # Suppress TF warnings
    USE_TENSORFLOW = True
    logger.info("Using TensorFlow for FCN models")
except ImportError:
    logger.warning("TensorFlow not available, falling back to sklearn MLPRegressor")
    from sklearn.neural_network import MLPRegressor
    USE_TENSORFLOW = False

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class FCNModelPredictor:
    def __init__(self):
        self.cpu_model = None
        self.gpu_model = None
        self.memory_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_scores = {}
        
        # Load existing model if available
        self.load_model()
        
        # If no model exists, create and train with synthetic data
        if not self.is_trained:
            self.create_training_data()
            self.train_models()

    def create_fcn_model(self, input_dim, model_name):
        """Create a Fully Connected Network model"""
        if USE_TENSORFLOW:
            model = keras.Sequential([
                layers.Dense(128, activation='relu', input_shape=(input_dim,)),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='linear')  # Output layer for regression
            ])
            
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            return model
        else:
            # Fallback to sklearn MLPRegressor
            model = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32, 16),
                activation='relu',
                solver='adam',
                max_iter=200,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2
            )
            return model

    def create_training_data(self):
        """Create synthetic training data based on the model formulas"""
        print("Creating synthetic training data...")
        
        np.random.seed(42)
        n_samples = 5000  # Increased for better FCN training
        
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
            cpu_usage = 0
            gpu_usage = 0
            memory_usage = 0
            
            if model_a_reqs > 0:
                cpu_usage += (15.0 + (model_a_tokens / 100.0) * 5.0) * model_a_reqs
                gpu_usage += (8.0 + (model_a_tokens / 100.0) * 3.0) * model_a_reqs
                memory_usage += 15.0 * min(2, model_a_reqs)
                
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
            cpu_usage += np.random.normal(0, 1)
            gpu_usage += np.random.normal(0, 1)
            memory_usage += np.random.normal(0, 1)
            
            # Ensure values are within reasonable bounds
            cpu_usage = max(0, cpu_usage)
            gpu_usage = max(0, gpu_usage)
            memory_usage = max(0, memory_usage)
            
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
        """Train the FCN prediction models with proper train/test split"""
        try:
            logger.info("Starting model training...")
            
            # Prepare features
            features = [
                'model_a_reqs', 'model_b_reqs', 'model_c_reqs', 'model_d_reqs',
                'model_a_tokens', 'model_b_tokens', 'model_c_tokens', 'model_d_tokens'
            ]
            
            X = self.training_data[features].values
            y_cpu = self.training_data['cpu_usage'].values
            y_gpu = self.training_data['gpu_usage'].values
            y_memory = self.training_data['memory_usage'].values
            
            # Train/test split
            X_train, X_test, y_cpu_train, y_cpu_test = train_test_split(
                X, y_cpu, test_size=0.2, random_state=42
            )
            _, _, y_gpu_train, y_gpu_test = train_test_split(
                X, y_gpu, test_size=0.2, random_state=42
            )
            _, _, y_memory_train, y_memory_test = train_test_split(
                X, y_memory, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create and train models
            input_dim = X_train_scaled.shape[1]
            
            if USE_TENSORFLOW:
                # TensorFlow training
                logger.info("Training CPU model...")
                self.cpu_model = self.create_fcn_model(input_dim, 'cpu')
                cpu_history = self.cpu_model.fit(
                    X_train_scaled, y_cpu_train,
                    epochs=50,  # Reduced epochs for faster training
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
                
                logger.info("Training GPU model...")
                self.gpu_model = self.create_fcn_model(input_dim, 'gpu')
                gpu_history = self.gpu_model.fit(
                    X_train_scaled, y_gpu_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
                
                logger.info("Training Memory model...")
                self.memory_model = self.create_fcn_model(input_dim, 'memory')
                memory_history = self.memory_model.fit(
                    X_train_scaled, y_memory_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
                
                # Evaluate models on test set
                cpu_pred = self.cpu_model.predict(X_test_scaled, verbose=0).flatten()
                gpu_pred = self.gpu_model.predict(X_test_scaled, verbose=0).flatten()
                memory_pred = self.memory_model.predict(X_test_scaled, verbose=0).flatten()
                
                # Calculate metrics
                self.training_scores = {
                    'cpu': {
                        'train_loss': float(cpu_history.history['loss'][-1]),
                        'val_loss': float(cpu_history.history['val_loss'][-1]),
                        'train_mae': float(cpu_history.history['mae'][-1]),
                        'val_mae': float(cpu_history.history['val_mae'][-1]),
                        'test_mse': float(mean_squared_error(y_cpu_test, cpu_pred)),
                        'test_mae': float(mean_absolute_error(y_cpu_test, cpu_pred)),
                        'test_r2': float(r2_score(y_cpu_test, cpu_pred))
                    },
                    'gpu': {
                        'train_loss': float(gpu_history.history['loss'][-1]),
                        'val_loss': float(gpu_history.history['val_loss'][-1]),
                        'train_mae': float(gpu_history.history['mae'][-1]),
                        'val_mae': float(gpu_history.history['val_mae'][-1]),
                        'test_mse': float(mean_squared_error(y_gpu_test, gpu_pred)),
                        'test_mae': float(mean_absolute_error(y_gpu_test, gpu_pred)),
                        'test_r2': float(r2_score(y_gpu_test, gpu_pred))
                    },
                    'memory': {
                        'train_loss': float(memory_history.history['loss'][-1]),
                        'val_loss': float(memory_history.history['val_loss'][-1]),
                        'train_mae': float(memory_history.history['mae'][-1]),
                        'val_mae': float(memory_history.history['val_mae'][-1]),
                        'test_mse': float(mean_squared_error(y_memory_test, memory_pred)),
                        'test_mae': float(mean_absolute_error(y_memory_test, memory_pred)),
                        'test_r2': float(r2_score(y_memory_test, memory_pred))
                    }
                }
            else:
                # Sklearn MLPRegressor training
                logger.info("Training CPU model (sklearn)...")
                self.cpu_model = self.create_fcn_model(input_dim, 'cpu')
                self.cpu_model.fit(X_train_scaled, y_cpu_train)
                
                logger.info("Training GPU model (sklearn)...")
                self.gpu_model = self.create_fcn_model(input_dim, 'gpu')
                self.gpu_model.fit(X_train_scaled, y_gpu_train)
                
                logger.info("Training Memory model (sklearn)...")
                self.memory_model = self.create_fcn_model(input_dim, 'memory')
                self.memory_model.fit(X_train_scaled, y_memory_train)
                
                # Evaluate models on test set
                cpu_pred = self.cpu_model.predict(X_test_scaled)
                gpu_pred = self.gpu_model.predict(X_test_scaled)
                memory_pred = self.memory_model.predict(X_test_scaled)
                
                # Calculate metrics for sklearn models
                self.training_scores = {
                    'cpu': {
                        'test_mse': float(mean_squared_error(y_cpu_test, cpu_pred)),
                        'test_mae': float(mean_absolute_error(y_cpu_test, cpu_pred)),
                        'test_r2': float(r2_score(y_cpu_test, cpu_pred)),
                        'train_score': float(self.cpu_model.score(X_train_scaled, y_cpu_train))
                    },
                    'gpu': {
                        'test_mse': float(mean_squared_error(y_gpu_test, gpu_pred)),
                        'test_mae': float(mean_absolute_error(y_gpu_test, gpu_pred)),
                        'test_r2': float(r2_score(y_gpu_test, gpu_pred)),
                        'train_score': float(self.gpu_model.score(X_train_scaled, y_gpu_train))
                    },
                    'memory': {
                        'test_mse': float(mean_squared_error(y_memory_test, memory_pred)),
                        'test_mae': float(mean_absolute_error(y_memory_test, memory_pred)),
                        'test_r2': float(r2_score(y_memory_test, memory_pred)),
                        'train_score': float(self.memory_model.score(X_train_scaled, y_memory_train))
                    }
                }
            
            # Add common training info
            self.training_scores['training_info'] = {
                'total_samples': len(self.training_data),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features': features,
                'model_type': 'TensorFlow FCN' if USE_TENSORFLOW else 'Sklearn MLP',
                'training_timestamp': datetime.now().isoformat()
            }
            
            self.is_trained = True
            
            # Save models
            self.save_model()
            
            # Print training results
            logger.info("Models trained successfully!")
            logger.info("\n=== TRAINING RESULTS ===")
            for resource in ['cpu', 'gpu', 'memory']:
                scores = self.training_scores[resource]
                logger.info(f"\n{resource.upper()} Model:")
                logger.info(f"  Test R² Score: {scores['test_r2']:.4f}")
                logger.info(f"  Test MAE: {scores['test_mae']:.4f}")
                logger.info(f"  Test MSE: {scores['test_mse']:.4f}")
                if USE_TENSORFLOW:
                    logger.info(f"  Final Train Loss: {scores.get('train_loss', 'N/A')}")
                    logger.info(f"  Final Val Loss: {scores.get('val_loss', 'N/A')}")
                else:
                    logger.info(f"  Train Score: {scores.get('train_score', 'N/A')}")
                    
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def predict(self, model_counts, token_counts):
        """Make predictions for resource usage"""
        try:
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
            if USE_TENSORFLOW:
                cpu_pred = max(0, self.cpu_model.predict(X_scaled, verbose=0)[0][0])
                gpu_pred = max(0, self.gpu_model.predict(X_scaled, verbose=0)[0][0])
                memory_pred = max(0, self.memory_model.predict(X_scaled, verbose=0)[0][0])
            else:
                cpu_pred = max(0, self.cpu_model.predict(X_scaled)[0])
                gpu_pred = max(0, self.gpu_model.predict(X_scaled)[0])
                memory_pred = max(0, self.memory_model.predict(X_scaled)[0])
            
            return {
                'predicted_cpu_usage': float(cpu_pred),
                'predicted_gpu_usage': float(gpu_pred),
                'predicted_memory_usage': float(memory_pred)
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'predicted_cpu_usage': 20.0,
                'predicted_gpu_usage': 15.0,
                'predicted_memory_usage': 10.0,
                'error': str(e)
            }

    def save_model(self):
        """Save trained models to disk"""
        try:
            os.makedirs('/app/data', exist_ok=True)
            
            if USE_TENSORFLOW:
                # Save Keras models
                self.cpu_model.save('/app/data/cpu_model.h5')
                self.gpu_model.save('/app/data/gpu_model.h5')
                self.memory_model.save('/app/data/memory_model.h5')
            else:
                # Save sklearn models
                joblib.dump(self.cpu_model, '/app/data/cpu_model.pkl')
                joblib.dump(self.gpu_model, '/app/data/gpu_model.pkl')
                joblib.dump(self.memory_model, '/app/data/memory_model.pkl')
            
            # Save scaler and training scores
            joblib.dump(self.scaler, '/app/data/scaler.pkl')
            with open('/app/data/training_scores.json', 'w') as f:
                json.dump(self.training_scores, f, indent=2)
                
            logger.info("Models saved successfully!")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")

    def load_model(self):
        """Load trained models from disk"""
        try:
            if USE_TENSORFLOW:
                # Try loading Keras models first
                if (os.path.exists('/app/data/cpu_model.h5') and 
                    os.path.exists('/app/data/gpu_model.h5') and
                    os.path.exists('/app/data/memory_model.h5') and
                    os.path.exists('/app/data/scaler.pkl')):
                    
                    self.cpu_model = keras.models.load_model('/app/data/cpu_model.h5')
                    self.gpu_model = keras.models.load_model('/app/data/gpu_model.h5')
                    self.memory_model = keras.models.load_model('/app/data/memory_model.h5')
                    self.scaler = joblib.load('/app/data/scaler.pkl')
                    
                    # Load training scores if available
                    if os.path.exists('/app/data/training_scores.json'):
                        with open('/app/data/training_scores.json', 'r') as f:
                            self.training_scores = json.load(f)
                    
                    self.is_trained = True
                    logger.info("Keras models loaded successfully!")
                    return True
            else:
                # Try loading sklearn models
                if (os.path.exists('/app/data/cpu_model.pkl') and 
                    os.path.exists('/app/data/gpu_model.pkl') and
                    os.path.exists('/app/data/memory_model.pkl') and
                    os.path.exists('/app/data/scaler.pkl')):
                    
                    self.cpu_model = joblib.load('/app/data/cpu_model.pkl')
                    self.gpu_model = joblib.load('/app/data/gpu_model.pkl')
                    self.memory_model = joblib.load('/app/data/memory_model.pkl')
                    self.scaler = joblib.load('/app/data/scaler.pkl')
                    
                    # Load training scores if available
                    if os.path.exists('/app/data/training_scores.json'):
                        with open('/app/data/training_scores.json', 'r') as f:
                            self.training_scores = json.load(f)
                    
                    self.is_trained = True
                    logger.info("Sklearn models loaded successfully!")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
        return False

# Initialize predictor
try:
    predictor = FCNModelPredictor()
    logger.info("Predictor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize predictor: {e}")
    logger.error(traceback.format_exc())
    # Create a fallback predictor that returns default values
    class FallbackPredictor:
        def __init__(self):
            self.is_trained = False
            self.training_scores = {}
        
        def predict(self, model_counts, token_counts):
            return {
                'predicted_cpu_usage': 20.0,
                'predicted_gpu_usage': 15.0,
                'predicted_memory_usage': 10.0
            }
        
        def create_training_data(self):
            logger.error("Fallback predictor - cannot create training data")
            return False
        
        def train_models(self):
            logger.error("Fallback predictor - cannot train models")
            return False
    
    predictor = FallbackPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_counts = data.get('modelCounts', {})
        token_counts = data.get('tokenCounts', {})
        
        prediction = predictor.predict(model_counts, token_counts)
        
        # Log prediction for debugging
        logger.info(f"Prediction request: {model_counts}, {token_counts}")
        logger.info(f"Prediction result: {prediction}")
        
        return jsonify(prediction)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'predicted_cpu_usage': 20.0,
            'predicted_gpu_usage': 15.0,
            'predicted_memory_usage': 10.0,
            'error': str(e)
        }), 500

@app.route('/training-scores', methods=['GET'])
def get_training_scores():
    """Endpoint to get training scores and model performance metrics"""
    try:
        if not predictor.is_trained or not predictor.training_scores:
            return jsonify({'error': 'Models not trained yet or scores not available'}), 400
        
        return jsonify({
            'status': 'success',
            'training_scores': predictor.training_scores
        })
        
    except Exception as e:
        logger.error(f"Training scores error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        logger.info("Starting retrain process...")
        
        if hasattr(predictor, 'create_training_data') and hasattr(predictor, 'train_models'):
            predictor.create_training_data()
            predictor.train_models()
            
            return jsonify({
                'status': 'success', 
                'message': 'FCN models retrained successfully',
                'training_scores': predictor.training_scores
            })
        else:
            return jsonify({
                'status': 'error', 
                'message': 'Predictor does not support retraining'
            }), 400
            
    except Exception as e:
        logger.error(f"Retrain error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error', 
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/health', methods=['GET'])
def health():
    try:
        model_type = 'TensorFlow FCN' if USE_TENSORFLOW else 'Sklearn MLP'
        if hasattr(predictor, 'is_trained'):
            is_trained = predictor.is_trained
        else:
            is_trained = False
            
        return jsonify({
            'status': 'healthy',
            'model_trained': is_trained,
            'model_type': model_type,
            'tensorflow_available': USE_TENSORFLOW,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    try:
        if not hasattr(predictor, 'is_trained') or not predictor.is_trained:
            return jsonify({'error': 'Models not trained yet'}), 400
        
        model_type = 'TensorFlow FCN' if USE_TENSORFLOW else 'Sklearn MLP'
        
        if USE_TENSORFLOW:
            architecture = {
                'layers': [
                    'Dense(128, activation=relu)',
                    'Dropout(0.3)',
                    'Dense(64, activation=relu)',
                    'Dropout(0.2)',
                    'Dense(32, activation=relu)',
                    'Dense(16, activation=relu)',
                    'Dense(1, activation=linear)'
                ]
            }
        else:
            architecture = {
                'hidden_layers': [128, 64, 32, 16],
                'activation': 'relu',
                'solver': 'adam'
            }
        
        model_info_data = {
            'model_type': model_type,
            'architecture': architecture,
            'feature_names': [
                'model_a_reqs', 'model_b_reqs', 'model_c_reqs', 'model_d_reqs',
                'model_a_tokens', 'model_b_tokens', 'model_c_tokens', 'model_d_tokens'
            ],
            'optimizer': 'adam',
            'loss_function': 'mse' if USE_TENSORFLOW else 'squared_error'
        }
        
        if hasattr(predictor, 'training_scores') and predictor.training_scores:
            model_info_data['latest_training_scores'] = predictor.training_scores
        
        return jsonify(model_info_data)
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, debug=True)
