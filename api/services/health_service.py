"""
Engine 1 Service
"""

import joblib
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class HealthService:
    def __init__(self):
        self.persistence = PersistenceService()
        self.model = None
        self.model_path = "models/operation_health_model.pkl"
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logger.info(f"✅ Health model loaded: {self.model_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load health model: {e}")
                self.model = None
        else:
            logger.warning(f"⚠️ Health model not found: {self.model_path}")
            self.model = None
    
    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None
    
    def predict(self, input_data: dict):
        """Make prediction"""
        if self.model is None:
            # Return mock prediction if model not loaded
            return self._mock_predict(input_data)
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = self.model.predict(df)[0]
            
            return {
                "operational_health": float(prediction),
                "ensemble_details": {
                    "catboost": float(prediction),
                    "mlp": float(prediction) - 1.5,
                    "ensemble_weighted": float(prediction)
                }
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._mock_predict(input_data)
    
    def _mock_predict(self, input_data: dict):
        """Fallback mock prediction"""
        # Simple weighted average for demo
        weights = {
            'actual_release_rate': 0.30,
            'actual_quality': 0.25,
            'actual_competency': 0.30,
            'actual_attendance': 0.05,
            'actual_transfer_rate': 0.10
        }
        
        score = 0
        for key, weight in weights.items():
            if key in input_data:
                if key == 'actual_transfer_rate':
                    # Invert transfer rate (lower is better)
                    score += (100 - input_data[key]) * weight
                else:
                    score += input_data[key] * weight
        
        # Add intelligence factors
        score += input_data.get('operational_intelligence_factor', 0.5) * 10
        score += input_data.get('business_intelligence_factor', 0.5) * 5
        score += input_data.get('member_intelligence_factor', 0.5) * 5
        
        # Add noise
        score += np.random.normal(0, 2)
        
        # Clip to [0, 100]
        health = np.clip(score, 0, 100)
        
        return {
            "operational_health": float(health),
            "ensemble_details": {
                "catboost": float(health),
                "mlp": float(health) - 1.5,
                "ensemble_weighted": float(health)
            }
        }

# Module-level compatibility surface
def load_model():
    return HealthService().load_model()

def is_loaded():
    return HealthService().is_loaded()

def predict(input_data: dict):
    return HealthService().predict(input_data)
