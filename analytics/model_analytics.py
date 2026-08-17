"""
Model Performance Analytics (Engine 1 & 2)
"""
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .base import AnalyticsBase

class ModelAnalytics(AnalyticsBase):

    def __init__(self, model_path=None, data=None, data_path=None):
        super().__init__(data, data_path)
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        if model_path is not None and Path(model_path).exists():
            loaded = joblib.load(model_path)
            # If loaded is a dict (from our persistence), extract the model
            if isinstance(loaded, dict):
                self.model = loaded.get("model")
                self.feature_names = loaded.get("feature_names", [])
            else:
                self.model = loaded
            self.metadata = loaded.get("metadata", {}) if isinstance(loaded, dict) else {}

    def set_feature_names(self, names):
        self.feature_names = names
        return self
    
    def evaluate_engine1(self, X_test, y_test):
        """Evaluate Engine 1 (single output regression)."""
        if self.model is None:
            raise ValueError("Model not loaded. Provide model_path.")
        
        y_pred = self.model.predict(X_test)
        
        self.results = {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "R2": float(r2_score(y_test, y_pred)),
            "MAPE": float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100),
        }
        return self.results
    
    def evaluate_engine2(self, X_test, y_test):
        """Evaluate Engine 2 (multi-output, 11 scores)."""
        if self.model is None:
            raise ValueError("Model not loaded. Provide model_path.")
        
        y_pred = self.model.predict(X_test)
        n_outputs = y_pred.shape[1] if len(y_pred.shape) > 1 else 1
        
        if n_outputs == 1:
            return self.evaluate_engine1(X_test, y_test)
        
        mae_per_output = []
        rmse_per_output = []
        for i in range(n_outputs):
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
            mae_per_output.append(float(mae))
            rmse_per_output.append(float(rmse))
        
        self.results = {
            "n_outputs": n_outputs,
            "MAE_per_score": mae_per_output,
            "Average_MAE": float(np.mean(mae_per_output)),
            "RMSE_per_score": rmse_per_output,
            "Average_RMSE": float(np.mean(rmse_per_output)),
            "MAE_by_score": {f"score_{i}": mae_per_output[i] for i in range(n_outputs)},
        }
        return self.results
    
    def evaluate_on_data(self, X_cols, y_cols):
        """Evaluate using columns from the loaded DataFrame."""
        if self.df is None:
            raise ValueError("No data loaded. Provide data or data_path.")
        
        X = self.df[X_cols].values
        y = self.df[y_cols].values if isinstance(y_cols, list) else self.df[y_cols].values
        
        if y.shape[1] > 1:
            return self.evaluate_engine2(X, y)
        return self.evaluate_engine1(X, y)
    
    def feature_importance(self, top_n=20):
        """Extract and sort feature importances."""
        if self.model is None:
            return None
        
        importances = None
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_)
        
        if importances is not None and self.feature_names:
            imp_dict = dict(zip(self.feature_names[:len(importances)], importances))
            return dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:top_n])
        return None
    
    def cross_validation_scores(self, X, y, cv=5):
        """Compute cross-validation scores."""
        from sklearn.model_selection import cross_val_score
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='r2')
        self.results['cv_scores'] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "folds": scores.tolist()
        }
        return self.results

# Module-level compatibility surface

def set_feature_names(names):
    return ModelAnalytics().set_feature_names(names)

def evaluate_engine1(X_test, y_test):
    return ModelAnalytics().evaluate_engine1(X_test, y_test)

def evaluate_engine2(X_test, y_test):
    return ModelAnalytics().evaluate_engine2(X_test, y_test)

def evaluate_on_data(X_cols, y_cols):
    return ModelAnalytics().evaluate_on_data(X_cols, y_cols)

def feature_importance(top_n=20):
    return ModelAnalytics().feature_importance(top_n)

def cross_validation_scores(X, y, cv=5):
    return ModelAnalytics().cross_validation_scores(X, y, cv)
