"""
Data Quality & Exploratory Analytics
"""
import pandas as pd
import numpy as np
from .base import AnalyticsBase

class DataAnalytics(AnalyticsBase):
    def quality_report(self):
        """Generate comprehensive data quality report."""
        if self.df is None:
            raise ValueError("No data loaded.")
        
        self.results = {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "missing_values": self.df.isnull().sum().to_dict(),
            "missing_percentage": (self.df.isnull().sum() / len(self.df) * 100).round(2).to_dict(),
            "duplicate_rows": int(self.df.duplicated().sum()),
            "column_types": self.df.dtypes.astype(str).to_dict(),
        }
        return self.results
    
    def outlier_detection(self, columns=None, method='iqr'):
        """Detect outliers using IQR or Z-score."""
        if self.df is None:
            raise ValueError("No data loaded.")
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        outliers = {}
        for col in columns:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier_count = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            else:  # z-score
                z = (self.df[col] - self.df[col].mean()) / self.df[col].std()
                outlier_count = (abs(z) > 3).sum()
                lower = self.df[col].mean() - 3 * self.df[col].std()
                upper = self.df[col].mean() + 3 * self.df[col].std()
            
            outliers[col] = {
                "count": int(outlier_count),
                "percentage": float(outlier_count / len(self.df) * 100),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "min": float(self.df[col].min()),
                "max": float(self.df[col].max()),
            }
        self.results = outliers
        return self.results
    
    def summary_stats(self, columns=None):
        """Get descriptive statistics."""
        if self.df is None:
            raise ValueError("No data loaded.")
        
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        self.results = {}
        for col in columns:
            self.results[col] = {
                "mean": float(self.df[col].mean()),
                "median": float(self.df[col].median()),
                "std": float(self.df[col].std()),
                "min": float(self.df[col].min()),
                "max": float(self.df[col].max()),
                "skew": float(self.df[col].skew()),
                "kurtosis": float(self.df[col].kurtosis()),
                "q25": float(self.df[col].quantile(0.25)),
                "q75": float(self.df[col].quantile(0.75)),
                "missing": int(self.df[col].isnull().sum()),
            }
        return self.results
    
    def correlation_matrix(self, method='pearson'):
        """Compute correlation matrix."""
        if self.df is None:
            raise ValueError("No data loaded.")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.results = self.df[numeric_cols].corr(method=method).to_dict()
        return self.results

# Module-level compatibility surface

def quality_report(self):
    return DataAnalytics().quality_report()

def outlier_detection(self, columns=None, method='iqr'):
    return DataAnalytics().outlier_detection(columns, method)

def summary_stats(self, columns=None):
    return DataAnalytics().summary_stats(columns)

def correlation_matrix(self, method='pearson'):
    return DataAnalytics().correlation_matrix(method)
