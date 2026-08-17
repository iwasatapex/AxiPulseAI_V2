"""
Business Metrics Analytics (NPS, OPERATIONS_HEALTH, KPIs, Call Volume)
"""
import pandas as pd
import numpy as np
from .base import AnalyticsBase

class BusinessAnalytics(AnalyticsBase):
    def __init__(self, data=None, data_path=None):
        super().__init__(data, data_path)
        self.date_col = 'date' if data is not None and 'date' in data.columns else None
    
    def nps_summary(self):
        """Generate NPS summary statistics."""
        if self.df is None:
            raise ValueError("No data loaded.")
        
        required = ['promoters', 'detractors', 'total_surveys']
        if not all(c in self.df.columns for c in required):
            return {"error": "Missing required columns: promoters, detractors, total_surveys"}
        
        self.df['nps'] = ((self.df['promoters'] - self.df['detractors']) / self.df['total_surveys']) * 100
        self.df['nps'] = self.df['nps'].fillna(0)
        
        self.results = {
            "avg_nps": float(self.df['nps'].mean()),
            "median_nps": float(self.df['nps'].median()),
            "min_nps": float(self.df['nps'].min()),
            "max_nps": float(self.df['nps'].max()),
            "std_nps": float(self.df['nps'].std()),
            "total_surveys": int(self.df['total_surveys'].sum()),
            "total_promoters": int(self.df['promoters'].sum()),
            "total_passives": int(self.df['passives'].sum()),
            "total_detractors": int(self.df['detractors'].sum()),
        }
        return self.results
    
    def nps_trend(self, window=7):
        """Calculate NPS trend with rolling averages."""
        if 'nps' not in self.df.columns:
            self.nps_summary()
        
        self.df['nps_rolling'] = self.df['nps'].rolling(window=window, min_periods=1).mean()
        self.results = {
            "daily": self.df[['date', 'nps']].to_dict(orient='records') if self.date_col else None,
            "rolling": self.df[['date', 'nps_rolling']].to_dict(orient='records') if self.date_col else None,
            "window": window
        }
        return self.results
    
    def oh_summary(self):
        """Generate Operational Health summary."""
        if 'operational_health' not in self.df.columns:
            return {"error": "Missing operational_health column"}
        
        operations_health = self.df['operational_health']
        self.results = {
            "avg_operational_health": float(operations_health.mean()),
            "median_oh": float(operations_health.median()),
            "min_oh": float(operations_health.min()),
            "max_oh": float(operations_health.max()),
            "std_oh": float(operations_health.std()),
            "oh_above_80": float((operations_health >= 80).sum() / len(operations_health) * 100),
            "oh_above_70": float((operations_health >= 70).sum() / len(operations_health) * 100),
            "oh_below_60": float((operations_health < 60).sum() / len(operations_health) * 100),
        }
        return self.results
    
    def kpi_gap_analysis(self):
        """Analyze gaps between target and actual KPIs."""
        kpis = ['quality', 'competency', 'attendance', 'release_rate', 'transfer_rate']
        self.results = {}
        
        for kpi in kpis:
            target = f'target_{kpi}'
            actual = f'actual_{kpi}'
            
            if target in self.df.columns and actual in self.df.columns:
                gap = self.df[target] - self.df[actual]
                self.results[kpi] = {
                    "mean_gap": float(gap.mean()),
                    "median_gap": float(gap.median()),
                    "std_gap": float(gap.std()),
                    "min_gap": float(gap.min()),
                    "max_gap": float(gap.max()),
                    "above_target": float((gap < 0).sum() / len(gap) * 100),
                    "below_target": float((gap > 0).sum() / len(gap) * 100),
                }
        return self.results
    
    def survey_distribution(self):
        """Analyze survey response distribution."""
        scores = [f'score_{i}' for i in range(11)]
        if not any(s in self.df.columns for s in scores):
            return {"error": "No score columns found"}
        
        total = self.df['total_surveys'].sum() if 'total_surveys' in self.df.columns else 1
        
        self.results = {}
        for score in scores:
            if score in self.df.columns:
                self.results[score] = {
                    "count": int(self.df[score].sum()),
                    "percentage": float(self.df[score].sum() / total * 100 if total > 0 else 0),
                }
        return self.results
    
    def call_patterns(self):
        """Analyze call volume patterns."""
        if 'total_calls_received' not in self.df.columns:
            return {"error": "Missing total_calls_received"}
        
        calls = self.df['total_calls_received']
        self.results = {
            "avg_daily": float(calls.mean()),
            "min_daily": int(calls.min()),
            "max_daily": int(calls.max()),
            "std_daily": float(calls.std()),
            "total_calls": int(calls.sum()),
        }
        
        if self.date_col:
            df = self.df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            
            self.results["weekday_avg"] = df.groupby('day_of_week')['total_calls_received'].mean().round(2).to_dict()
            self.results["month_avg"] = df.groupby('month')['total_calls_received'].mean().round(2).to_dict()
        
        return self.results
    
    def release_transfer_analysis(self):
        """Analyze Release Rate + Transfer Rate compliance."""
        if 'actual_release_rate' not in self.df.columns or 'actual_transfer_rate' not in self.df.columns:
            return {"error": "Missing release_rate or transfer_rate"}
        
        sum_rt = self.df['actual_release_rate'] + self.df['actual_transfer_rate']
        self.results = {
            "avg_sum": float(sum_rt.mean()),
            "max_sum": float(sum_rt.max()),
            "min_sum": float(sum_rt.min()),
            "valid_percentage": float((sum_rt < 100).sum() / len(sum_rt) * 100),
            "invalid_percentage": float((sum_rt >= 100).sum() / len(sum_rt) * 100),
        }
        return self.results


def nps_summary(*args, **kwargs):
    return BusinessAnalytics(*args, **kwargs).nps_summary()

def nps_trend(*args, **kwargs):
    return BusinessAnalytics(*args, **kwargs).nps_trend()

def oh_summary(*args, **kwargs):
    return BusinessAnalytics(*args, **kwargs).oh_summary()

def kpi_gap_analysis(*args, **kwargs):
    return BusinessAnalytics(*args, **kwargs).kpi_gap_analysis()

def survey_distribution(*args, **kwargs):
    return BusinessAnalytics(*args, **kwargs).survey_distribution()

def call_patterns(*args, **kwargs):
    return BusinessAnalytics(*args, **kwargs).call_patterns()

def release_transfer_analysis(*args, **kwargs):
    return BusinessAnalytics(*args, **kwargs).release_transfer_analysis()
