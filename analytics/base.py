"""Base class for all analytics modules."""
import pandas as pd
import numpy as np
from pathlib import Path
import json

class AnalyticsBase:
    def __init__(self, data=None, data_path=None):
        if data is not None:
            self.df = data
        elif data_path is not None and Path(data_path).exists():
            self.df = pd.read_csv(data_path)
        else:
            self.df = None
        self.results = {}
    
    def load_data(self, data_path):
        self.df = pd.read_csv(data_path)
        return self
    
    def to_json(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        return self


def load_data(data_path):
    return AnalyticsBase().load_data(data_path)

def to_json(filepath):
    return AnalyticsBase().to_json(filepath)
