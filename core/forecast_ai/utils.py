import json
from datetime import datetime, timedelta

def validate_inputs(state):
    pass

def convert_to_serializable(obj):
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return str(obj)

def safe_divide(a, b, default=0.0):
    return default if b == 0 else a / b

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def date_range(start, end):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)
