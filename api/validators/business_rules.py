"""
Business Rules Validation
"""

def validate_business_rules(data: dict) -> dict:
    """
    Validate input against business rules
    """
    errors = []
    warnings = []
    
    # Check bounds
    bounds = {
        'target_quality': (60, 100),
        'target_competency': (60, 100),
        'target_attendance': (70, 100),
        'target_release_rate': (50, 100),
        'target_transfer_rate': (0, 13),
        'actual_quality': (60, 100),
        'actual_competency': (60, 100),
        'actual_attendance': (70, 100),
        'actual_release_rate': (50, 100),
        'actual_transfer_rate': (0, 13)
    }
    
    for field, (low, high) in bounds.items():
        if field in data:
            value = data[field]
            if value < low or value > high:
                errors.append(f"{field} must be between {low} and {high}, got {value}")
    
    # Check release + transfer < 100
    if 'target_release_rate' in data and 'target_transfer_rate' in data:
        if data['target_release_rate'] + data['target_transfer_rate'] >= 100:
            errors.append(f"Release + Transfer must be < 100, got {data['target_release_rate'] + data['target_transfer_rate']}")
    
    if 'actual_release_rate' in data and 'actual_transfer_rate' in data:
        if data['actual_release_rate'] + data['actual_transfer_rate'] >= 100:
            errors.append(f"Release + Transfer must be < 100, got {data['actual_release_rate'] + data['actual_transfer_rate']}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }
