def validate_business_rules(state):
    if not isinstance(state, dict):
        raise ValueError("state must be a mapping")
    rules = {
        "release": (50.0, 100.0),
        "transfer": (0.0, 20.0),
        "quality": (60.0, 100.0),
        "competency": (55.0, 100.0),
        "attendance": (65.0, 100.0),
        "operations_health": (0.0, 100.0),
    }
    for key, (lower, upper) in rules.items():
        if key in state and not lower <= float(state[key]) <= upper:
            raise ValueError(f"{key} must be within {lower} and {upper}")
    return True
