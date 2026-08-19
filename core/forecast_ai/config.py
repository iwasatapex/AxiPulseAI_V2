"""
ForecastAI – Configuration Constants.
All thresholds, weights, and bounds in one place.
"""

# Forecast horizons (days)
HORIZONS = [1, 7, 30, 90, 180, 365]
DEFAULT_HORIZON = 1

# Confidence default
DEFAULT_CONFIDENCE_LEVEL = 0.95

# Scenario names
SCENARIO_NAMES = {
    'baseline': 'Baseline',
    'optimistic': 'Optimistic',
    'pessimistic': 'Pessimistic',
    'aep': 'AEP',
    'oep': 'OEP',
    'training': 'Training',
    'staffing_shortage': 'Staffing Shortage',
    'technology_upgrade': 'Technology Upgrade'
}

# Risk thresholds (legacy – kept for backward compatibility)
RISK_THRESHOLDS_LEGACY = {'oh_low': 70, 'nps_low': 60, 'transfer_high': 15, 'release_low': 20, 'attendance_low': 85}

# Environment
ENV = 'development'
LOG_LEVEL = 'INFO'
CACHE_TTL = 300
MAX_ITERATIONS = 1000
TOLERANCE = 0.01

# KPI canonical hard bounds (for optimization and validation).
# These are the single source of truth for every generated operational state:
#   quality     60..100
#   competency  55..100
#   attendance  65..100
#   release     50..100
#   transfer     0..20
# A state outside these bounds is operationally invalid and must NEVER be
# generated, returned as a recommended state, exposed as a feasible
# candidate, or used to claim the target was achieved.
KPI_BOUNDS = {
    'quality': (60, 100),
    'competency': (55, 100),
    'attendance': (65, 100),
    'release': (50, 100),
    'transfer': (0, 20)
}

# Trend thresholds
TREND_THRESHOLDS = {
    'strong_slope': 0.15,
    'moderate_slope': 0.05,
    'volatility_low': 0.05,
    'volatility_medium': 0.15,
    'oscillation_sign_changes': 0.5,
    'spike_std_multiplier': 2.0,
    'recovery_ratio': 1.05,
}

# Sensitivity classification thresholds
SENSITIVITY_THRESHOLDS = {
    'very_high': 1.0,
    'high': 0.5,
    'medium': 0.2,
    'low': 0.05,
}

# Confidence weights for individual metrics (sum to 1.0)
CONFIDENCE_WEIGHTS = {
    'prediction_stability': 0.30,
    'trend_consistency': 0.20,
    'sensitivity_consistency': 0.20,
    'recommendation_agreement': 0.15,
    'strategy_completeness': 0.15,
}

# Confidence component weights for overall score (sum to 1.0)
COMPONENT_WEIGHTS = {
    'forecast': 0.30,
    'trend': 0.20,
    'sensitivity': 0.20,
    'recommendation': 0.15,
    'strategy': 0.15,
}

# Confidence classification thresholds
CONFIDENCE_THRESHOLDS = {
    'very_high': 0.90,
    'high': 0.70,
    'medium': 0.50,
    'low': 0.30,
}

# ================================================================
# RISK ENGINE CONFIGURATION
# ================================================================

# Risk severity classification thresholds (0-1)
RISK_THRESHOLDS = {
    'critical': 0.75,
    'high': 0.55,
    'medium': 0.35,
    'low': 0.15,
}

# Risk component weights for overall risk score (sum to 1.0)
RISK_WEIGHTS = {
    'forecast': 0.25,
    'trend': 0.20,
    'sensitivity': 0.15,
    'recommendation': 0.15,
    'strategy': 0.15,
    'confidence': 0.10,
}

# How to aggregate multiple risk scores into an overall score:
# Supported: 'max', 'weighted_average', 'top3_average'
RISK_AGGREGATION = 'max'

# Weights for calculating individual risk factor score:
# sum must equal 1.0
RISK_SCORE_WEIGHTS = {
    'severity': 0.40,
    'probability': 0.30,
    'impact': 0.30,
}

# Detector thresholds (all configurable)
RISK_DETECTOR_THRESHOLDS = {
    'forecast_cv_threshold': 0.15,
    'forecast_degradation_threshold': 0.9,
    'sensitivity_high_threshold': 1.5,
    'sensitivity_weak_threshold': 0.1,
    'recommendation_overload_threshold': 5,
    'recommendation_scatter_threshold': 3,
    'strategy_complexity_threshold': 0.7,
    'strategy_duration_threshold': 8,
    'confidence_low_threshold': 0.5,
}

# ================================================================
# RISK ENGINE CONFIGURATION
# ================================================================

# Risk severity classification thresholds (0-1)
RISK_THRESHOLDS = {
    'critical': 0.75,
    'high': 0.55,
    'medium': 0.35,
    'low': 0.15,
}

# Risk component weights for overall risk score (sum to 1.0)
RISK_WEIGHTS = {
    'forecast': 0.25,
    'trend': 0.20,
    'sensitivity': 0.15,
    'recommendation': 0.15,
    'strategy': 0.15,
    'confidence': 0.10,
}

# How to aggregate multiple risk factors within a single component:
# Supported: 'max', 'weighted_average', 'top3_average'
COMPONENT_RISK_AGGREGATION = 'max'

# Weights for calculating individual risk factor score:
# sum must equal 1.0
RISK_SCORE_WEIGHTS = {
    'severity': 0.40,
    'probability': 0.30,
    'impact': 0.30,
}

# Detector thresholds (all configurable)
RISK_DETECTOR_THRESHOLDS = {
    'forecast_cv_threshold': 0.15,
    'forecast_degradation_threshold': 0.9,
    'sensitivity_high_threshold': 1.5,
    'sensitivity_weak_threshold': 0.1,
    'recommendation_overload_threshold': 5,
    'recommendation_scatter_threshold': 3,
    'strategy_complexity_threshold': 0.7,
    'strategy_duration_threshold': 8,
    'confidence_low_threshold': 0.5,
}
