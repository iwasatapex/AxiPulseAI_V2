"""
AxiPulseAI – Reverse Optimizer
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from .feature_engineering import align_features
from .inference import predict_single
def reverse_optimize_nps(predictor, target_nps, optimize_factors, fixed_values, total_calls):

    # Default business inputs
    defaults = {
        "quality": 90,
        "competency": 90,
        "attendance": 90,
        "actual_release_rate": 60,
        "transfer_rate": 9,
        "operational_intelligence_factor": 0,
        "business_intelligence_factor": 0,
        "member_intelligence_factor": 0,
        "total_calls_received": 1000,
    }

    allowed = {
    "quality",
    "competency",
    "attendance",
    "actual_release_rate",
    "transfer_rate",
    "operational_intelligence_factor",
    "business_intelligence_factor",
    "member_intelligence_factor",
    "total_calls_received",
}

    invalid = set(optimize_factors) - allowed
    if invalid:
        raise ValueError(f"Invalid factors: {invalid}. Allowed: {allowed}")

    if not predictor.trained:
        raise RuntimeError("Model not trained.")

    def objective(vector):
        vals = {optimize_factors[i]: float(vector[i]) for i in range(len(optimize_factors))}
        row = fixed_values.copy()
        row.update(vals)
        row["total_calls_received"] = total_calls
        if "operational_health" not in row:
            row["operational_health"] = 70
        X = align_features(
            row,
            predictor.feature_names,
            predictor._feature_stats,
            (predictor._history_buffer if predictor._history_buffer is not None else None)
        )
        pred = predict_single(predictor, X, row)
        return abs(pred["nps"] - target_nps)

    bounds = []
    for f in optimize_factors:
        if f == "operational_health":
            bounds.append((
                predictor.config.ops_health_min,
                predictor.config.ops_health_max,
            ))
        elif f in (
            "business_intelligence_factor",
            "member_intelligence_factor",
            "operational_intelligence_factor",
        ):
            bounds.append((
                predictor.config.bc_me_min,
                predictor.config.bc_me_max,
            ))
        elif f in (
            "actual_release_rate",
            "target_release_rate",
            "transfer_rate",
        ):
            bounds.append((
                predictor.config.release_rate_min,
                predictor.config.release_rate_max,
            ))
        elif f == "total_calls_received":
            bounds.append((
                predictor.config.call_volume_min,
                predictor.config.call_volume_max,
            ))
        else:
            bounds.append((0,100))

    result = differential_evolution(
        objective,
        bounds,
        strategy="best1bin",
        maxiter=predictor.config.de_maxiter,
        popsize=predictor.config.de_popsize,
        tol=predictor.config.de_tol,
        seed=42,
        disp=False,
    )

    refined = minimize(
        objective,
        result.x,
        method="Nelder-Mead",
        options={"maxiter": predictor.config.nm_maxiter, "xatol": predictor.config.nm_xatol, "fatol": predictor.config.nm_fatol},
    )

    best = refined.x if refined.success else result.x
    optimized = {optimize_factors[i]: float(best[i]) for i in range(len(optimize_factors))}

    row = fixed_values.copy()
    row.update(optimized)
    row["total_calls_received"] = total_calls
    if "operational_health" not in row:
        row["operational_health"] = 70

    X = align_features(
            row,
            predictor.feature_names,
            predictor._feature_stats,
            (predictor._history_buffer if predictor._history_buffer is not None else None)
        )
    pred = predict_single(predictor, X, row)

    return {
        "optimized_factors": optimized,
        "predicted_nps": pred["nps"],
        "target_nps": target_nps,
        "error": abs(pred["nps"] - target_nps),
    }
