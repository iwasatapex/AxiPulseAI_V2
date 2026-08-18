from core.forecast_ai.sensitivity.models import SensitivityAnalysis
from core.forecast_ai.sensitivity.ranking import SensitivityRanker


def _analysis(metric, sensitivity):
    return SensitivityAnalysis(
        metric=metric,
        baseline_output_oh=80.0,
        baseline_output_nps=70.0,
        modified_output_oh=80.0,
        modified_output_nps=70.0,
        operations_health_change=0.0,
        nps_change=0.0,
        sensitivity_score_oh=sensitivity,
        sensitivity_score_nps=0.0,
        elasticity_oh=0.0,
        elasticity_nps=0.0,
    )


def test_transfer_improvement_direction_is_decrease():
    analysis = _analysis("transfer", 0.1191)

    assert analysis.improvement_direction == "decrease"
    assert analysis.sensitivity_score_oh == 0.1191
    assert analysis.improvement_sensitivity_oh == -0.1191


def test_positive_kpi_improvement_direction_is_increase():
    for metric in ("quality", "competency", "attendance", "release"):
        analysis = _analysis(metric, -0.15)
        assert analysis.improvement_direction == "increase"
        assert analysis.sensitivity_score_oh == -0.15
        assert analysis.improvement_sensitivity_oh == -0.15


def test_ranker_uses_actual_sensitivity_magnitude():
    analyses = [
        _analysis("quality", 0.05),
        _analysis("competency", -0.1585),
        _analysis("attendance", -0.1324),
        _analysis("release", -0.0217),
        _analysis("transfer", 0.1191),
    ]

    ranked = SensitivityRanker.rank(analyses)

    assert [a.metric for a in ranked] == [
        "competency",
        "attendance",
        "transfer",
        "quality",
        "release",
    ]
