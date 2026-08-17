"""
Basic integration tests for AxiPulseAI Predictor.
Run with: pytest tests/
"""

import os
import tempfile
import pandas as pd
import numpy as np
from core.operation_health_predictor import OperationalHealthPredictor

def create_sample_data(n=10):
    """Create a small sample dataset for testing."""
    data = {
        "date": pd.date_range("2025-01-01", periods=n),
        "target_quality": [85]*n,
        "actual_quality": np.random.randint(60, 95, n),
        "target_competency": [82]*n,
        "actual_competency": np.random.randint(60, 90, n),
        "target_attendance": [92]*n,
        "actual_attendance": np.random.randint(70, 98, n),
        "target_release_rate": [60]*n,
        "actual_release_rate": np.random.randint(55, 80, n),
        "target_transfer_rate": [9]*n,
        "actual_transfer_rate": np.random.randint(5, 14, n),
        "total_calls_received": np.random.randint(2000, 5000, n),
        "operational_intelligence_factor": np.random.randint(-10, 10, n),
        "issue_type_ucard": np.random.randint(10, 40, n),
        "issue_type_claims": np.random.randint(20, 50, n),
        "issue_type_enrollment": np.random.randint(10, 30, n),
        "issue_type_disenrollment": np.random.randint(5, 20, n),
        "operational_health": np.random.randint(80, 110, n),
    }
    df = pd.DataFrame(data)
    # Ensure issue types sum to 100
    for i in range(len(df)):
        s = df.iloc[i][['issue_type_ucard','issue_type_claims','issue_type_enrollment','issue_type_disenrollment']].sum()
        if s != 100:
            df.loc[i, 'issue_type_disenrollment'] += (100 - s)
    return df

def test_end_to_end():
    """Test training, prediction, reverse, and persistence."""
    predictor = OperationalHealthPredictor()

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        df = create_sample_data(15)
        df.to_csv(f.name, index=False)
        tmp_path = f.name

    # Train
    predictor.train(tmp_path, tune=False)
    assert predictor.trained
    assert predictor.model_name is not None

    # Predict
    row = df.iloc[0].to_dict()
    # date needs to be string
    row['date'] = row['date'].strftime('%Y-%m-%d')
    score = predictor.predict(row, apply_oif=True)
    assert np.isfinite(score)

    # Leaderboard
    lb, failed = predictor.predict_leaderboard(row)
    assert isinstance(lb, dict)
    assert isinstance(failed, list)
    assert len(lb) > 0

    # Reverse
    result = predictor.reverse_optimize(
        target_score=105,
        optimize_factors=['actual_quality'],
        fixed_values={k: v for k, v in row.items() if k not in ['actual_quality', 'date']}
    )
    assert 'optimized_factors' in result
    assert np.isfinite(result['predicted_score'])

    # Save/Load
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as m:
        model_path = m.name
    predictor.save_model(model_path)
    predictor2 = OperationalHealthPredictor()
    predictor2.load_model(model_path)
    assert predictor2.trained
    assert predictor2.model_name == predictor.model_name

    # Cleanup
    os.unlink(tmp_path)
    os.unlink(model_path)

    print("✅ All tests passed!")

if __name__ == "__main__":
    test_end_to_end()