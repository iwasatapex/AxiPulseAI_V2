from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[2]


def _py_files(package):
    return (ROOT / package).rglob("*.py")


def test_nps_has_no_private_axi_probabilistic_implementation():
    text = (ROOT / "core/nps_predictor/inference.py").read_text()
    for name in (
        "_axi_bayesian_update_0_10",
        "_axi_monte_carlo_0_10",
        "_axi_attach_0_10_probabilistic_analysis",
    ):
        assert name not in text


def test_v3_has_no_nps_probabilistic_dependency():
    for path in _py_files("core/decision_intelligence/v3"):
        text = path.read_text()
        assert "core.nps_predictor" not in text
        assert "core.nps_predictor.inference" not in text


def test_domain_packages_do_not_instantiate_scalar_engines():
    forbidden = {
        "BayesianInferenceEngine",
        "MonteCarloEngine",
    }
    packages = (
        "core/nps_predictor",
        "core/operation_health_predictor",
        "core/forecast_ai",
        "core/decision_intelligence/v3",
    )
    for package in packages:
        root = ROOT / package
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id not in forbidden, f"forbidden engine instantiation: {path}:{node.lineno}"
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    assert node.func.attr not in forbidden, f"forbidden engine instantiation: {path}:{node.lineno}"


def test_only_universal_adapter_instantiates_scalar_engines():
    text = (ROOT / "core/probabilistic/adapter.py").read_text()
    assert "BayesianInferenceEngine()" in text
    assert "MonteCarloEngine()" in text
