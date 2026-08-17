import unittest
from dataclasses import dataclass, field

from core.forecast_ai.models import ForecastRequest

from core.forecast_ai.engines.report_engine import ReportEngine
from core.forecast_ai.engines.sensitivity_engine import SensitivityEngine
from core.forecast_ai.engines.trend_engine import TrendEngine


@dataclass
class DummyReportResult:
    success: bool = True
    title: str = "Test Report"
    executive_summary: object = None
    sections: list = field(default_factory=list)
    metadata: object = None
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)


class DummyReportCore:
    def generate(self, **kwargs):
        return DummyReportResult()


@dataclass
class DummySensitivityResult:
    success: bool = True
    analyses: list = field(default_factory=list)
    ranking: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class DummySensitivityCore:
    def analyze(self, state, metrics=None):
        return DummySensitivityResult()


@dataclass
class DummyTrendResult:
    success: bool = True
    analyses: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class DummyTrendCore:
    def analyze(self, series):
        return DummyTrendResult()


class TestAnalyticsEngines(unittest.TestCase):

    def test_report_missing_parameters(self):
        resp = ReportEngine(core=DummyReportCore()).execute(
            ForecastRequest(operation="report")
        )
        self.assertFalse(resp.success)

    def test_report_success(self):
        resp = ReportEngine(core=DummyReportCore()).execute(
            ForecastRequest(
                operation="report",
                parameters={}
            )
        )
        self.assertTrue(resp.success)
        self.assertEqual(resp.engine, "ReportEngine")

    def test_sensitivity_missing_state(self):
        resp = SensitivityEngine(sen_core=DummySensitivityCore()).execute(
            ForecastRequest(
                operation="sensitivity",
                parameters={}
            )
        )
        self.assertFalse(resp.success)

    def test_sensitivity_success(self):
        resp = SensitivityEngine(sen_core=DummySensitivityCore()).execute(
            ForecastRequest(
                operation="sensitivity",
                parameters={"state": {"quality": 85}}
            )
        )
        self.assertTrue(resp.success)

    def test_trend_missing_series(self):
        resp = TrendEngine(trend_core=DummyTrendCore()).execute(
            ForecastRequest(
                operation="trend",
                parameters={}
            )
        )
        self.assertFalse(resp.success)

    def test_trend_success(self):
        resp = TrendEngine(trend_core=DummyTrendCore()).execute(
            ForecastRequest(
                operation="trend",
                parameters={
                    "series": [
                        {
                            "metric": "quality",
                            "values": [80, 81],
                            "timestamps": ["d1", "d2"]
                        }
                    ]
                }
            )
        )
        self.assertTrue(resp.success)
        self.assertEqual(resp.engine, "TrendEngine")


if __name__ == "__main__":
    unittest.main()
