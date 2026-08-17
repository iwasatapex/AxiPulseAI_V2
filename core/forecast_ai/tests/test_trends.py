import unittest
from core.forecast_ai.trends import (
    TrendEngine, TrendSeries, TrendAnalysis, TrendResult,
    TrendAnalyzer, Statistics, PatternDetector, TrendFormatter
)
from core.forecast_ai.config import TREND_THRESHOLDS

class TestStatistics(unittest.TestCase):
    def test_mean(self):
        self.assertAlmostEqual(Statistics.mean([1,2,3,4,5]), 3.0)
    def test_median(self):
        self.assertEqual(Statistics.median([1,3,5,7]), 4.0)
    def test_slope(self):
        # Perfect increasing line: slope should be (y_end - y_start) / (n-1) / mean
        vals = [10, 12, 14, 16, 18]
        slope = Statistics.slope(vals)
        # Expected slope = (18-10)/(4) / 14 = 2/14 ≈ 0.1428
        self.assertAlmostEqual(slope, 0.142857, places=4)
    def test_moving_average(self):
        result = Statistics.moving_average([1,2,3,4,5], window=3)
        expected = [1.0, 1.5, 2.0, 3.0, 4.0]  # expanding then sliding
        self.assertEqual(result, expected)

class TestPatternDetector(unittest.TestCase):
    def test_oscillation(self):
        pattern, _ = PatternDetector.detect([10,20,10,20,10])
        self.assertEqual(pattern, "Oscillation")
    def test_spike(self):
        pattern, _ = PatternDetector.detect([10,12,10,30,10,11])
        self.assertEqual(pattern, "Spike")
    def test_recovery(self):
        pattern, _ = PatternDetector.detect([10,8,7,9,12,14])
        self.assertEqual(pattern, "Recovery")
    def test_plateau(self):
        pattern, _ = PatternDetector.detect([10,10.1,9.9,10.0,10.2])
        self.assertEqual(pattern, "Plateau")
    def test_stable(self):
        pattern, _ = PatternDetector.detect([10,10.5,11,10.8,11.2])
        self.assertEqual(pattern, "Stable")

class TestTrendAnalyzer(unittest.TestCase):
    def test_analyze_increasing(self):
        series = TrendSeries("OH", [80,82,84,86,88], ["d1","d2","d3","d4","d5"])
        analysis = TrendAnalyzer.analyze(series)
        self.assertEqual(analysis.trend_direction, "Strong Increase")
        self.assertEqual(analysis.trend_strength, "Strong")
        # Pattern should be "Stable" or "Plateau" (not trend)
        self.assertIn(analysis.pattern, ["Stable", "Plateau"])

    def test_analyze_decreasing(self):
        series = TrendSeries("NPS", [90,88,86,84,82], ["d1","d2","d3","d4","d5"])
        analysis = TrendAnalyzer.analyze(series)
        self.assertEqual(analysis.trend_direction, "Strong Decrease")
        self.assertEqual(analysis.trend_strength, "Strong")

    def test_analyze_oscillation(self):
        series = TrendSeries("Transfer", [5,10,5,10,5], ["d1","d2","d3","d4","d5"])
        analysis = TrendAnalyzer.analyze(series)
        # Trend direction should be "Stable" (since slope near zero)
        self.assertEqual(analysis.trend_direction, "Stable")
        # Pattern should be Oscillation
        self.assertEqual(analysis.pattern, "Oscillation")

class TestTrendEngine(unittest.TestCase):
    def test_engine(self):
        engine = TrendEngine()
        series_list = [
            TrendSeries("OH", [80,82,84,86,88], ["d1","d2","d3","d4","d5"]),
            TrendSeries("NPS", [70,72,74,73,75], ["d1","d2","d3","d4","d5"])
        ]
        result = engine.analyze(series_list)
        self.assertTrue(result.success)
        self.assertEqual(len(result.analyses), 2)

class TestTrendFormatter(unittest.TestCase):
    def test_formatter(self):
        analysis = TrendAnalysis(
            metric="OH",
            trend_direction="Increase",
            trend_strength="Moderate",
            moving_average=[80,82,84],
            minimum=80.0,
            maximum=88.0,
            mean=84.0,
            median=84.0,
            variance=0.0,
            standard_deviation=0.0,
            volatility="Low",
            absolute_change=8.0,
            percent_change=10.0,
            pattern="Plateau",
            confidence=0.9
        )
        text = TrendFormatter.to_text([analysis])
        self.assertIn("OH", text)
        markdown = TrendFormatter.to_markdown([analysis])
        self.assertIn("## OH", markdown)
        as_dict = TrendFormatter.to_dict([analysis])
        self.assertEqual(as_dict[0]["metric"], "OH")
        json_output = TrendFormatter.to_json([analysis])
        self.assertIn("OH", json_output)

if __name__ == '__main__':
    unittest.main()
