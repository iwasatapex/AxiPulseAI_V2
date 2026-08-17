"""
KPI state transition layer.

Applies controlled momentum between forecast steps.
No OH/NPS generation.
No circular dependency.
"""

from dataclasses import dataclass


@dataclass
class KPITransition:
    autocorrelation: float = 0.6

    def _move(self, current, target, minimum, maximum):
        value = (
            self.autocorrelation * current
            + (1 - self.autocorrelation) * target
        )
        return max(minimum, min(maximum, value))

    def apply(self, state):
        return {
            "quality": self._move(
                state["quality"], 87.0, 60.0, 100.0
            ),
            "competency": self._move(
                state["competency"], 93.0, 55.0, 100.0
            ),
            "attendance": self._move(
                state["attendance"], 90.0, 65.0, 100.0
            ),
            "release": self._move(
                state["release"], 60.0, 50.0, 100.0
            ),
            "transfer": self._move(
                state["transfer"], 9.0, 0.0, 20.0
            ),
        }
