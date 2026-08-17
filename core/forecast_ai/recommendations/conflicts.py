"""
Conflict detection between recommendations (Phase 10).

Two recommendations conflict when they drive the SAME target KPI in OPPOSITE
directions. Detection prefers the structured fields on ``Recommendation``
(``target_kpi`` + ``direction``); keyword parsing is used only as a
compatibility fallback for recommendations that lack structured metadata.

Keyword fallback also handles common negations ("don't increase", "avoid
raising", "no decrease") so that negated directional language is not treated
as a genuine directional intent.
"""
from typing import List, Tuple
from .models import Recommendation


# KPI alias map for the keyword fallback.
_KPI_ALIASES = {
    "quality": ["quality"],
    "competency": ["competency", "competence", "skill"],
    "attendance": ["attendance"],
    "release": ["release rate", "release", "throughput"],
    "transfer": ["transfer rate", "transfer"],
    "nps": ["nps", "net promoter"],
    "operations_health": ["operations health", "operational health", "oh"],
}

_DIRECTION_UP = {
    "increase", "raise", "boost", "improve", "expand", "grow",
}
_DIRECTION_DOWN = {
    "decrease", "reduce", "lower", "cut", "shrink", "minimize",
}
_NEGATIONS = {
    "don't", "do not", "dont", "avoid", "no", "not", "never", "without",
    "instead of", "stop",
}


class ConflictDetector:
    @staticmethod
    def structured_direction(rec: Recommendation) -> Tuple[str, str] | None:
        """Return (kpi, direction) from a recommendation's structured fields.

        ``direction`` is normalized to ``increase`` or ``decrease`` even when
        the caller supplied ``improve``/``reduce`` etc. Returns None when the
        structured metadata is incomplete.
        """
        kpi = (rec.target_kpi or "").strip().lower().replace(" ", "_")
        direction = (rec.direction or "").strip().lower()
        if not kpi or not direction:
            return None

        if direction in _DIRECTION_UP or direction in {
            "up", "increase", "raise", "boost", "improve", "expand", "grow",
        }:
            return (kpi, "increase")
        if direction in _DIRECTION_DOWN or direction in {
            "down", "decrease", "reduce", "lower", "cut", "shrink", "minimize",
        }:
            return (kpi, "decrease")
        return None

    @staticmethod
    def _keyword_directions(rec: Recommendation) -> set:
        """Keyword fallback: return {(kpi, direction)} for directional
        statements, ignoring negated phrasing."""
        text = " ".join([
            rec.description or "",
            rec.reasoning or "",
            " ".join(rec.actions or []),
        ]).lower()

        # Build negation-aware token stream by removing negated windows.
        import re
        tokens = re.findall(r"\w+", text)
        negated = set()
        for i, token in enumerate(tokens):
            if token in _NEGATIONS:
                # The next 1-3 tokens fall into a negated phrase.
                for j in range(i + 1, min(i + 4, len(tokens))):
                    negated.add(tokens[j])

        out = set()
        words = set(tokens) - negated
        for kpi, aliases in _KPI_ALIASES.items():
            present = any(alias in text for alias in aliases)
            if not present:
                continue
            # Count directional keywords only outside negated windows.
            up_hits = sum(
                1 for w in words if w in _DIRECTION_UP
            )
            down_hits = sum(
                1 for w in words if w in _DIRECTION_DOWN
            )
            if up_hits and not down_hits:
                out.add((kpi, "increase"))
            elif down_hits and not up_hits:
                out.add((kpi, "decrease"))
        return out

    @classmethod
    def detect_conflicts(
        cls,
        recommendations: List[Recommendation],
    ) -> List[Tuple[Recommendation, Recommendation, str]]:
        """Return conflicting (rec_a, rec_b, reason) tuples.

        Structured fields take precedence; keyword parsing is the fallback.
        Two recommendations conflict only when they share a KPI with opposite
        directions.
        """
        if not recommendations:
            return []

        directions = []
        for rec in recommendations:
            structured = cls.structured_direction(rec)
            if structured is not None:
                directions.append((rec, {structured}))
            else:
                directions.append((rec, cls._keyword_directions(rec)))

        conflicts: List[Tuple[Recommendation, Recommendation, str]] = []
        n = len(directions)
        for i in range(n):
            for j in range(i + 1, n):
                rec_a, dir_a_set = directions[i]
                rec_b, dir_b_set = directions[j]
                for kpi, d_a in list(dir_a_set):
                    opposite = "decrease" if d_a == "increase" else "increase"
                    if (kpi, opposite) in dir_b_set:
                        conflicts.append((
                            rec_a,
                            rec_b,
                            f"Conflicting directions on {kpi}: "
                            f"'{rec_a.title or 'rec-a'}' wants {d_a}, while "
                            f"'{rec_b.title or 'rec-b'}' wants the opposite.",
                        ))
        return conflicts


detect_conflicts = ConflictDetector.detect_conflicts

