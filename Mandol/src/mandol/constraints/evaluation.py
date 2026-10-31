"""Metrics for the constraint-inference gold set.

The metrics stay independent of the graph and of any model so they can score
recorded predictions in tests and in the evaluation driver alike. Predicates
are canonicalized to hashable tuples, which makes set intersection the whole
comparison: a predicted leaf either matches a gold leaf exactly or counts as a
false positive (and the gold leaf as a false negative).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Collection, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..indexes.timestamp_normalizer import TimestampNormalizer
from .query_constraints import QueryConstraints

Predicate = Tuple[Any, ...]

_NORMALIZER = TimestampNormalizer()
_KIND_INDEX = {"tp": 0, "fp": 1, "fn": 2}


@dataclass(frozen=True)
class TypeScore:
    """Precision/recall/F1 over one predicate kind (or all of them)."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass(frozen=True)
class ExtractionScore:
    """Overall extraction score plus a breakdown by predicate kind."""

    overall: TypeScore = field(default_factory=TypeScore)
    by_type: Dict[str, TypeScore] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.overall.to_dict(),
            "by_type": {name: score.to_dict() for name, score in sorted(self.by_type.items())},
        }


def canonical_predicates(constraints: Optional[QueryConstraints]) -> Set[Predicate]:
    """Flatten constraints into a hashable predicate set for comparison."""
    if constraints is None:
        return set()
    leaves: Set[Predicate] = set()
    time_range = constraints.time_range
    if time_range is not None:
        leaves.add(
            (
                "time",
                time_range.field,
                _bound(time_range.start),
                _bound(time_range.end),
            )
        )
    for name, spec in (constraints.metadata_filters or {}).items():
        for operator, value in spec.items():
            leaves.add(_filter_predicate(name, operator, value))
    for name in constraints.space_names or []:
        leaves.add(("space", str(name)))
    return leaves


def score_predictions(
    pairs: Iterable[Tuple[Set[Predicate], Set[Predicate]]],
) -> ExtractionScore:
    """Score ``(predicted, expected)`` predicate-set pairs."""
    totals = {"tp": 0, "fp": 0, "fn": 0}
    by_type: Dict[str, List[int]] = {}
    for predicted, expected in pairs:
        predicted_set, expected_set = set(predicted), set(expected)
        matches = (
            ("tp", predicted_set & expected_set),
            ("fp", predicted_set - expected_set),
            ("fn", expected_set - predicted_set),
        )
        for kind, items in matches:
            totals[kind] += len(items)
            for leaf in items:
                by_type.setdefault(str(leaf[0]), [0, 0, 0])[_KIND_INDEX[kind]] += 1
    return ExtractionScore(
        overall=TypeScore(**totals),
        by_type={
            name: TypeScore(tp=counts[0], fp=counts[1], fn=counts[2])
            for name, counts in sorted(by_type.items())
        },
    )


def score_extraction(
    predicted: Iterable[Optional[QueryConstraints]],
    expected: Iterable[Optional[QueryConstraints]],
) -> ExtractionScore:
    """Score constraint bundles directly, one pair per query."""
    return score_predictions(
        (canonical_predicates(p), canonical_predicates(e))
        for p, e in zip(predicted, expected)
    )


def empty_result_rate(result_counts: Iterable[int]) -> float:
    """Fraction of runs that returned no results."""
    counts = [int(count) for count in result_counts]
    if not counts:
        return 0.0
    return sum(1 for count in counts if count <= 0) / len(counts)


def recall_at_k(
    retrieved: Sequence[str],
    relevant: Collection[str],
    k: int,
) -> float:
    """Share of ``relevant`` UIDs present in the first ``k`` results.

    An empty relevant set counts as fully recalled: there is nothing a retriever
    could have missed.
    """
    if not relevant:
        return 1.0
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))


def _filter_predicate(name: str, operator: Any, value: Any) -> Predicate:
    operator = str(operator or "eq").strip().lower()
    if operator in {"in", "nin"} and isinstance(value, (list, tuple, set)):
        value = tuple(sorted(value, key=repr))
    return ("filter", str(name), operator, value)


def _bound(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 3)
    if hasattr(value, "timestamp"):
        return round(float(value.timestamp()), 3)
    try:
        normalized = _NORMALIZER.normalize(value)
    except Exception:
        normalized = None
    if isinstance(normalized, (int, float)) and not isinstance(normalized, bool):
        return round(float(normalized), 3)
    return value
