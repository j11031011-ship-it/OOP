"""Sorted value index for ordering operators (``gt`` / ``gte`` / ``lt`` / ``lte``).

Values are normalized to comparable floats by ``TimestampNormalizer`` and kept
in one tuple-sorted list, so every ordering condition becomes a pair of
``bisect`` boundaries instead of a payload scan. The index is the P1 half of
the time-range resolver: ``TimeRange`` windows arrive as ``gte`` / ``lte``
conditions and are answered from the sorted structure when the field is
registered and both bounds normalize.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from typing import Any, Dict, List, Optional, Tuple

from ..constraints.candidate_set import CandidateSet
from ..utils.logging_config import create_module_logger
from .base_index import BaseIndex
from .timestamp_normalizer import TimestampNormalizer

logger = create_module_logger("sorted_index")

_OPERATORS = frozenset({"gt", "gte", "lt", "lte"})


class SortedIndex(BaseIndex):
    """Bisect-backed ordering index over normalized numeric values.

    Items are ``(epoch, uid)`` tuples kept sorted by value; bisect boundaries
    are computed on the value alone (``key``-based bisect), and the half-open
    slice ``[lo, hi)`` is the answer. Units whose value does not normalize are
    skipped at insertion time and therefore never match an ordering predicate.
    """

    def __init__(self, field: str, normalizer: Optional[TimestampNormalizer] = None):
        """Initialize the index.

        Args:
            field: Payload field this index serves.
            normalizer: Value normalizer; defaults to ``TimestampNormalizer``.
        """
        super().__init__(field)
        self.normalizer = normalizer or TimestampNormalizer()
        self._items: List[Tuple[float, str]] = []
        self._epochs: Dict[str, float] = {}

    @property
    def uid_count(self) -> int:
        """Number of UIDs currently indexed."""
        return len(self._epochs)

    def add(self, uid: str, value: Any) -> None:
        """Index ``value`` for ``uid``, replacing any previous entry."""
        uid = str(uid)
        if uid in self._epochs:
            self.remove(uid)
        epoch = self.normalizer.normalize(value)
        if epoch is None:
            logger.debug(
                "Unit %s skipped from sorted index %r: value does not normalize.",
                uid,
                self.field,
            )
            return
        self._epochs[uid] = epoch
        insort(self._items, (epoch, uid))

    def remove(self, uid: str) -> None:
        """Drop ``uid`` from the index if present."""
        uid = str(uid)
        epoch = self._epochs.pop(uid, None)
        if epoch is None:
            return
        position = bisect_left(self._items, (epoch, uid))
        if position < len(self._items) and self._items[position] == (epoch, uid):
            self._items.pop(position)

    def can_serve(self, conditions: Dict[str, Any]) -> bool:
        """Serve non-empty ordering conditions whose bounds all normalize."""
        if not conditions:
            return False
        for operator, expected in conditions.items():
            if operator not in _OPERATORS:
                return False
            if self.normalizer.normalize(expected) is None:
                return False
        return True

    def lookup(self, conditions: Dict[str, Any]) -> CandidateSet:
        """Evaluate the ordering conditions against the sorted slice."""
        bounds = self._bounds(conditions)
        if bounds is None:
            return CandidateSet.from_iterable((), source=f"index:{self.field}")
        lo, hi = bounds
        return CandidateSet.from_iterable(
            (uid for _, uid in self._items[lo:hi]), source=f"index:{self.field}"
        )

    def estimate_selectivity(self, conditions: Dict[str, Any]) -> Optional[float]:
        """Return the match fraction over indexed units."""
        if not self._items:
            return None
        bounds = self._bounds(conditions)
        if bounds is None:
            return 0.0
        lo, hi = bounds
        return (hi - lo) / len(self._items)

    def _bounds(self, conditions: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """Return the ``[lo, hi)`` slice satisfying every condition."""
        lo, hi = 0, len(self._items)
        for operator, expected in conditions.items():
            epoch = self.normalizer.normalize(expected)
            if epoch is None:
                return None
            if operator == "gt":
                lo = max(lo, bisect_right(self._items, epoch, key=lambda item: item[0]))
            elif operator == "gte":
                lo = max(lo, bisect_left(self._items, epoch, key=lambda item: item[0]))
            elif operator == "lt":
                hi = min(hi, bisect_left(self._items, epoch, key=lambda item: item[0]))
            elif operator == "lte":
                hi = min(hi, bisect_right(self._items, epoch, key=lambda item: item[0]))
            else:
                raise ValueError(
                    f"SortedIndex cannot evaluate operator {operator!r} "
                    f"on field {self.field!r}."
                )
        if lo >= hi:
            return None
        return lo, hi