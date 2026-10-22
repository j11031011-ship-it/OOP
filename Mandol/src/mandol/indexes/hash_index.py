"""Equality index for metadata fields (``eq`` / ``in`` / ``ne`` / ``nin``).

The structure is a value-to-UIDs hash map mirroring ``==`` semantics:
buckets are keyed by a hashable projection of the payload value, so Python's
own equality rules (``1 == True``, ``None`` for missing fields) carry over.
Lists, tuples, mappings, and sets are normalized recursively; values that
cannot be projected (for example arbitrary objects) fall back to the scan so
equality remains exact. Units are tracked in a full UID set as well, which
turns the negative operators into complements over the indexed population.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Set

from ..constraints.candidate_set import CandidateSet
from .base_index import BaseIndex

_OPERATORS = frozenset({"eq", "in", "ne", "nin"})
_UNNORMALIZABLE = object()


@dataclass(frozen=True)
class _NormalizedValue:
    """Hashable projection preserving the equality semantics of a container."""

    kind: str
    payload: Any


def _normalize(value: Any, seen: Optional[Set[int]] = None) -> Any:
    """Return a hashable value that compares like ``value``, or a sentinel.

    Lists keep list identity (a list never equals a tuple), sets and frozensets
    share one projection (they compare equal in Python), and all mappings share
    one projection. Scalars that are already hashable are returned unchanged,
    which preserves Python's cross-type equality such as ``1 == True``.
    """
    if seen is None:
        seen = set()
    if not isinstance(value, (list, tuple, set, frozenset, Mapping)):
        try:
            hash(value)
        except TypeError:
            return _UNNORMALIZABLE
        return value

    if id(value) in seen:
        return _UNNORMALIZABLE
    seen = seen | {id(value)}

    if isinstance(value, list):
        items = [_normalize(item, seen) for item in value]
        if any(item is _UNNORMALIZABLE for item in items):
            return _UNNORMALIZABLE
        return _NormalizedValue("list", tuple(items))
    if isinstance(value, tuple):
        items = [_normalize(item, seen) for item in value]
        if any(item is _UNNORMALIZABLE for item in items):
            return _UNNORMALIZABLE
        return _NormalizedValue("tuple", tuple(items))
    if isinstance(value, (set, frozenset)):
        items = {_normalize(item, seen) for item in value}
        if any(item is _UNNORMALIZABLE for item in items):
            return _UNNORMALIZABLE
        return _NormalizedValue("set", frozenset(items))
    if isinstance(value, Mapping):
        pairs = {
            (_normalize(key, seen), _normalize(item, seen))
            for key, item in value.items()
        }
        if any(
            key is _UNNORMALIZABLE or item is _UNNORMALIZABLE
            for key, item in pairs
        ):
            return _UNNORMALIZABLE
        return _NormalizedValue("mapping", frozenset(pairs))
    try:
        hash(value)
    except TypeError:
        return _UNNORMALIZABLE
    return value


class HashIndex(BaseIndex):
    """Value-bucket index serving equality and set-membership operators.

    Container-valued fields (``entity_ids``, tags, structured attributes) are
    projected to hashable keys instead of disabling the whole field. Only values
    that cannot be projected at all force the resolver back to the scan; that is
    the only way to keep ``eq`` results identical for such payloads.
    """

    def __init__(self, field: str):
        """Initialize the index.

        Args:
            field: Payload field this index serves.
        """
        super().__init__(field)
        self._buckets: Dict[Any, Set[str]] = {}
        self._uid_keys: Dict[str, Any] = {}
        self._all_uids: Set[str] = set()
        self._unhashable_uids: Set[str] = set()

    @property
    def uid_count(self) -> int:
        """Number of UIDs currently indexed."""
        return len(self._all_uids)

    def add(self, uid: str, value: Any) -> None:
        """Index ``value`` for ``uid``, replacing any previous entry."""
        uid = str(uid)
        if uid in self._all_uids:
            self.remove(uid)
        self._all_uids.add(uid)
        key = _normalize(value)
        self._uid_keys[uid] = key
        if key is _UNNORMALIZABLE:
            self._unhashable_uids.add(uid)
        else:
            self._buckets.setdefault(key, set()).add(uid)

    def remove(self, uid: str) -> None:
        """Drop ``uid`` from the index if present."""
        uid = str(uid)
        if uid not in self._all_uids:
            return
        key = self._uid_keys.pop(uid)
        self._all_uids.discard(uid)
        self._unhashable_uids.discard(uid)
        if key is _UNNORMALIZABLE:
            return
        bucket = self._buckets.get(key)
        if bucket is not None:
            bucket.discard(uid)
            if not bucket:
                del self._buckets[key]

    def can_serve(self, conditions: Dict[str, Any]) -> bool:
        """Serve non-empty conditions whose operands are projectable."""
        if not conditions or self._unhashable_uids:
            return False
        for operator, expected in conditions.items():
            if operator not in _OPERATORS:
                return False
            if operator in ("in", "nin"):
                if not isinstance(expected, (list, tuple, set, frozenset)):
                    return False
                if any(_normalize(item) is _UNNORMALIZABLE for item in expected):
                    return False
            elif _normalize(expected) is _UNNORMALIZABLE:
                return False
        return True

    def lookup(self, conditions: Dict[str, Any]) -> CandidateSet:
        """Evaluate the conditions, mirroring ``filter_memory_units`` semantics."""
        matched: Optional[Set[str]] = None
        for operator, expected in conditions.items():
            part = self._match_set(operator, expected)
            matched = part if matched is None else matched & part
        return CandidateSet.from_iterable(
            matched or (), source=f"index:{self.field}"
        )

    def estimate_selectivity(self, conditions: Dict[str, Any]) -> Optional[float]:
        """Return the match fraction over indexed units."""
        if not self._all_uids:
            return None
        return len(self.lookup(conditions)) / len(self._all_uids)

    def _match_set(self, operator: str, expected: Any) -> Set[str]:
        """Return the UIDs satisfying one operator condition."""
        if operator == "eq":
            key = _normalize(expected)
            return set(self._buckets.get(key, ())) if key is not _UNNORMALIZABLE else set()
        if operator == "in":
            return {
                uid
                for item in expected
                if (key := _normalize(item)) is not _UNNORMALIZABLE
                for uid in self._buckets.get(key, ())
            }
        if operator == "ne":
            key = _normalize(expected)
            matched = (
                set(self._buckets.get(key, ()))
                if key is not _UNNORMALIZABLE
                else set()
            )
            return self._all_uids - matched
        if operator == "nin":
            excluded = {
                uid
                for item in expected
                if (key := _normalize(item)) is not _UNNORMALIZABLE
                for uid in self._buckets.get(key, ())
            }
            return self._all_uids - excluded
        raise ValueError(
            f"HashIndex cannot evaluate operator {operator!r} on field {self.field!r}."
        )
