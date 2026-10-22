"""Base contract for filterable payload-field indexes.

Every index answers the same question for one payload field — "which UIDs
satisfy this operator condition?" — so ``MetadataIndex`` can route decoded
query filters to a strategy without knowing how each structure is built. The
linear-scan resolvers remain the fallback: an index only ever *replaces* a scan
when it can guarantee the same answer (see ``can_serve``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..constraints.candidate_set import CandidateSet


class BaseIndex(ABC):
    """Strategy contract for one filterable payload field.

    Concrete strategies receive resolved payload values through ``add`` /
    ``remove`` (mount points live in ``SemanticMap`` so indexes track the L1
    population incrementally) and evaluate query-side conditions through
    ``lookup``. ``estimate_selectivity`` exposes the match fraction without
    materializing UIDs, which the scheduling milestone uses to compare plans.
    """

    def __init__(self, field: str):
        """Initialize the index.

        Args:
            field: Payload field this index serves.
        """
        self.field = field

    @abstractmethod
    def add(self, uid: str, value: Any) -> None:
        """Index ``value`` for ``uid``, replacing any previous entry."""

    @abstractmethod
    def remove(self, uid: str) -> None:
        """Drop ``uid`` from the index if present."""

    @abstractmethod
    def can_serve(self, conditions: Dict[str, Any]) -> bool:
        """Return whether every operator condition can be evaluated here.

        ``MetadataIndex`` falls back to the linear resolver whenever any field
        of a filter bundle answers ``False``, so ``can_serve`` is the single
        place where a strategy declares its operator vocabulary *and* its
        data-shape guarantees (hashable values, normalizable bounds, ...).
        """

    @abstractmethod
    def lookup(self, conditions: Dict[str, Any]) -> CandidateSet:
        """Evaluate one field's operator conditions (precondition: ``can_serve``)."""

    @abstractmethod
    def estimate_selectivity(self, conditions: Dict[str, Any]) -> Optional[float]:
        """Return the fraction of indexed units matching, or ``None`` when unknown."""