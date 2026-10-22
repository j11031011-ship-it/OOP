"""Candidate-set algebra shared by constraint resolvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, Iterator, Optional


@dataclass(frozen=True)
class CandidateSet:
    """Immutable UID set produced by one constraint predicate.

    The value object decouples constraint evaluation from retrieval: resolvers
    produce candidate sets, the planner combines them through set algebra, and
    the retrieval layer only ever sees plain UIDs. ``source`` records the
    provenance of the set for diagnostics, and ``total`` (when known) yields a
    selectivity estimate for future pre-filter decisions.

    Attributes:
        uids: Candidate UIDs.
        source: Human-readable provenance label.
        total: Optional corpus size used for selectivity estimation.
    """

    uids: FrozenSet[str]
    source: str = "unknown"
    total: Optional[int] = None

    @classmethod
    def from_iterable(
        cls,
        uids: Iterable[str],
        source: str = "unknown",
        total: Optional[int] = None,
    ) -> "CandidateSet":
        """Build a candidate set from any UID iterable."""
        return cls(frozenset(str(uid) for uid in uids), source=source, total=total)

    def __len__(self) -> int:
        return len(self.uids)

    def __contains__(self, uid: str) -> bool:
        return uid in self.uids

    def __iter__(self) -> Iterator[str]:
        return iter(self.uids)

    @property
    def selectivity(self) -> Optional[float]:
        """Return ``len(uids) / total`` when the corpus size is known."""
        if not self.total:
            return None
        return len(self.uids) / self.total

    def intersect(self, *others: "CandidateSet") -> "CandidateSet":
        """Return the intersection with the given candidate sets."""
        uids = self.uids
        sources = [self.source]
        for other in others:
            uids = uids & other.uids
            sources.append(other.source)
        return CandidateSet(uids, source=" & ".join(sources), total=self._merge_total(others))

    def union(self, *others: "CandidateSet") -> "CandidateSet":
        """Return the union with the given candidate sets."""
        uids = self.uids
        sources = [self.source]
        for other in others:
            uids = uids | other.uids
            sources.append(other.source)
        return CandidateSet(uids, source=" | ".join(sources), total=self._merge_total(others))

    def difference(self, other: "CandidateSet") -> "CandidateSet":
        """Return the UIDs of this set that are absent from ``other``."""
        return CandidateSet(
            self.uids - other.uids,
            source=f"{self.source} - {other.source}",
            total=self._merge_total((other,)),
        )

    def _merge_total(self, others: Iterable["CandidateSet"]) -> Optional[int]:
        """Carry the corpus size across set operations when any input knows it."""
        for candidate in (self, *others):
            if candidate.total is not None:
                return candidate.total
        return None