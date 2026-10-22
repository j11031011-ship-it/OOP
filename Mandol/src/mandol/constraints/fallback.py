"""Degenerate-candidate fallback policy (P3).

By default a query whose predicates are mutually unsatisfiable yields an empty
candidate set and therefore no results — the honest answer for a hard AND. This
policy makes the alternative configurable: relax the query by dropping the
predicate that contributes most to the degeneracy, logging a warning for every
dropped predicate, so the caller gets a best-effort answer plus an audit trail
instead of silence.
"""

from __future__ import annotations

from dataclasses import dataclass

RETURN_EMPTY = "return_empty"
DROP_CONSTRAINT = "drop_constraint"

_ON_DEGENERATE_CHOICES = (RETURN_EMPTY, DROP_CONSTRAINT)


@dataclass(frozen=True)
class FallbackPolicy:
    """What to do when the intersected candidate set is degenerate.

    A candidate set is *degenerate* when it holds fewer than ``min_candidates``
    UIDs — most notably when it is empty because two predicates cannot hold at
    once.

    Attributes:
        on_degenerate: ``"return_empty"`` (default) keeps the degenerate set, so
            retrieval returns nothing; ``"drop_constraint"`` relaxes the query by
            dropping one predicate at a time until the set is no longer
            degenerate.
        min_candidates: Size below which the candidate set counts as degenerate.
            ``1`` (default) means "empty only".
    """

    on_degenerate: str = RETURN_EMPTY
    min_candidates: int = 1

    def __post_init__(self) -> None:
        if self.on_degenerate not in _ON_DEGENERATE_CHOICES:
            raise ValueError(
                f"on_degenerate must be one of {_ON_DEGENERATE_CHOICES}, "
                f"got {self.on_degenerate!r}."
            )
        if self.min_candidates < 0:
            raise ValueError(
                f"min_candidates must be non-negative, got {self.min_candidates}."
            )

    @property
    def relaxes_degenerate_queries(self) -> bool:
        """Whether this policy drops predicates instead of returning nothing."""
        return self.on_degenerate == DROP_CONSTRAINT
