"""Constraint-aware planner for constrained hybrid retrieval.

The planner is the coordination layer requested by the assignment: it resolves
every declared constraint into a ``CandidateSet``, intersects the sets (AND
semantics), and drives ``MultiRetriever.smart_search`` with the resulting
execution plan. Selectivity decides how the candidate set reaches retrieval:
sets covering less than the configured fraction of the corpus are pushed down
as the ``candidate_uids`` pre-filter, while broad sets are applied to fused
results afterwards (post-filter). The post-filter pool then scales with
selectivity so ``top_k`` can still be filled, and a configurable
``FallbackPolicy`` decides what happens when the intersection is degenerate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ..utils.logging_config import create_module_logger
from .candidate_set import CandidateSet
from .fallback import FallbackPolicy
from .query_constraints import QueryConstraints
from .resolvers import (
    BaseConstraintResolver,
    MetadataConstraintResolver,
    RelationConstraintResolver,
    SpaceConstraintResolver,
    TimeRangeConstraintResolver,
)

if TYPE_CHECKING:
    from ..core.memory_unit import MemoryUnit
    from ..core.semantic_graph import SemanticGraph

logger = create_module_logger("constraint_planner")

DEFAULT_PREFILTER_SELECTIVITY_THRESHOLD = 0.2


def _expand_top_k(top_k: int) -> int:
    """Return the retrieval pool size used before post-filtering.

    Mirrors the expansion already applied by ``_execute_base_retrieval``
    (``max(top_k * 3, 50)``) so post-filtering still has enough fused results to
    fill ``top_k`` after unsatisfying candidates are dropped.
    """
    return max(top_k * 3, 50)


def _intersect_all(candidate_sets: Sequence[CandidateSet]) -> Optional[CandidateSet]:
    """Intersect the given candidate sets, or return ``None`` when there are none."""
    if not candidate_sets:
        return None
    combined = candidate_sets[0]
    for candidate in candidate_sets[1:]:
        combined = combined.intersect(candidate)
    return combined


@dataclass(frozen=True)
class ConstraintExecutionPlan:
    """Execution plan assembled by ``ConstraintAwarePlanner.plan``.

    Attributes:
        candidate_set: Intersected candidate UIDs, or ``None`` when the query
            declares no constraints and therefore needs no filtering.
        mode: ``"pre_filter"`` when the candidate set is pushed down as the
            ``candidate_uids`` retrieval filter, ``"post_filter"`` when it is
            applied to fused results, or ``"none"`` for unconstrained queries.
        retrieval_top_k: Pool size requested from ``smart_search``.
        mode_reason: Human-readable justification for ``mode``, kept for
            diagnostics and the benchmark report.
        dropped_constraints: Constraint sources the fallback policy relaxed
            away; empty when every declared predicate was honored.
    """

    candidate_set: Optional[CandidateSet]
    mode: str
    retrieval_top_k: int
    mode_reason: str = ""
    dropped_constraints: Tuple[str, ...] = ()


class ConstraintAwarePlanner:
    """Resolve constraints and execute constrained retrieval.

    The planner owns the coordination responsibilities and nothing else: it
    never encodes, ranks, or fuses results. Semantic similarity remains a pure
    ranking signal applied by the existing retrieval pipeline, while constraints
    only ever restrict which UIDs are allowed through.
    """

    def __init__(
        self,
        semantic_graph: "SemanticGraph",
        prefilter_selectivity_threshold: float = DEFAULT_PREFILTER_SELECTIVITY_THRESHOLD,
        fallback_policy: Optional[FallbackPolicy] = None,
        max_pool_expansions: int = 2,
    ):
        """Initialize the planner.

        Args:
            semantic_graph: Graph layer providing the SemanticMap used for
                constraint resolution and the MultiRetriever used for ranking.
            prefilter_selectivity_threshold: Candidate fraction below which the
                plan pushes ``candidate_uids`` down into retrieval instead of
                post-filtering fused results.
            fallback_policy: Degenerate-candidate handling; defaults to keeping
                the degenerate set (so an unsatisfiable query returns nothing).
            max_pool_expansions: How many times post-filtering may double the
                retrieval pool while the backend is still saturated and the
                constrained results have not filled ``top_k``.
        """
        if max_pool_expansions < 0:
            raise ValueError("max_pool_expansions must be non-negative.")
        self.semantic_graph = semantic_graph
        self.prefilter_selectivity_threshold = float(prefilter_selectivity_threshold)
        self.fallback_policy = fallback_policy or FallbackPolicy()
        self.max_pool_expansions = int(max_pool_expansions)

    def _build_resolvers(
        self, constraints: QueryConstraints
    ) -> List[BaseConstraintResolver]:
        """Build one resolver per declared leaf constraint."""
        semantic_map = self.semantic_graph.semantic_map
        resolvers: List[BaseConstraintResolver] = []
        if constraints.metadata_filters:
            resolvers.append(
                MetadataConstraintResolver(semantic_map, constraints.metadata_filters)
            )
        if constraints.time_range is not None:
            resolvers.append(
                TimeRangeConstraintResolver(semantic_map, constraints.time_range)
            )
        if constraints.relation is not None:
            resolvers.append(
                RelationConstraintResolver(semantic_map, constraints.relation)
            )
        if constraints.space_names:
            resolvers.append(
                SpaceConstraintResolver(semantic_map, constraints.space_names)
            )
        return resolvers

    def resolve_candidates(self, constraints: QueryConstraints) -> Optional[CandidateSet]:
        """Resolve every leaf constraint and intersect the results.

        When the intersection is degenerate, ``fallback_policy`` decides whether
        to keep it (the default) or to relax the query by dropping predicates.

        Returns:
            The intersected candidate set, or ``None`` when the query declares
            no constraints.
        """
        candidate_set, _ = self._resolve_candidates(constraints)
        return candidate_set

    def _resolve_candidates(
        self, constraints: QueryConstraints
    ) -> Tuple[Optional[CandidateSet], Tuple[str, ...]]:
        """Resolve constraints and report which sources the fallback dropped."""
        resolvers = self._build_resolvers(constraints)
        if not resolvers:
            return None, ()

        resolved = [(resolver.source, resolver.resolve()) for resolver in resolvers]
        for source, candidate in resolved:
            logger.debug(
                "Constraint %s resolved %d candidates.", source, len(candidate)
            )

        combined = _intersect_all([candidate for _, candidate in resolved])
        return self._apply_fallback(combined, resolved)

    def _apply_fallback(
        self,
        combined: CandidateSet,
        resolved: Sequence[Tuple[str, CandidateSet]],
    ) -> Tuple[CandidateSet, Tuple[str, ...]]:
        """Relax a degenerate candidate set by dropping the most restrictive leaf.

        Predicates are dropped one at a time — always the one whose removal
        leaves the largest intersection — until the set is no longer degenerate.
        The last predicate is never dropped, so a genuinely unmatched single
        predicate still yields its (possibly empty) set rather than silently
        turning the query into an unconstrained one.

        Returns:
            The (possibly relaxed) candidate set and the dropped constraint
            sources, so plans and benchmarks can report the relaxation.
        """
        policy = self.fallback_policy
        if (
            len(combined) >= policy.min_candidates
            or not policy.relaxes_degenerate_queries
        ):
            return combined, ()

        remaining = list(resolved)
        relaxed = combined
        dropped: List[str] = []
        while len(relaxed) < policy.min_candidates and len(remaining) > 1:
            drop_index, candidate = self._most_relaxing_drop(remaining)
            dropped_source = remaining[drop_index][0]
            remaining.pop(drop_index)
            dropped.append(dropped_source)
            logger.warning(
                "Fallback policy dropped constraint '%s': %d candidate(s) is "
                "below min_candidates=%d; relaxed set holds %d candidate(s).",
                dropped_source,
                len(relaxed),
                policy.min_candidates,
                len(candidate),
            )
            relaxed = candidate
        return relaxed, tuple(dropped)

    @staticmethod
    def _most_relaxing_drop(
        remaining: Sequence[Tuple[str, CandidateSet]],
    ) -> Tuple[int, CandidateSet]:
        """Return the index whose removal yields the largest intersection."""
        best_index, best_set = 0, None
        for index in range(len(remaining)):
            others = [
                candidate
                for position, (_, candidate) in enumerate(remaining)
                if position != index
            ]
            candidate = _intersect_all(others)
            if best_set is None or len(candidate) > len(best_set):
                best_index, best_set = index, candidate
        return best_index, best_set

    def _retrieval_top_k(
        self, top_k: int, candidate_set: Optional[CandidateSet], mode: str
    ) -> int:
        """Return the retrieval pool size for the planned query.

        Post-filtering discards non-candidate units from the fused pool, so the
        pool must hold enough survivors: with selectivity ``s``, roughly ``s`` of
        the pool survives, so about ``top_k / s`` fused results are needed to fill
        ``top_k``. The pre-filter path already restricts retrieval to candidates
        and keeps the baseline expansion.
        """
        if candidate_set is None:
            return top_k
        pool = _expand_top_k(top_k)
        selectivity = candidate_set.selectivity if mode == "post_filter" else None
        if selectivity:
            pool = max(pool, math.ceil(top_k / selectivity))
        return pool

    def plan(self, constraints: QueryConstraints, top_k: int) -> ConstraintExecutionPlan:
        """Assemble the execution plan for one constrained query.

        Selectivity chooses the execution mode: candidate sets covering less
        than ``prefilter_selectivity_threshold`` of the corpus are pushed down
        as a retrieval filter, while broad sets (and sets of unknown size) keep
        the post-filter path — push-down only pays off when it removes enough
        candidates to shrink the retrieval work.
        """
        candidate_set, dropped = self._resolve_candidates(constraints)
        if candidate_set is None:
            return ConstraintExecutionPlan(
                candidate_set=None,
                mode="none",
                retrieval_top_k=top_k,
                mode_reason="no_constraints",
            )

        selectivity = candidate_set.selectivity
        if selectivity is None:
            mode, reason = "post_filter", "selectivity_unknown"
        elif selectivity < self.prefilter_selectivity_threshold:
            mode, reason = "pre_filter", "selective_below_threshold"
        else:
            mode, reason = "post_filter", "broad_above_threshold"

        return ConstraintExecutionPlan(
            candidate_set=candidate_set,
            mode=mode,
            retrieval_top_k=self._retrieval_top_k(top_k, candidate_set, mode),
            mode_reason=reason,
            dropped_constraints=dropped,
        )

    def execute_query(
        self,
        query_text: str,
        constraints: QueryConstraints,
        top_k: int = 10,
        **kwargs: Any,
    ) -> List[Tuple["MemoryUnit", float]]:
        """Run constrained retrieval end to end.

        Args:
            query_text: Query text passed to ``smart_search``.
            constraints: Declarative constraint bundle (AND semantics).
            top_k: Final number of results.
            **kwargs: Additional ``MultiRetriever.smart_search`` options
                (``methods``, ``fusion_method``, ``rerank_method``, ...). A
                caller-provided ``candidate_uids`` is intersected with the
                resolved constraint candidates before any push-down.

        Returns:
            Fused results satisfying every declared constraint, ordered by the
            existing ranking pipeline and truncated to ``top_k``.
        """
        if kwargs.pop("return_detailed", False):
            raise ValueError(
                "search_constrained does not support return_detailed=True; use "
                "MultiRetriever.smart_search for detailed execution diagnostics."
            )

        plan = self.plan(constraints, top_k)
        if plan.candidate_set is not None and len(plan.candidate_set) == 0:
            logger.info(
                "Constraint resolution produced an empty candidate set; "
                "skipping retrieval and returning no results."
            )
            return []

        caller_candidates = kwargs.pop("candidate_uids", None)
        search_kwargs: Dict[str, Any] = dict(kwargs)
        effective_set = plan.candidate_set
        if caller_candidates is not None:
            caller_set = CandidateSet.from_iterable(
                caller_candidates, source="caller_candidates"
            )
            effective_set = (
                caller_set
                if effective_set is None
                else effective_set.intersect(caller_set)
            )
            if len(effective_set) == 0:
                logger.info(
                    "Caller candidate filter and constraints share no UIDs; "
                    "skipping retrieval and returning no results."
                )
                return []
            search_kwargs["candidate_uids"] = list(effective_set)
        elif plan.mode == "pre_filter":
            search_kwargs["candidate_uids"] = list(effective_set)

        multi_retriever = self.semantic_graph.get_multi_retriever()
        requested = plan.retrieval_top_k
        latest = multi_retriever.smart_search(
            query_text, top_k=requested, **search_kwargs
        )
        results = latest

        if effective_set is None:
            return results[:top_k]

        # Final verification: backends and graph expansion may return units the
        # push-down could not exclude, so constraints are re-checked even when
        # the query was only narrowed by caller-supplied candidates.
        filtered = self._filter_results(results, effective_set)
        expansions = 0
        while (
            len(filtered) < top_k
            and len(latest) >= requested
            and expansions < self.max_pool_expansions
        ):
            requested *= 2
            expansions += 1
            logger.debug(
                "Constrained retrieval filled %d/%d results; expanding the "
                "retrieval pool to %d.",
                len(filtered),
                top_k,
                requested,
            )
            latest = multi_retriever.smart_search(
                query_text, top_k=requested, **search_kwargs
            )
            results = self._merge_results(results, latest)
            filtered = self._filter_results(results, effective_set)

        logger.debug(
            "Constrained retrieval kept %d of %d fused results (mode=%s).",
            len(filtered),
            len(results),
            plan.mode,
        )
        return filtered[:top_k]

    @staticmethod
    def _filter_results(
        results: Sequence[Tuple["MemoryUnit", float]],
        effective_set: CandidateSet,
    ) -> List[Tuple["MemoryUnit", float]]:
        """Keep the fused results whose UID belongs to the effective set."""
        return [(unit, score) for unit, score in results if unit.uid in effective_set]

    @staticmethod
    def _merge_results(
        existing: Sequence[Tuple["MemoryUnit", float]],
        incoming: Sequence[Tuple["MemoryUnit", float]],
    ) -> List[Tuple["MemoryUnit", float]]:
        """Merge two result pools by UID, keeping the latest score per unit."""
        by_uid = {unit.uid: (unit, score) for unit, score in existing}
        for unit, score in incoming:
            by_uid[unit.uid] = (unit, score)
        return sorted(by_uid.values(), key=lambda item: item[1], reverse=True)