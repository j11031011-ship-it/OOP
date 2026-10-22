"""Constraint resolvers: leaf predicates evaluated into candidate sets.

Every resolver answers the same question — "which UIDs satisfy this leaf
constraint?" — so the planner can combine them with set algebra without knowing
how each predicate is evaluated. Metadata and time predicates consult the
per-field indexes registered on ``SemanticMap`` (P1) and fall back to the
linear-scan implementation whenever a field or operator is not servable; the
graph BFS resolver (P2) walks ``SemanticGraph.rx_graph`` directly. All of them
share the same ``BaseConstraintResolver.resolve`` contract, so the planner and
the retrieval backends stay unaware of how a predicate is evaluated.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Set, Tuple

from .candidate_set import CandidateSet
from .query_constraints import RelationConstraint, TimeRange

if TYPE_CHECKING:
    from ..core.memory_unit import MemoryUnit
    from ..core.semantic_map import SemanticMap

_ORDERING_OPERATORS = ("gt", "gte", "lt", "lte")
_SUPPORTED_OPERATORS = frozenset(
    {"eq", "ne", "in", "nin", "gt", "gte", "lt", "lte", "contain", "not_contain"}
)


def _resolve_field_value(unit: "MemoryUnit", field: str) -> Any:
    """Resolve a field with the same lookup order as ``filter_memory_units``."""
    value = getattr(unit, field, None)
    if value is None:
        value = unit.raw_data.get(field)
    if value is None and isinstance(unit.metadata, dict):
        value = unit.metadata.get(field)
    return value


def _compare_ordering(value: Any, operator: str, expected: Any) -> bool:
    """Compare ordering operators, treating missing values as non-matches.

    ``SemanticMap.filter_memory_units`` raises ``TypeError`` when a unit lacks
    the filtered field (``None > expected``); resolvers must not abort a whole
    scan on one payload, so an absent value simply fails the predicate.
    """
    if value is None:
        return False
    try:
        if operator == "gt":
            return value > expected
        if operator == "gte":
            return value >= expected
        if operator == "lt":
            return value < expected
        return value <= expected
    except TypeError:
        return False


def _unit_matches(unit: "MemoryUnit", filter_condition: Dict[str, Dict[str, Any]]) -> bool:
    """Evaluate a field condition using ``filter_memory_units`` semantics."""
    for field, conditions in filter_condition.items():
        value = _resolve_field_value(unit, field)
        for operator, expected in conditions.items():
            if operator not in _SUPPORTED_OPERATORS:
                raise ValueError(
                    f"Unsupported filter operator {operator!r} on field {field!r}; "
                    f"expected one of {sorted(_SUPPORTED_OPERATORS)}."
                )
            if operator == "eq" and value != expected:
                return False
            if operator == "ne" and value == expected:
                return False
            if operator == "in" and value not in expected:
                return False
            if operator == "nin" and value in expected:
                return False
            if operator in _ORDERING_OPERATORS and not _compare_ordering(
                value, operator, expected
            ):
                return False
            if operator == "contain" and expected not in str(value):
                return False
            if operator == "not_contain" and expected in str(value):
                return False
    return True


class BaseConstraintResolver(ABC):
    """Resolve one leaf constraint into a ``CandidateSet``.

    Subclasses receive the retrieval source and their leaf predicate at
    construction time and stay stateless afterwards; the planner builds one
    resolver per declared leaf constraint and intersects the results.
    """

    def __init__(self, semantic_map: "SemanticMap"):
        """Bind the resolver to the map whose payloads back the predicate.

        Args:
            semantic_map: Map owning the L1 unit population scanned by P0
                resolvers and, from P1 on, the metadata index.
        """
        self.semantic_map = semantic_map

    @abstractmethod
    def resolve(self) -> CandidateSet:
        """Return the UID candidate set satisfying this leaf constraint."""

    def _l1_units(self) -> List["MemoryUnit"]:
        """Return L1-resident payloads.

        Candidate resolution observes the same population as
        ``filter_memory_units`` (L1 payloads only); the metadata index mirrors
        that population rather than paging cold records in, so scans and index
        lookups stay interchangeable and neither materializes the corpus.
        """
        return list(self.semantic_map.memory_units.values())

    def _total_units(self) -> int:
        """Return the known-unit count used for selectivity estimation.

        Cold payloads remain known (and indexed) after tiered paging, so using
        the resident count here would overestimate selectivity and make the
        planner push a broad constraint down as if it were selective.
        """
        return self.semantic_map.total_unit_count


class MetadataConstraintResolver(BaseConstraintResolver):
    """Metadata field-condition resolver: index-backed, scan otherwise.

    ``resolve`` first asks the map's ``MetadataIndex``; the index returns
    ``None`` unless every field is registered and every operator is servable,
    in which case the linear scan below runs with exactly the P0 semantics.
    """

    source = "metadata_filters"

    def __init__(
        self,
        semantic_map: "SemanticMap",
        filters: Dict[str, Dict[str, Any]],
    ):
        """Initialize the resolver.

        Args:
            semantic_map: Map providing the unit population and the metadata index.
            filters: Field conditions using the ``filter_memory_units``
                operator vocabulary.
        """
        super().__init__(semantic_map)
        self.filters = filters
        self._validate_operators()

    def _validate_operators(self) -> None:
        """Reject unknown operators instead of silently ignoring the condition."""
        for field, conditions in self.filters.items():
            if not isinstance(conditions, dict):
                raise ValueError(
                    f"Filter for field {field!r} must be an operator mapping."
                )
            unknown = set(conditions) - _SUPPORTED_OPERATORS
            if unknown:
                raise ValueError(
                    f"Unsupported filter operator(s) {sorted(unknown)} on field "
                    f"{field!r}; expected one of {sorted(_SUPPORTED_OPERATORS)}."
                )

    def resolve(self) -> CandidateSet:
        """Resolve the field conditions, preferring the registered index."""
        indexed = self._resolve_via_index()
        if indexed is not None:
            return indexed
        return self._resolve_via_scan()

    def _resolve_via_index(self) -> Optional[CandidateSet]:
        """Answer the filters from the map's metadata index when servable."""
        metadata_index = getattr(self.semantic_map, "_metadata_index", None)
        if metadata_index is None:
            return None
        resolved = metadata_index.lookup(self.filters)
        if resolved is None:
            return None
        return CandidateSet.from_iterable(
            resolved.uids, source=self.source, total=self._total_units()
        )

    def _resolve_via_scan(self) -> CandidateSet:
        """Scan L1 payloads and return the UIDs matching every field condition."""
        total = self._total_units()
        uids: Set[str] = {
            unit.uid
            for unit in self._l1_units()
            if _unit_matches(unit, self.filters)
        }
        return CandidateSet.from_iterable(uids, source=self.source, total=total)


class TimeRangeConstraintResolver(MetadataConstraintResolver):
    """Time-window resolver with a sorted-epoch fast path (P1).

    When the field is registered as a sorted index, the lookup is a bisect over
    normalized epoch floats. Otherwise the scan below normalizes every payload
    timestamp through ``TimestampNormalizer`` as well, so ISO strings, epoch
    numbers, and LoCoMo-style strings compare correctly even when the sorted
    index was not registered. Payloads whose timestamp cannot be parsed never
    match an ordering predicate.
    """

    source = "time_range"

    def __init__(self, semantic_map: "SemanticMap", time_range: TimeRange):
        """Initialize the resolver.

        Args:
            semantic_map: Map providing the unit population.
            time_range: Inclusive window applied to ``time_range.field``.
        """
        conditions: Dict[str, Any] = {}
        if time_range.start is not None:
            conditions["gte"] = time_range.start
        if time_range.end is not None:
            conditions["lte"] = time_range.end
        super().__init__(semantic_map, {time_range.field: conditions})
        self.time_range = time_range

    def _resolve_via_scan(self) -> CandidateSet:
        """Scan L1 payloads with normalized timestamp comparison."""
        from ..indexes.timestamp_normalizer import TimestampNormalizer

        normalizer = TimestampNormalizer()
        start = self._normalize_bound(normalizer, self.time_range.start, "start")
        end = self._normalize_bound(normalizer, self.time_range.end, "end")
        total = self._total_units()
        uids: Set[str] = set()
        for unit in self._l1_units():
            epoch = normalizer.normalize(
                _resolve_field_value(unit, self.time_range.field)
            )
            if epoch is None:
                continue
            if start is not None and epoch < start:
                continue
            if end is not None and epoch > end:
                continue
            uids.add(unit.uid)
        return CandidateSet.from_iterable(uids, source=self.source, total=total)

    def _normalize_bound(
        self,
        normalizer: Any,
        bound: Any,
        name: str,
    ) -> Optional[float]:
        """Normalize one query bound, failing loudly when it is unparsable."""
        if bound is None:
            return None
        epoch = normalizer.normalize(bound)
        if epoch is None:
            raise ValueError(
                f"TimeRange.{name} is not a recognized timestamp: {bound!r}."
            )
        return epoch


class SpaceConstraintResolver(BaseConstraintResolver):
    """Adapter exposing MemorySpace membership as a candidate set.

    The resolver delegates to the map's space-filter resolution — the same
    membership logic the dense/BM25/SPLADE backends use for ``space_names`` — so
    the space predicate stays isomorphic to the other constraints without
    materializing payloads.
    """

    source = "space_names"

    def __init__(self, semantic_map: "SemanticMap", space_names: List[str]):
        """Initialize the resolver.

        Args:
            semantic_map: Map owning the MemorySpace hierarchy.
            space_names: Space names with or without the ``ms:`` prefix.
        """
        super().__init__(semantic_map)
        self.space_names = [str(name) for name in space_names]

    def resolve(self) -> CandidateSet:
        """Return the UIDs belonging to any requested MemorySpace."""
        space_uids: Optional[Set[str]] = self.semantic_map._get_candidate_uids_set(
            self.space_names, None
        )
        return CandidateSet.from_iterable(
            space_uids or (),
            source=self.source,
            total=self._total_units(),
        )


class RelationConstraintResolver(BaseConstraintResolver):
    """Resolve a graph-neighborhood constraint with a depth-limited BFS (P2).

    The resolver walks ``SemanticGraph.rx_graph`` directly instead of the legacy
    graph helpers ``search_graph_relations(seed_nodes=...)`` /
    ``get_node_neighbors``: those call ``GraphRetriever`` methods that do not
    exist and raise ``AttributeError`` at runtime, and they return edge lists
    rather than the node set a predicate needs. Starting from the seed UIDs it
    follows edges up to ``max_depth`` hops in the requested direction, only
    across the declared ``relation_types`` when given, and returns the reached
    memory-unit UIDs — the neighborhood predicate becomes a candidate set
    isomorphic to the metadata/time/space predicates.

    The seeds are excluded from the result: the constraint describes the
    neighborhood of the seeds ("neighbors of an entity"), not the seeds. The
    traversal may pass through memory-space nodes (``ms:`` prefixed) to reach
    further units, but those nodes are never returned because only memory units
    are retrievable.
    """

    source = "relation"

    def __init__(self, semantic_map: "SemanticMap", relation: RelationConstraint):
        """Initialize the resolver.

        Args:
            semantic_map: Map whose parent graph owns the adjacency and the
                UID/node-index mapping used by the traversal.
            relation: Neighborhood predicate (seeds, depth, relation types,
                direction).
        """
        super().__init__(semantic_map)
        self.relation = relation

    def resolve(self) -> CandidateSet:
        """Return the memory-unit UIDs reachable from the seed UIDs."""
        graph = getattr(self.semantic_map, "_parent_semantic_graph", None)
        total = self._total_units()
        if graph is None:
            return CandidateSet.from_iterable((), source=self.source, total=total)

        uid_to_index = graph._uid_to_index
        index_to_uid = graph._index_to_uid
        relation_types = (
            set(self.relation.relation_types) if self.relation.relation_types else None
        )
        # Seeds start visited so they never enter the result and a seed reached
        # as another seed's neighbor is not reported either.
        visited: Set[str] = {
            uid for uid in self.relation.seed_uids if uid in uid_to_index
        }
        frontier = set(visited)
        neighbors: Set[str] = set()
        for _ in range(max(int(self.relation.max_depth), 0)):
            next_frontier: Set[str] = set()
            for uid in frontier:
                for neighbor_idx in self._iter_neighbor_indices(
                    graph.rx_graph,
                    uid_to_index[uid],
                    self.relation.direction,
                    relation_types,
                ):
                    neighbor_uid = index_to_uid.get(neighbor_idx)
                    if not neighbor_uid or neighbor_uid in visited:
                        continue
                    visited.add(neighbor_uid)
                    next_frontier.add(neighbor_uid)
                    if not neighbor_uid.startswith("ms:"):
                        neighbors.add(neighbor_uid)
            frontier = next_frontier
            if not frontier:
                break
        return CandidateSet.from_iterable(neighbors, source=self.source, total=total)

    @staticmethod
    def _iter_neighbor_indices(
        rx_graph: Any,
        node_idx: int,
        direction: str,
        relation_types: Optional[Set[str]],
    ) -> Iterator[int]:
        """Yield neighbor node indices honoring direction and relation types.

        ``out_edges`` yields ``(parent, child, data)`` and ``in_edges`` yields
        the same tuple shape, so the neighbor is the endpoint that is not
        ``node_idx``. Untyped edges are treated as relations only when no type
        filter is requested.
        """
        edges: List[Tuple[int, int, Any]] = []
        if direction in ("successors", "both"):
            edges.extend(rx_graph.out_edges(node_idx))
        if direction in ("predecessors", "both"):
            edges.extend(rx_graph.in_edges(node_idx))
        for src_idx, tgt_idx, edge_data in edges:
            if relation_types is not None and (
                not edge_data or edge_data.get("type") not in relation_types
            ):
                continue
            yield tgt_idx if src_idx == node_idx else src_idx