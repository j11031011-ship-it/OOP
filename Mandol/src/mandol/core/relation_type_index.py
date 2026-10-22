"""Relation-type inverted index over explicit graph edges (P2+).

The rustworkx graph stays the single source of truth for topology; this
collaborator is a derived structure that answers "which edges carry relation
type T?" without enumerating every edge of the graph. ``SemanticGraph``
maintains it incrementally on relationship add/delete and rebuilds it after
loading a persisted graph, mirroring the "index mirrors the authoritative
population" pattern of ``mandol.indexes.MetadataIndex``.

It is deliberately *not* used by the constraint resolver: the depth-limited BFS
already walks only per-node adjacency, so an inverted index would not lower its
cost. The consumer is the seed-less, relation-type-filtered edge lookup
(``SemanticGraph.get_edges_by_relation_types`` /
``SemanticGraph.search_graph_relations``), which previously scanned every edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

DEFAULT_RELATION_TYPE = "RELATED_TO"


@dataclass(frozen=True)
class IndexedEdge:
    """One indexed directed edge: public UIDs plus its attribute payload.

    ``source_uid``/``target_uid`` are normalized public UIDs (memory spaces
    carry their ``ms:`` prefix), matching ``SemanticGraph._index_to_uid``.
    """

    source_uid: str
    target_uid: str
    relation_type: str
    data: Mapping[str, Any]


class RelationTypeIndex:
    """``relation_type -> edges`` inverted index with insertion-order buckets.

    Insertion order is preserved so type-filtered lookups are deterministic
    (matching the ``limit`` early-out of ``search_graph_relations``). Removal
    drops one ``(source, target, type)`` occurrence at a time, which is exactly
    the granularity ``delete_relationship`` operates at.
    """

    def __init__(self) -> None:
        self._edges_by_type: Dict[str, List[IndexedEdge]] = {}
        self._all_edges: List[IndexedEdge] = []

    def add(
        self,
        source_uid: str,
        target_uid: str,
        relation_type: Optional[str],
        data: Optional[Mapping[str, Any]] = None,
    ) -> IndexedEdge:
        """Index one directed edge and return the stored entry.

        A missing ``relation_type`` falls back to ``RELATED_TO``, matching the
        default used across the graph's relationship APIs.
        """
        entry = IndexedEdge(
            source_uid=str(source_uid),
            target_uid=str(target_uid),
            relation_type=str(relation_type or DEFAULT_RELATION_TYPE),
            data=dict(data or {}),
        )
        self._edges_by_type.setdefault(entry.relation_type, []).append(entry)
        self._all_edges.append(entry)
        return entry

    def remove(
        self,
        source_uid: str,
        target_uid: str,
        relation_type: Optional[str],
    ) -> bool:
        """Drop one matching edge; return whether an entry was removed."""
        rel_type = str(relation_type or DEFAULT_RELATION_TYPE)
        bucket = self._edges_by_type.get(rel_type)
        if not bucket:
            return False
        for position, entry in enumerate(bucket):
            if entry.source_uid == source_uid and entry.target_uid == target_uid:
                del bucket[position]
                if not bucket:
                    del self._edges_by_type[rel_type]
                self._drop_from_all(entry)
                return True
        return False

    def _drop_from_all(self, entry: IndexedEdge) -> None:
        for position, candidate in enumerate(self._all_edges):
            if candidate is entry:
                del self._all_edges[position]
                return

    def lookup(
        self, relation_types: Optional[Iterable[str]] = None
    ) -> List[IndexedEdge]:
        """Return indexed edges, optionally restricted to the given types.

        ``None`` or an empty iterable means "every relation type" and returns
        all edges in insertion order. Requested types are de-duplicated while
        preserving the caller's order.
        """
        if not relation_types:
            return list(self._all_edges)
        results: List[IndexedEdge] = []
        for rel_type in dict.fromkeys(str(t) for t in relation_types):
            results.extend(self._edges_by_type.get(rel_type, ()))
        return results

    def clear(self) -> None:
        """Drop every entry."""
        self._edges_by_type.clear()
        self._all_edges.clear()

    def rebuild(self, rx_graph: Any, index_to_uid: Mapping[int, str]) -> None:
        """Repopulate the index from the authoritative rustworkx graph.

        Called after a persisted graph is restored so the derived index cannot
        drift from the topology it mirrors.
        """
        self.clear()
        for edge_idx in rx_graph.edge_indices():
            source_idx, target_idx = rx_graph.get_edge_endpoints_by_index(edge_idx)
            source_uid = index_to_uid.get(source_idx)
            target_uid = index_to_uid.get(target_idx)
            if not source_uid or not target_uid:
                continue
            data = rx_graph.get_edge_data_by_index(edge_idx) or {}
            self.add(source_uid, target_uid, data.get("type"), data)

    @property
    def relation_types(self) -> Tuple[str, ...]:
        """Relation types currently present, in first-seen order."""
        return tuple(self._edges_by_type)

    def edge_count(
        self, relation_types: Optional[Iterable[str]] = None
    ) -> int:
        """Number of indexed edges, optionally restricted to given types."""
        return len(self.lookup(relation_types))

    def __len__(self) -> int:
        return len(self._all_edges)
