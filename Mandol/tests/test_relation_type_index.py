"""Offline tests for the relation-type inverted index (P2+).

``RelationTypeIndex`` is exercised against a raw ``rustworkx.PyDiGraph`` so these
tests need neither an embedding model nor a ``SemanticMap``; the graph-level
integration (maintenance on add/delete/remove and the indexed lookup used by
``search_graph_relations``) lives in ``test_constrained_retrieval.py`` where the
graph fixture already exists.
"""

from __future__ import annotations

import rustworkx as rx

from mandol.core.relation_type_index import DEFAULT_RELATION_TYPE, RelationTypeIndex


def _graph_with_edges() -> rx.PyDiGraph:
    graph = rx.PyDiGraph(multigraph=True)
    first = graph.add_node({"uid": "u1"})
    second = graph.add_node({"uid": "u2"})
    third = graph.add_node({"uid": "u3"})
    graph.add_edge(first, second, {"type": "FOLLOWED_BY"})
    graph.add_edge(first, third, {"type": "MENTION_OF"})
    graph.add_edge(third, first, {"type": "MENTION_OF"})
    graph.add_edge(second, third, {"type": "NEXT"})
    return graph


def test_add_and_lookup_by_type() -> None:
    index = RelationTypeIndex()
    index.add("u1", "u2", "FOLLOWED_BY", {"type": "FOLLOWED_BY", "note": "a"})
    index.add("u1", "u3", "MENTION_OF", {"type": "MENTION_OF"})
    index.add("u4", "u1", "MENTION_OF", {"type": "MENTION_OF"})

    followed = index.lookup(["FOLLOWED_BY"])
    assert [(e.source_uid, e.target_uid) for e in followed] == [("u1", "u2")]
    assert dict(followed[0].data) == {"type": "FOLLOWED_BY", "note": "a"}

    mentions = index.lookup(["MENTION_OF"])
    assert {(e.source_uid, e.target_uid) for e in mentions} == {("u1", "u3"), ("u4", "u1")}

    assert index.edge_count() == 3
    assert index.edge_count(["MENTION_OF"]) == 2
    assert set(index.relation_types) == {"FOLLOWED_BY", "MENTION_OF"}
    assert len(index) == 3


def test_lookup_without_filter_returns_all_in_insertion_order() -> None:
    index = RelationTypeIndex()
    index.add("u1", "u2", "A")
    index.add("u2", "u3", "B")
    index.add("u3", "u1", "A")

    assert [(e.source_uid, e.target_uid, e.relation_type) for e in index.lookup()] == [
        ("u1", "u2", "A"),
        ("u2", "u3", "B"),
        ("u3", "u1", "A"),
    ]
    assert len(index.lookup([])) == 3


def test_lookup_deduplicates_requested_types() -> None:
    index = RelationTypeIndex()
    index.add("u1", "u2", "A")
    index.add("u2", "u3", "B")

    assert index.edge_count(["A", "A", "B"]) == 2
    assert [e.relation_type for e in index.lookup(["A", "A"])] == ["A"]


def test_remove_drops_one_occurrence_at_a_time() -> None:
    index = RelationTypeIndex()
    index.add("u1", "u2", "MENTION_OF", {"type": "MENTION_OF", "seq": 1})
    index.add("u1", "u2", "MENTION_OF", {"type": "MENTION_OF", "seq": 2})

    assert index.remove("u1", "u2", "MENTION_OF") is True
    remaining = index.lookup(["MENTION_OF"])
    assert len(remaining) == 1
    assert remaining[0].data["seq"] == 2

    assert index.remove("u1", "u2", "MENTION_OF") is True
    # The emptied bucket is dropped entirely.
    assert "MENTION_OF" not in index.relation_types
    assert len(index) == 0


def test_remove_unknown_edges_and_types_returns_false() -> None:
    index = RelationTypeIndex()
    index.add("u1", "u2", "A")

    assert index.remove("u1", "u2", "B") is False
    assert index.remove("u2", "u1", "A") is False
    assert index.edge_count(["A"]) == 1


def test_missing_relation_type_falls_back_to_default() -> None:
    index = RelationTypeIndex()
    index.add("u1", "u2", None)
    index.add("u3", "u4", "")

    assert index.edge_count([DEFAULT_RELATION_TYPE]) == 2
    assert index.edge_count() == 2


def test_rebuild_replaces_previous_state_from_graph() -> None:
    index = RelationTypeIndex()
    index.add("stale", "entry", "OLD")

    graph = _graph_with_edges()
    index_to_uid = {
        node: data["uid"] for node, data in zip(range(graph.num_nodes()), graph.nodes())
    }
    index.rebuild(graph, index_to_uid)

    assert len(index) == 4
    assert index.edge_count(["FOLLOWED_BY"]) == 1
    assert index.edge_count(["MENTION_OF"]) == 2
    assert index.edge_count(["NEXT"]) == 1
    assert "OLD" not in index.relation_types


def test_rebuild_skips_edges_with_unmapped_endpoints() -> None:
    graph = rx.PyDiGraph(multigraph=True)
    first = graph.add_node({})
    second = graph.add_node({})
    graph.add_edge(first, second, {"type": "FOLLOWED_BY"})

    index = RelationTypeIndex()
    # Only the second endpoint is mapped; the edge cannot be exposed by UID.
    index.rebuild(graph, {second: "u2"})

    assert len(index) == 0
