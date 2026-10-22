"""Offline tests for constraint-aware hybrid retrieval (P0/P1/P2).

The suite follows the offline paradigm of ``test_rocksdb_tiered_cache.py``:
``global_model_manager`` is patched with deterministic dummies and every test
runs without network access or real model downloads. P1 additions cover the
metadata-index fast paths, incremental index maintenance, and the
selectivity-driven ``candidate_uids`` pre-filter push-down; P2 additions cover
the graph-neighborhood BFS resolver over ``SemanticGraph.rx_graph``; P2+
additions cover maintenance of the relation-type inverted index; P3 additions
cover the degenerate-candidate fallback policy, selectivity-aware post-filter
pool sizing, and the recorded mode rationale; the last two tests pin the
``SemanticGraph`` facade fixes for the ``RetrievalMethod`` runtime ``NameError``
and the misnamed node-search dispatch targets.
"""

from __future__ import annotations

import numpy as np
import pytest

from mandol.constraints import (
    DROP_CONSTRAINT,
    CandidateSet,
    ConstraintAwarePlanner,
    FallbackPolicy,
    MetadataConstraintResolver,
    QueryConstraints,
    RelationConstraint,
    RelationConstraintResolver,
    SpaceConstraintResolver,
    TimeRange,
    TimeRangeConstraintResolver,
)
from mandol.core.memory_unit import MemoryUnit
from mandol.core.semantic_graph import SemanticGraph
from mandol.core.semantic_map import SemanticMap
from mandol.retrieval.retrieval_interface import RetrievalMethod
from mandol.constraints import planner as constraint_planner_module
from mandol.utils.model_manager import global_model_manager

# uid, text, timestamp, speaker, space
DEFAULT_UNITS = [
    ("u1", "morning coffee with alice", "2024-01-05T08:00:00", "speaker_a", "s-daily"),
    ("u2", "project deadline discussion", "2024-02-10T14:30:00", "speaker_b", "s-work"),
    ("u3", "coffee beans shopping trip", "2024-03-20T19:00:00", "speaker_a", "s-daily"),
    ("u4", "weekend hiking adventure", "2024-04-02T07:15:00", "speaker_c", "s-travel"),
    ("u5", "coffee machine repair notes", "2024-05-11T21:45:00", "speaker_b", "s-work"),
    ("u6", "evening reading list update", "2024-06-18T22:30:00", "speaker_a", "s-daily"),
]


class _DummyEmbeddingModel:
    """Deterministic 2-dim encoder so cosine ranking varies with the text."""

    def encode(self, texts, **kwargs):
        del kwargs
        if isinstance(texts, str):
            return self._vector(texts)
        return np.array([self._vector(text) for text in texts], dtype=np.float32)

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        seed = sum(ord(ch) for ch in str(text))
        return np.array([1.0 + seed % 5, 1.0 + (seed // 5) % 7], dtype=np.float32)


class _DummySpladeModel:
    def encode_query(self, text):
        del text
        return {7: 1.0}


class _StubMultiRetriever:
    """Records smart_search calls and replays a fixed result list."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def smart_search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return list(self.results)


class _SaturatedMultiRetriever:
    """Serves a full pool of non-candidate units, then reveals candidates."""

    def __init__(self, semantic_map):
        self.semantic_map = semantic_map
        self.calls = []
        self._outsiders = [
            MemoryUnit(f"x{index}", {"text_content": f"outsider {index}"})
            for index in range(200)
        ]

    def smart_search(self, query, **kwargs):
        del query
        limit = kwargs["top_k"]
        self.calls.append(limit)
        if limit <= 50:
            return [(unit, 2.0) for unit in self._outsiders[:limit]]
        valid = [
            (self.semantic_map.get_unit("u1"), 1.0),
            (self.semantic_map.get_unit("u3"), 0.9),
        ]
        return valid + [(unit, 2.0) for unit in self._outsiders[: limit - len(valid)]]


@pytest.fixture
def graph_factory(monkeypatch):
    original_loader = global_model_manager.get_or_load_model

    def load_model(**kwargs):
        if kwargs.get("model_type") == "text_embedding":
            return _DummyEmbeddingModel()
        return original_loader(**kwargs)

    monkeypatch.setattr(global_model_manager, "get_or_load_model", load_model)
    monkeypatch.setattr(
        global_model_manager,
        "get_splade_model",
        lambda *args, **kwargs: _DummySpladeModel(),
    )
    monkeypatch.setattr(
        SemanticMap,
        "_incremental_aux_retriever_add",
        lambda self, units: None,
    )

    def create() -> SemanticGraph:
        semantic_map = SemanticMap(
            embedding_model_name="test/dummy",
            embedding_dim=2,
            use_flash_attention=False,
        )
        return SemanticGraph(semantic_map)

    return create


def _spy_on_planner_warnings(monkeypatch) -> list:
    """Capture planner warnings as plain strings.

    The project logger tree stops propagation before the root logger, so
    ``caplog`` never sees these records; spying on ``logger.warning`` keeps the
    assertion independent of the logging configuration.
    """
    messages: list = []
    monkeypatch.setattr(
        constraint_planner_module.logger,
        "warning",
        lambda message, *args: messages.append(message % args if args else message),
    )
    return messages


def _build_graph(graph_factory, units) -> SemanticGraph:
    graph = graph_factory()
    for uid, text, timestamp, speaker, space in units:
        raw_data = {"text_content": text, "speaker": speaker}
        if timestamp is not None:
            raw_data["timestamp"] = timestamp
        graph.semantic_map.add_unit(
            MemoryUnit(uid, raw_data),
            explicit_content_for_embedding=text,
            space_names=[space],
            generate_sparse_embedding=False,
        )
    return graph


def test_candidate_set_algebra() -> None:
    first = CandidateSet.from_iterable({"u1", "u2", "u3"}, source="meta", total=10)
    second = CandidateSet.from_iterable({"u2", "u3", "u4"}, source="time", total=10)

    intersection = first.intersect(second)
    assert intersection.uids == {"u2", "u3"}
    assert intersection.source == "meta & time"
    assert intersection.selectivity == pytest.approx(0.2)

    union = first.union(second)
    assert union.uids == {"u1", "u2", "u3", "u4"}
    assert union.selectivity == pytest.approx(0.4)

    assert first.difference(second).uids == {"u1"}
    assert len(first) == 3
    assert "u1" in first
    assert set(first) == {"u1", "u2", "u3"}
    assert CandidateSet.from_iterable([]).selectivity is None


def test_metadata_resolver_matches_filter_memory_units_semantics(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    semantic_map = graph.semantic_map

    contains = {"speaker": {"in": ["speaker_a", "speaker_b"]}, "text_content": {"contain": "coffee"}}
    contains_uids = {
        unit.uid for unit in semantic_map.filter_memory_units(filter_condition=contains)
    }
    resolved_contains = MetadataConstraintResolver(semantic_map, contains).resolve()
    assert resolved_contains.uids == contains_uids == {"u1", "u3", "u5"}
    assert resolved_contains.total == len(DEFAULT_UNITS)

    excludes = {"speaker": {"in": ["speaker_a", "speaker_b"]}, "text_content": {"not_contain": "coffee"}}
    excludes_uids = {
        unit.uid for unit in semantic_map.filter_memory_units(filter_condition=excludes)
    }
    resolved_excludes = MetadataConstraintResolver(semantic_map, excludes).resolve()
    assert resolved_excludes.uids == excludes_uids == {"u2", "u6"}


def test_filterable_field_registry_survives_save_and_load(
    graph_factory, tmp_path
) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    semantic_map = graph.semantic_map
    semantic_map.register_filterable_fields({"speaker": "hash", "timestamp": "sorted"})

    checkpoint = tmp_path / "map"
    semantic_map.save_map(str(checkpoint), build_sparse_vectors=False)
    restored = SemanticMap.load_map(str(checkpoint))

    assert restored._metadata_index.field_kinds == {
        "speaker": "hash",
        "timestamp": "sorted",
    }
    assert MetadataConstraintResolver(
        restored, {"speaker": {"eq": "speaker_a"}}
    ).resolve().uids == {"u1", "u3", "u6"}
    assert TimeRangeConstraintResolver(
        restored, TimeRange(start="2024-05-01T00:00:00")
    ).resolve().uids == {"u5", "u6"}


def test_time_range_resolver_filters_raw_timestamps(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    semantic_map = graph.semantic_map

    after = TimeRangeConstraintResolver(
        semantic_map, TimeRange(start="2024-03-01T00:00:00")
    ).resolve()
    assert after.uids == {"u3", "u4", "u5", "u6"}

    window = TimeRangeConstraintResolver(
        semantic_map,
        TimeRange(start="2024-02-01T00:00:00", end="2024-04-30T23:59:59"),
    ).resolve()
    assert window.uids == {"u2", "u3", "u4"}


def test_time_range_scan_normalizes_mixed_timestamp_formats(graph_factory) -> None:
    """Without a sorted index the scan must still compare normalized epochs.

    ``n1`` and ``n2`` are the same instant in ISO and epoch form; ``n3`` is a
    LoCoMo-style string outside the window; ``n4`` is unparsable and dropped.
    """
    units = [
        ("n1", "iso timestamp", "2024-01-05T08:00:00", "speaker_a", "s-daily"),
        ("n2", "epoch timestamp", 1704441600.0, "speaker_a", "s-daily"),
        ("n3", "locomo timestamp", "1:56 pm on 8 May, 2023", "speaker_a", "s-daily"),
        ("n4", "garbage timestamp", "not a date", "speaker_a", "s-daily"),
    ]
    graph = _build_graph(graph_factory, units)

    resolved = TimeRangeConstraintResolver(
        graph.semantic_map,
        TimeRange(start="2024-01-01T00:00:00", end="2024-01-31T23:59:59"),
    ).resolve()

    assert resolved.uids == {"n1", "n2"}


def test_time_range_scan_rejects_unparsable_bounds(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)

    with pytest.raises(ValueError, match="TimeRange.start"):
        TimeRangeConstraintResolver(
            graph.semantic_map, TimeRange(start="yesterday")
        ).resolve()


def test_constraint_declarations_are_validated() -> None:
    with pytest.raises(ValueError, match="field"):
        TimeRange(field=" ")
    with pytest.raises(ValueError, match="seed_uids"):
        RelationConstraint(seed_uids=[])
    with pytest.raises(ValueError, match="max_depth"):
        RelationConstraint(seed_uids=["u1"], max_depth=0)
    with pytest.raises(ValueError, match="direction"):
        RelationConstraint(seed_uids=["u1"], direction="outgoing")
    with pytest.raises(ValueError, match="relation_types"):
        RelationConstraint(seed_uids=["u1"], relation_types=[""])


def test_metadata_resolver_rejects_unknown_operator(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)

    with pytest.raises(ValueError, match="Unsupported filter operator"):
        MetadataConstraintResolver(
            graph.semantic_map, {"speaker": {"equals": "speaker_a"}}
        )


def test_metadata_resolver_reads_unit_metadata_dict(graph_factory) -> None:
    graph = graph_factory()
    graph.semantic_map.add_unit(
        MemoryUnit(
            "m1",
            {"text_content": "metadata-only field"},
            {"project": "alpha"},
        ),
        explicit_content_for_embedding="metadata-only field",
        space_names=["s-work"],
        generate_sparse_embedding=False,
    )

    resolved = MetadataConstraintResolver(
        graph.semantic_map, {"project": {"eq": "alpha"}}
    ).resolve()
    baseline = {
        unit.uid
        for unit in graph.semantic_map.filter_memory_units(
            filter_condition={"project": {"eq": "alpha"}}
        )
    }

    assert resolved.uids == baseline == {"m1"}


def test_time_range_resolver_skips_units_without_timestamp(graph_factory) -> None:
    units = [
        ("n1", "has a timestamp", "2024-03-01T10:00:00", "speaker_a", "s-daily"),
        ("n2", "no timestamp at all", None, "speaker_a", "s-daily"),
    ]
    graph = _build_graph(graph_factory, units)

    resolved = TimeRangeConstraintResolver(
        graph.semantic_map, TimeRange(start="2024-01-01T00:00:00")
    ).resolve()
    assert resolved.uids == {"n1"}


def test_space_resolver_returns_space_membership(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)

    resolved = SpaceConstraintResolver(
        graph.semantic_map, ["s-daily", "ms:s-work"]
    ).resolve()
    assert resolved.uids == {"u1", "u2", "u3", "u5", "u6"}


def test_resolvers_on_empty_map_return_empty_sets(graph_factory) -> None:
    semantic_map = graph_factory().semantic_map

    assert (
        MetadataConstraintResolver(semantic_map, {"speaker": {"eq": "speaker_a"}})
        .resolve()
        .uids
        == frozenset()
    )
    assert (
        TimeRangeConstraintResolver(semantic_map, TimeRange(start="2024-01-01"))
        .resolve()
        .uids
        == frozenset()
    )
    assert (
        SpaceConstraintResolver(semantic_map, ["s-daily"]).resolve().uids
        == frozenset()
    )


def test_plan_without_constraints_keeps_plain_retrieval(graph_factory) -> None:
    planner = ConstraintAwarePlanner(graph_factory())

    plan = planner.plan(QueryConstraints(), top_k=3)
    assert plan.mode == "none"
    assert plan.candidate_set is None
    assert plan.retrieval_top_k == 3


def test_plan_post_filter_expands_pool_and_intersects_leaves(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    planner = ConstraintAwarePlanner(graph)

    constraints = QueryConstraints(
        metadata_filters={"speaker": {"eq": "speaker_a"}},
        time_range=TimeRange(start="2024-01-01T00:00:00"),
        space_names=["s-daily"],
    )
    plan = planner.plan(constraints, top_k=2)

    assert plan.mode == "post_filter"
    assert plan.retrieval_top_k == 50
    assert plan.candidate_set.uids == {"u1", "u3", "u6"}
    assert plan.candidate_set.selectivity == pytest.approx(0.5)


def _build_relation_graph(graph_factory) -> SemanticGraph:
    """Build the default units plus a small explicit-relationship topology.

    Edges (all monodirectional unless noted): ``u1 -> u2`` (FOLLOWED_BY),
    ``u1 -> u3`` (MENTION_OF), ``u4 -> u1`` (MENTION_OF), ``u2 -> u5`` (NEXT).
    """
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    assert graph.add_relationship("u1", "u2", "FOLLOWED_BY")
    assert graph.add_relationship("u1", "u3", "MENTION_OF")
    assert graph.add_relationship("u4", "u1", "MENTION_OF")
    assert graph.add_relationship("u2", "u5", "NEXT")
    return graph


def test_relation_resolver_one_hop_both_directions(graph_factory) -> None:
    graph = _build_relation_graph(graph_factory)

    resolved = RelationConstraintResolver(
        graph.semantic_map, RelationConstraint(seed_uids=["u1"])
    ).resolve()

    # Successors u2/u3 and predecessor u4; the seed u1 itself is excluded.
    assert resolved.uids == {"u2", "u3", "u4"}
    assert resolved.source == "relation"
    assert resolved.total == len(DEFAULT_UNITS)


def test_relation_resolver_honors_direction_and_depth(graph_factory) -> None:
    graph = _build_relation_graph(graph_factory)
    semantic_map = graph.semantic_map

    successors = RelationConstraintResolver(
        semantic_map,
        RelationConstraint(seed_uids=["u1"], direction="successors"),
    ).resolve()
    assert successors.uids == {"u2", "u3"}

    predecessors = RelationConstraintResolver(
        semantic_map,
        RelationConstraint(seed_uids=["u1"], direction="predecessors"),
    ).resolve()
    assert predecessors.uids == {"u4"}

    depth_two = RelationConstraintResolver(
        semantic_map,
        RelationConstraint(seed_uids=["u1"], max_depth=2, direction="successors"),
    ).resolve()
    # u5 is two hops away through u2.
    assert depth_two.uids == {"u2", "u3", "u5"}


def test_relation_resolver_filters_by_relation_type(graph_factory) -> None:
    graph = _build_relation_graph(graph_factory)

    resolved = RelationConstraintResolver(
        graph.semantic_map,
        RelationConstraint(seed_uids=["u1"], relation_types=["MENTION_OF"]),
    ).resolve()

    assert resolved.uids == {"u3", "u4"}


def test_relation_resolver_handles_unknown_and_multiple_seeds(graph_factory) -> None:
    graph = _build_relation_graph(graph_factory)
    semantic_map = graph.semantic_map

    # A seed that is absent from the graph contributes no neighborhood.
    assert (
        RelationConstraintResolver(
            semantic_map,
            RelationConstraint(seed_uids=["missing"], max_depth=2),
        )
        .resolve()
        .uids
        == frozenset()
    )

    # Seeds are excluded even when they are neighbors of one another.
    resolved = RelationConstraintResolver(
        semantic_map, RelationConstraint(seed_uids=["u1", "u2"])
    ).resolve()
    assert resolved.uids == {"u3", "u4", "u5"}


def test_relation_resolver_excludes_memory_space_nodes(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    # Memory spaces are graph nodes; only retrievable memory units may enter the
    # candidate set, so the reached space node is dropped.
    assert graph.add_relationship("u1", "s-daily", "IN_SPACE")

    resolved = RelationConstraintResolver(
        graph.semantic_map, RelationConstraint(seed_uids=["u1"])
    ).resolve()

    assert resolved.uids == frozenset()


def test_relation_resolver_without_parent_graph_returns_empty(graph_factory) -> None:
    standalone_map = SemanticMap(
        embedding_model_name="test/dummy",
        embedding_dim=2,
        use_flash_attention=False,
    )

    resolved = RelationConstraintResolver(
        standalone_map, RelationConstraint(seed_uids=["u1"])
    ).resolve()

    assert resolved.uids == frozenset()


def test_plan_intersects_relation_with_other_constraints(graph_factory) -> None:
    graph = _build_relation_graph(graph_factory)
    planner = ConstraintAwarePlanner(graph)

    constraints = QueryConstraints(
        metadata_filters={"speaker": {"eq": "speaker_b"}},
        relation=RelationConstraint(seed_uids=["u1"]),
    )
    plan = planner.plan(constraints, top_k=2)

    # neighbors(u1) = {u2, u3, u4}; speaker_b among them is only u2.
    assert plan.candidate_set.uids == {"u2"}
    assert plan.mode == "pre_filter"


def test_execute_query_with_relation_constraint_verifies_neighbors(
    graph_factory, monkeypatch
) -> None:
    graph = _build_relation_graph(graph_factory)
    semantic_map = graph.semantic_map
    stub = _StubMultiRetriever(
        [
            (semantic_map.get_unit("u2"), 0.9),
            (semantic_map.get_unit("u1"), 0.8),
            (semantic_map.get_unit("u5"), 0.7),
        ]
    )
    monkeypatch.setattr(graph, "get_multi_retriever", lambda: stub)

    planner = ConstraintAwarePlanner(graph)
    constraints = QueryConstraints(relation=RelationConstraint(seed_uids=["u1"]))
    results = planner.execute_query("coffee", constraints, top_k=3)

    # u1 (the seed) and u5 (two hops away) violate the 1-hop constraint.
    assert [unit.uid for unit, _ in results] == ["u2"]


def test_search_constrained_combines_relation_and_metadata(graph_factory) -> None:
    graph = _build_relation_graph(graph_factory)
    constraints = QueryConstraints(
        metadata_filters={"speaker": {"eq": "speaker_b"}},
        relation=RelationConstraint(seed_uids=["u1"]),
    )

    results = graph.search_constrained(
        "coffee", constraints, top_k=5, methods=["cosine_similarity"]
    )

    # neighbors(u1) = {u2, u3, u4}; the only speaker_b neighbor is u2.
    assert {unit.uid for unit in results} == {"u2"}


def test_execute_query_post_filters_without_candidate_pushdown(
    graph_factory, monkeypatch
) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    semantic_map = graph.semantic_map
    stub = _StubMultiRetriever(
        [
            (semantic_map.get_unit("u1"), 0.9),
            (semantic_map.get_unit("u6"), 0.8),
            (semantic_map.get_unit("u3"), 0.7),
            (semantic_map.get_unit("u4"), 0.6),
        ]
    )
    monkeypatch.setattr(graph, "get_multi_retriever", lambda: stub)

    planner = ConstraintAwarePlanner(graph)
    constraints = QueryConstraints(time_range=TimeRange(start="2024-03-01T00:00:00"))
    results = planner.execute_query("coffee", constraints, top_k=2)

    # u1 falls outside the time window and is dropped; ranking order is kept.
    assert [unit.uid for unit, _ in results] == ["u6", "u3"]
    _, call_kwargs = stub.calls[0]
    assert call_kwargs["top_k"] == 50
    assert "candidate_uids" not in call_kwargs


def test_execute_query_skips_retrieval_when_candidates_are_empty(
    graph_factory, monkeypatch
) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    stub = _StubMultiRetriever([])
    monkeypatch.setattr(graph, "get_multi_retriever", lambda: stub)

    planner = ConstraintAwarePlanner(graph)
    constraints = QueryConstraints(metadata_filters={"speaker": {"eq": "speaker_z"}})

    assert planner.execute_query("coffee", constraints, top_k=3) == []
    assert stub.calls == []


def test_search_constrained_u1_equivalence_with_linear_baseline(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    constraints = QueryConstraints(time_range=TimeRange(start="2024-02-01T00:00:00"))
    top_k = 3

    constrained = graph.search_constrained(
        "coffee",
        constraints,
        top_k=top_k,
        return_score=True,
        methods=["cosine_similarity"],
    )

    # Baseline: full retrieval, then the existing linear filter.
    retriever = graph.get_multi_retriever()
    baseline_pool = retriever.smart_search(
        "coffee", top_k=50, methods=["cosine_similarity"]
    )
    allowed = {
        unit.uid
        for unit in graph.semantic_map.filter_memory_units(
            filter_condition={"timestamp": {"gte": "2024-02-01T00:00:00"}}
        )
    }
    baseline = [(unit, score) for unit, score in baseline_pool if unit.uid in allowed][:top_k]

    assert constrained
    assert [unit.uid for unit, _ in constrained] == [unit.uid for unit, _ in baseline]


def test_search_constrained_without_constraints_matches_plain_smart_search(
    graph_factory,
) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    retriever = graph.get_multi_retriever()

    plain = retriever.smart_search("coffee", top_k=3, methods=["cosine_similarity"])
    constrained = graph.search_constrained(
        "coffee", top_k=3, return_score=True, methods=["cosine_similarity"]
    )

    assert [unit.uid for unit, _ in constrained] == [unit.uid for unit, _ in plain]


def test_search_constrained_satisfies_all_constraints_end_to_end(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    constraints = QueryConstraints(
        metadata_filters={"speaker": {"in": ["speaker_a", "speaker_b"]}},
        time_range=TimeRange(end="2024-04-30T23:59:59"),
        space_names=["s-daily", "s-work"],
    )

    results = graph.search_constrained(
        "coffee", constraints, top_k=5, methods=["cosine_similarity"]
    )

    assert {unit.uid for unit in results} == {"u1", "u2", "u3"}
    for unit in results:
        assert isinstance(unit, MemoryUnit)
        assert unit.raw_data["speaker"] in {"speaker_a", "speaker_b"}
        assert unit.raw_data["timestamp"] <= "2024-04-30T23:59:59"
        assert unit.uid in {"u1", "u2", "u3", "u5", "u6"}


def test_search_constrained_returns_empty_for_unsatisfiable_constraints(
    graph_factory,
) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    constraints = QueryConstraints(metadata_filters={"speaker": {"eq": "speaker_z"}})

    assert graph.search_constrained("coffee", constraints) == []


def _spy_index_lookup(monkeypatch, semantic_map) -> list:
    """Record ``MetadataIndex.lookup`` calls while preserving behavior."""
    calls: list = []
    index = semantic_map._metadata_index
    original_lookup = index.lookup

    def counting_lookup(filters):
        calls.append(filters)
        return original_lookup(filters)

    monkeypatch.setattr(index, "lookup", counting_lookup)
    return calls


def test_metadata_resolver_uses_registered_index(graph_factory, monkeypatch) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    semantic_map = graph.semantic_map
    semantic_map.register_filterable_fields({"speaker": "hash"})
    calls = _spy_index_lookup(monkeypatch, semantic_map)

    filters = {"speaker": {"in": ["speaker_a", "speaker_b"]}}
    resolved = MetadataConstraintResolver(semantic_map, filters).resolve()
    baseline = {
        unit.uid
        for unit in semantic_map.filter_memory_units(filter_condition=filters)
    }

    assert calls, "the registered field should be served by the metadata index"
    assert resolved.uids == baseline == {"u1", "u2", "u3", "u5", "u6"}
    assert resolved.total == len(DEFAULT_UNITS)


def test_unregistered_fields_fall_back_to_linear_scan(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    semantic_map = graph.semantic_map
    semantic_map.register_filterable_fields(["speaker"])

    filters = {"text_content": {"contain": "coffee"}}
    resolved = MetadataConstraintResolver(semantic_map, filters).resolve()
    baseline = {
        unit.uid
        for unit in semantic_map.filter_memory_units(filter_condition=filters)
    }

    assert resolved.uids == baseline == {"u1", "u3", "u5"}


def test_time_range_resolver_uses_sorted_index(graph_factory, monkeypatch) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    semantic_map = graph.semantic_map
    semantic_map.register_filterable_fields({"timestamp": "sorted"})
    calls = _spy_index_lookup(monkeypatch, semantic_map)

    window = TimeRange(start="2024-02-01T00:00:00", end="2024-04-30T23:59:59")
    resolved = TimeRangeConstraintResolver(semantic_map, window).resolve()
    baseline = {
        unit.uid
        for unit in semantic_map.filter_memory_units(
            filter_condition={"timestamp": {"gte": window.start, "lte": window.end}}
        )
    }

    assert calls
    assert resolved.uids == baseline == {"u2", "u3", "u4"}


def test_index_maintenance_tracks_insertions_deletions_and_tiered_swap(
    graph_factory,
) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    semantic_map = graph.semantic_map
    semantic_map.register_filterable_fields(["speaker"])
    filters = {"speaker": {"eq": "speaker_a"}}

    extra = MemoryUnit("u7", {"text_content": "late coffee note", "speaker": "speaker_a"})
    semantic_map.add_unit(
        extra,
        explicit_content_for_embedding="late coffee note",
        space_names=["s-daily"],
        generate_sparse_embedding=False,
    )
    assert MetadataConstraintResolver(semantic_map, filters).resolve().uids == {
        "u1",
        "u3",
        "u6",
        "u7",
    }

    semantic_map.delete_unit("u7")
    assert MetadataConstraintResolver(semantic_map, filters).resolve().uids == {
        "u1",
        "u3",
        "u6",
    }

    evicted = semantic_map.get_unit("u1")
    # Simulate the known-UID catalog a tiered map maintains across paging.
    semantic_map._storage_uids = {uid for uid, *_ in DEFAULT_UNITS}
    assert semantic_map._remove_from_l1_for_tiered_swap(["u1"]) == 1

    # The constraint index retains evicted values so constrained retrieval can
    # still resolve cold units; the legacy L1-only linear filter cannot.
    resolved = MetadataConstraintResolver(semantic_map, filters).resolve()
    baseline = {
        unit.uid
        for unit in semantic_map.filter_memory_units(filter_condition=filters)
    }
    assert resolved.uids == {"u1", "u3", "u6"}
    assert baseline == {"u3", "u6"}
    assert resolved.total == len(DEFAULT_UNITS)

    semantic_map._add_to_l1_from_tiered_swap([evicted])
    assert MetadataConstraintResolver(semantic_map, filters).resolve().uids == {
        "u1",
        "u3",
        "u6",
    }


def test_plan_prefilters_selective_candidates(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    planner = ConstraintAwarePlanner(graph)

    selective = QueryConstraints(metadata_filters={"speaker": {"eq": "speaker_c"}})
    plan = planner.plan(selective, top_k=2)
    assert plan.mode == "pre_filter"
    assert plan.candidate_set.uids == {"u4"}
    assert plan.retrieval_top_k == 50

    broad = QueryConstraints(space_names=["s-daily", "s-work"])
    assert planner.plan(broad, top_k=2).mode == "post_filter"


def test_execute_query_pushes_candidate_uids_when_prefiltering(
    graph_factory, monkeypatch
) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    semantic_map = graph.semantic_map
    stub = _StubMultiRetriever(
        [(semantic_map.get_unit("u4"), 0.9), (semantic_map.get_unit("u1"), 0.8)]
    )
    monkeypatch.setattr(graph, "get_multi_retriever", lambda: stub)

    planner = ConstraintAwarePlanner(graph)
    constraints = QueryConstraints(metadata_filters={"speaker": {"eq": "speaker_c"}})
    results = planner.execute_query("hiking", constraints, top_k=2)

    # u1 violates the constraint and is verified out of the fused results.
    assert [unit.uid for unit, _ in results] == ["u4"]
    _, call_kwargs = stub.calls[0]
    assert call_kwargs["candidate_uids"] == ["u4"]
    assert call_kwargs["top_k"] == 50


def test_execute_query_intersects_caller_candidates_on_pushdown(
    graph_factory, monkeypatch
) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    semantic_map = graph.semantic_map
    stub = _StubMultiRetriever([(semantic_map.get_unit("u4"), 0.9)])
    monkeypatch.setattr(graph, "get_multi_retriever", lambda: stub)

    planner = ConstraintAwarePlanner(graph)
    constraints = QueryConstraints(metadata_filters={"speaker": {"eq": "speaker_c"}})

    results = planner.execute_query(
        "hiking", constraints, top_k=2, candidate_uids=["u4", "u0"]
    )
    assert [unit.uid for unit, _ in results] == ["u4"]
    _, call_kwargs = stub.calls[0]
    assert call_kwargs["candidate_uids"] == ["u4"]

    stub.calls.clear()
    assert (
        planner.execute_query("hiking", constraints, top_k=2, candidate_uids=["u1"])
        == []
    )
    assert stub.calls == []


def test_execute_query_propagates_caller_candidates_on_post_filter(
    graph_factory, monkeypatch
) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    semantic_map = graph.semantic_map
    stub = _StubMultiRetriever(
        [(semantic_map.get_unit("u6"), 0.9), (semantic_map.get_unit("u3"), 0.8)]
    )
    monkeypatch.setattr(graph, "get_multi_retriever", lambda: stub)

    planner = ConstraintAwarePlanner(graph)
    constraints = QueryConstraints(time_range=TimeRange(start="2024-03-01T00:00:00"))
    assert planner.plan(constraints, top_k=2).mode == "post_filter"

    results = planner.execute_query(
        "coffee", constraints, top_k=2, candidate_uids=["u6", "u1"]
    )
    assert [unit.uid for unit, _ in results] == ["u6"]
    _, call_kwargs = stub.calls[0]
    # The caller filter is intersected with the constraints before push-down.
    assert call_kwargs["candidate_uids"] == ["u6"]


def test_execute_query_verifies_caller_candidates_without_constraints(
    graph_factory, monkeypatch
) -> None:
    """A caller-supplied candidate filter is a constraint and must be re-checked.

    Backends may push ``candidate_uids`` down, but graph expansion can still
    return units outside the caller set; the planner must filter them even when
    no declarative constraint is present.
    """
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    semantic_map = graph.semantic_map
    stub = _StubMultiRetriever(
        [(semantic_map.get_unit("u4"), 0.9), (semantic_map.get_unit("u1"), 0.8)]
    )
    monkeypatch.setattr(graph, "get_multi_retriever", lambda: stub)

    planner = ConstraintAwarePlanner(graph)
    results = planner.execute_query(
        "hiking", QueryConstraints(), top_k=2, candidate_uids=["u4"]
    )

    assert [unit.uid for unit, _ in results] == ["u4"]
    _, call_kwargs = stub.calls[0]
    assert call_kwargs["candidate_uids"] == ["u4"]


def test_execute_query_expands_pool_until_constrained_results_fill_top_k(
    graph_factory, monkeypatch
) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    stub = _SaturatedMultiRetriever(graph.semantic_map)
    monkeypatch.setattr(graph, "get_multi_retriever", lambda: stub)

    planner = ConstraintAwarePlanner(graph)
    results = planner.execute_query(
        "coffee", QueryConstraints(space_names=["s-daily"]), top_k=2
    )

    assert [unit.uid for unit, _ in results] == ["u1", "u3"]
    assert stub.calls == [50, 100]


def test_plan_records_dropped_constraints(graph_factory, monkeypatch) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    planner = ConstraintAwarePlanner(
        graph, fallback_policy=FallbackPolicy(DROP_CONSTRAINT)
    )
    constraints = QueryConstraints(
        metadata_filters={"speaker": {"eq": "speaker_c"}},
        space_names=["s-work"],
    )

    _spy_on_planner_warnings(monkeypatch)
    plan = planner.plan(constraints, top_k=5)

    assert plan.candidate_set.uids == {"u2", "u5"}
    assert plan.dropped_constraints == ("metadata_filters",)


def test_search_constrained_prefilter_matches_post_filter_baseline(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    constraints = QueryConstraints(metadata_filters={"speaker": {"eq": "speaker_c"}})
    top_k = 3

    constrained = graph.search_constrained(
        "weekend hiking",
        constraints,
        top_k=top_k,
        return_score=True,
        methods=["cosine_similarity"],
    )

    # Baseline: full retrieval, then the existing linear filter.
    retriever = graph.get_multi_retriever()
    baseline_pool = retriever.smart_search(
        "weekend hiking", top_k=50, methods=["cosine_similarity"]
    )
    allowed = {
        unit.uid
        for unit in graph.semantic_map.filter_memory_units(
            filter_condition={"speaker": {"eq": "speaker_c"}}
        )
    }
    baseline = [
        (unit, score) for unit, score in baseline_pool if unit.uid in allowed
    ][:top_k]

    assert constrained
    assert [unit.uid for unit, _ in constrained] == [unit.uid for unit, _ in baseline]
    assert [unit.uid for unit, _ in constrained] == ["u4"]


def test_relation_type_index_tracks_add_delete_and_unit_removal(graph_factory) -> None:
    graph = _build_relation_graph(graph_factory)
    index = graph._relation_type_index

    assert set(index.relation_types) == {"FOLLOWED_BY", "MENTION_OF", "NEXT"}
    assert index.edge_count(["MENTION_OF"]) == 2

    # Bidirectional relationships register both directions.
    assert graph.add_relationship("u5", "u6", "SIBLING", bidirectional=True)
    assert index.edge_count(["SIBLING"]) == 2

    assert graph.delete_relationship("u1", "u3", "MENTION_OF") is True
    assert index.edge_count(["MENTION_OF"]) == 1

    # Deleting a unit drops every incident edge from the index.
    graph.delete_unit("u2")
    assert index.edge_count(["FOLLOWED_BY"]) == 0
    assert index.edge_count(["NEXT"]) == 0
    assert index.edge_count(["MENTION_OF"]) == 1  # u4 -> u1 survives


def test_get_edges_by_relation_types_filters_and_limits(graph_factory) -> None:
    graph = _build_relation_graph(graph_factory)

    all_edges = graph.get_edges_by_relation_types()
    assert len(all_edges) == 4
    assert all(isinstance(attrs, dict) for _, _, attrs in all_edges)

    mentions = graph.get_edges_by_relation_types(["MENTION_OF"])
    assert {tuple(sorted((src, tgt))) for src, tgt, _ in mentions} == {
        ("u1", "u3"),
        ("u1", "u4"),
    }
    assert {attrs["type"] for _, _, attrs in mentions} == {"MENTION_OF"}

    assert len(graph.get_edges_by_relation_types(limit=2)) == 2
    assert graph.get_edges_by_relation_types(["MISSING"]) == []


def test_search_graph_relations_reads_the_relation_type_index(
    graph_factory, monkeypatch
) -> None:
    graph = _build_relation_graph(graph_factory)

    calls = []
    original_lookup = graph._relation_type_index.lookup

    def spy_lookup(relation_types=None):
        calls.append(relation_types)
        return original_lookup(relation_types)

    monkeypatch.setattr(graph._relation_type_index, "lookup", spy_lookup)

    edges = graph.search_graph_relations(relation_types=["MENTION_OF"], limit=10)
    # The type-filtered path goes through the inverted index, not an edge scan.
    assert calls == [["MENTION_OF"]]
    assert {tuple(sorted((src, tgt))) for src, tgt, _ in edges} == {
        ("u1", "u3"),
        ("u1", "u4"),
    }

    # An unfiltered lookup returns every indexed edge, with no retriever needed.
    assert len(graph.search_graph_relations(limit=10)) == 4
    assert calls[-1] is None


def test_fallback_policy_default_keeps_degenerate_candidate_set(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    planner = ConstraintAwarePlanner(graph)
    # speaker_c -> {u4} and s-work -> {u2, u5} share no unit.
    constraints = QueryConstraints(
        metadata_filters={"speaker": {"eq": "speaker_c"}},
        space_names=["s-work"],
    )

    candidate_set = planner.resolve_candidates(constraints)

    assert len(candidate_set) == 0


def test_fallback_policy_drop_constraint_relaxes_to_the_largest_intersection(
    graph_factory, monkeypatch
) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    planner = ConstraintAwarePlanner(
        graph, fallback_policy=FallbackPolicy(DROP_CONSTRAINT)
    )
    constraints = QueryConstraints(
        metadata_filters={"speaker": {"eq": "speaker_c"}},
        space_names=["s-work"],
    )

    warnings = _spy_on_planner_warnings(monkeypatch)
    candidate_set = planner.resolve_candidates(constraints)

    # Dropping the more restrictive metadata leaf ({u4}) leaves the larger space
    # set, so that is the predicate the policy gives up.
    assert candidate_set.uids == {"u2", "u5"}
    assert any("metadata_filters" in message for message in warnings)


def test_fallback_policy_never_drops_the_last_constraint(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    planner = ConstraintAwarePlanner(
        graph, fallback_policy=FallbackPolicy(DROP_CONSTRAINT)
    )
    constraints = QueryConstraints(metadata_filters={"speaker": {"eq": "nobody"}})

    # A genuinely unmatched single predicate still yields its empty set rather
    # than silently becoming an unconstrained query.
    assert len(planner.resolve_candidates(constraints)) == 0


def test_fallback_policy_min_candidates_drops_repeatedly(
    graph_factory, monkeypatch
) -> None:
    graph = _build_relation_graph(graph_factory)
    planner = ConstraintAwarePlanner(
        graph, fallback_policy=FallbackPolicy(DROP_CONSTRAINT, min_candidates=2)
    )
    constraints = QueryConstraints(
        metadata_filters={"speaker": {"eq": "speaker_a"}},
        space_names=["s-work"],
        relation=RelationConstraint(seed_uids=["u1"]),
    )

    warnings = _spy_on_planner_warnings(monkeypatch)
    candidate_set = planner.resolve_candidates(constraints)

    assert len(candidate_set) >= 2
    assert sum("dropped constraint" in message for message in warnings) >= 2


def test_plan_records_mode_reason(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    planner = ConstraintAwarePlanner(graph)

    assert planner.plan(QueryConstraints(), top_k=5).mode_reason == "no_constraints"

    selective = QueryConstraints(metadata_filters={"speaker": {"eq": "speaker_c"}})
    selective_plan = planner.plan(selective, top_k=5)
    assert selective_plan.mode == "pre_filter"
    assert selective_plan.mode_reason == "selective_below_threshold"

    broad = QueryConstraints(metadata_filters={"speaker": {"eq": "speaker_a"}})
    broad_plan = planner.plan(broad, top_k=5)
    assert broad_plan.mode == "post_filter"
    assert broad_plan.mode_reason == "broad_above_threshold"


def test_post_filter_pool_scales_with_selectivity(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    planner = ConstraintAwarePlanner(graph)
    # A quarter of a 1000-unit corpus: filling top_k=20 needs ~80 fused results,
    # more than the baseline max(top_k * 3, 50) expansion.
    candidate_set = CandidateSet(
        frozenset(str(index) for index in range(250)), source="meta", total=1000
    )

    assert planner._retrieval_top_k(20, candidate_set, "post_filter") == 80
    # The pre-filter path already restricts retrieval, so it keeps the baseline.
    assert planner._retrieval_top_k(20, candidate_set, "pre_filter") == 60
    assert planner._retrieval_top_k(20, None, "none") == 20


def test_search_constrained_accepts_fallback_policy(graph_factory) -> None:
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    constraints = QueryConstraints(
        metadata_filters={"speaker": {"eq": "speaker_c"}},
        space_names=["s-work"],
    )

    relaxed = graph.search_constrained(
        "coffee",
        constraints,
        top_k=5,
        methods=["cosine_similarity"],
        fallback_policy=FallbackPolicy(DROP_CONSTRAINT),
    )

    assert relaxed
    assert all(unit.uid in {"u2", "u5"} for unit in relaxed)


class _StubGraphRetriever:
    """Records the node-search dispatch performed by ``search_graph_nodes``."""

    def __init__(self) -> None:
        self.calls = []

    def _semantic_node_search(self, query, top_k, **kwargs):
        self.calls.append(("semantic", query, top_k, kwargs))
        return ["semantic-result"]

    def _fulltext_node_search(self, query, top_k, **kwargs):
        self.calls.append(("fulltext", query, top_k, kwargs))
        return ["fulltext-result"]

    def _hybrid_node_search(self, query, top_k, **kwargs):
        self.calls.append(("hybrid", query, top_k, kwargs))
        return ["hybrid-result"]


class _StubRetrieverRegistry:
    """Minimal stand-in exposing only the retrievers mapping."""

    def __init__(self, retriever) -> None:
        self.retrievers = {RetrievalMethod.GRAPH_TRAVERSAL: retriever}


def test_search_graph_nodes_dispatches_to_existing_retriever_methods(
    graph_factory, monkeypatch
) -> None:
    """The facade must call the methods ``GraphRetriever`` actually defines.

    Before the fix every branch raised: ``RetrievalMethod`` is imported only
    under ``TYPE_CHECKING`` (runtime ``NameError``) and the three dispatch
    targets were misnamed (``hybrid_node_search`` etc.).
    """
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    stub = _StubGraphRetriever()
    monkeypatch.setattr(
        graph, "get_multi_retriever", lambda: _StubRetrieverRegistry(stub)
    )

    assert graph.search_graph_nodes("q", top_k=2, search_method="semantic") == [
        "semantic-result"
    ]
    assert graph.search_graph_nodes("q", top_k=2, search_method="fulltext") == [
        "fulltext-result"
    ]
    assert graph.search_graph_nodes("q", top_k=3, search_method="hybrid") == [
        "hybrid-result"
    ]
    # An unknown method still falls back to the hybrid path.
    assert graph.search_graph_nodes("q", top_k=3, search_method="other") == [
        "hybrid-result"
    ]

    assert [call[0] for call in stub.calls] == [
        "semantic",
        "fulltext",
        "hybrid",
        "hybrid",
    ]
    assert stub.calls[2][2] == 3


def test_get_node_neighbors_resolves_retrieval_method(
    graph_factory, monkeypatch
) -> None:
    """The delegation must not fail on the lazily imported ``RetrievalMethod``.

    ``GraphRetriever`` does not implement ``get_relevant_nodes``, so the stub
    supplies it: this pins the facade's own import and result hand-off, which is
    what the ``NameError`` used to break before any call could be made.
    """
    graph = _build_graph(graph_factory, DEFAULT_UNITS)
    unit = graph.get_unit("u1")

    class _StubNeighborRetriever:
        def get_relevant_nodes(self, seed_nodes, **kwargs):
            return {"u1": [unit]}

    monkeypatch.setattr(
        graph,
        "get_multi_retriever",
        lambda: _StubRetrieverRegistry(_StubNeighborRetriever()),
    )

    neighbors = graph.get_node_neighbors("u1")

    assert neighbors["all_neighbors"] == [unit]