"""M5: ``search_constrained`` delegation and the default ``constraints`` flip."""

from __future__ import annotations

import inspect

from mandol.constraints import DROP_CONSTRAINT, FallbackPolicy, QueryConstraints
from mandol.triple_retrieval.triple_tower_retriever import TripleTowerRetriever

SPEAKER_A = QueryConstraints(metadata_filters={"speaker": {"eq": "speaker_a"}})
SPEAKER_A_UIDS = {"e0", "h0", "h2"}


def test_public_entries_default_to_auto():
    for func in (
        TripleTowerRetriever.search,
        TripleTowerRetriever.search_async,
        TripleTowerRetriever.tri_tower_search_quantification,
        TripleTowerRetriever.tri_tower_search_quantification_async,
        TripleTowerRetriever.smart_search,
        TripleTowerRetriever.smart_search_async,
    ):
        default = inspect.signature(func).parameters["constraints"].default
        assert default == "auto", func.__name__


def test_default_entry_infers_space_constraints(tower_retriever_factory):
    retriever, _graph = tower_retriever_factory()
    payload = retriever.tri_tower_search_quantification("在情景记忆里找 alpha")
    info = payload["quantification"]["constraint"]
    assert info["mode"] == "auto"
    assert info["source"] == "rule"
    assert info["branch"] == "space_only"
    assert info["spaces"] == ["episodic_memory"]
    assert payload["results"]
    assert {unit.uid for unit, _score in payload["results"]} <= {"e0", "e1"}


def test_default_search_exposes_auto_mode(tower_retriever_factory):
    retriever, _graph = tower_retriever_factory()
    result = retriever.search("alpha")
    assert result.quantification["constraint"]["mode"] == "auto"
    assert result.quantification["constraint"]["branch"] == "none"


def test_search_constrained_delegates_explicit_constraints(tower_graph_factory):
    graph = tower_graph_factory()
    unified = graph.search_constrained(
        "alpha", SPEAKER_A, top_k=5, return_score=True, use_unified=True
    )
    planner = graph.search_constrained("alpha", SPEAKER_A, top_k=5, return_score=True)

    assert unified
    assert {unit.uid for unit, _ in unified} == {unit.uid for unit, _ in planner}
    assert {unit.uid for unit, _ in unified} <= SPEAKER_A_UIDS
    assert all(unit.raw_data["speaker"] == "speaker_a" for unit, _ in unified)
    assert graph._unified_tower_retriever is not None


def test_search_constrained_delegation_satisfies_constraints(tower_graph_factory):
    graph = tower_graph_factory()
    constraints = QueryConstraints(
        metadata_filters={"speaker": {"eq": "speaker_b"}},
        space_names=["episodic_memory"],
    )
    units = graph.search_constrained("conversation", constraints, top_k=5, use_unified=True)

    assert [unit.uid for unit in units] == ["e1"]
    assert units[0].raw_data["speaker"] == "speaker_b"


def test_search_constrained_auto_is_forwarded_to_the_unified_entry(tower_graph_factory):
    graph = tower_graph_factory()
    units = graph.search_constrained(
        "在情景记忆里找 alpha", constraints="auto", top_k=5, use_unified=True
    )
    assert units
    assert {unit.uid for unit in units} <= {"e0", "e1"}


def test_search_constrained_none_stays_on_the_planner(tower_graph_factory):
    graph = tower_graph_factory()
    baseline = graph.search_constrained("alpha", top_k=3, return_score=True)
    repeated = graph.search_constrained(
        "alpha", top_k=3, return_score=True, use_unified=True
    )

    assert [unit.uid for unit, _ in repeated] == [unit.uid for unit, _ in baseline]
    assert getattr(graph, "_unified_tower_retriever", None) is None


def test_legacy_overrides_keep_the_planner(tower_graph_factory):
    graph = tower_graph_factory()
    constrained = graph.search_constrained(
        "alpha",
        SPEAKER_A,
        top_k=5,
        use_unified=True,
        prefilter_selectivity_threshold=0.9,
    )
    assert constrained
    assert getattr(graph, "_unified_tower_retriever", None) is None

    relaxed = graph.search_constrained(
        "alpha",
        QueryConstraints(
            metadata_filters={"speaker": {"eq": "ghost"}},
            space_names=["episodic_memory"],
        ),
        top_k=5,
        use_unified=True,
        fallback_policy=FallbackPolicy(DROP_CONSTRAINT),
    )
    assert relaxed
    assert {unit.uid for unit in relaxed} <= {"e0", "e1"}
    assert getattr(graph, "_unified_tower_retriever", None) is None


def test_delegated_retriever_is_shared_across_calls(tower_graph_factory):
    graph = tower_graph_factory()
    graph.search_constrained("alpha", SPEAKER_A, top_k=3, use_unified=True)
    shared = graph._unified_tower_retriever
    graph.search_constrained("beta", SPEAKER_A, top_k=3, use_unified=True)

    assert graph._unified_tower_retriever is shared
