"""Unified entry tests: baseline snapshot, routing branches, tower push-down.

The recorder retrievers stand in for the three tower backends so the tests can
observe exactly which keyword arguments the pipeline forwards; the baseline
snapshot asserts that ``constraints=None`` keeps the pre-change result order.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mandol.constraints import QueryConstraints
from mandol.triple_retrieval.triple_tower_retriever import TripleTowerConfig, TripleTowerRetriever

GOLDEN_PATH = Path(__file__).with_name("golden") / "tower_baseline.json"
TOWER_EXTRA_KWARGS = {"methods", "candidate_uids", "space_names"}


class _Recorder:
    def __init__(self):
        self.calls = []

    @property
    def extra_keys(self):
        return [set(call["extra"]) for call in self.calls]


class _RecordingHierarchical(_Recorder):
    def hierarchical_search(
        self,
        query,
        config=None,
        top_k=10,
        space_names=None,
        retrieval_methods=None,
        rerank_method=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "top_k": top_k,
                "space_names": space_names,
                "retrieval_methods": retrieval_methods,
                "extra": dict(kwargs),
            }
        )
        return SimpleNamespace(results_by_layer={})

    async def hierarchical_search_async(self, query, **kwargs):
        return self.hierarchical_search(query, **kwargs)


class _RecordingGraph(_Recorder):
    def entity_relation_search(
        self, query, top_k=10, final_topk=10, rerank_method=None, space_names=None, **kwargs
    ):
        self.calls.append(
            {"top_k": top_k, "space_names": space_names, "extra": dict(kwargs)}
        )
        return []

    async def entity_relation_search_async(self, query, **kwargs):
        return self.entity_relation_search(query, **kwargs)


class _RecordingEpisodic(_Recorder):
    def episodic_search(self, query, top_k=10, rerank_method=None, space_names=None, **kwargs):
        self.calls.append(
            {"top_k": top_k, "space_names": space_names, "extra": dict(kwargs)}
        )
        return []

    async def episodic_search_async(self, query, **kwargs):
        return self.episodic_search(query, **kwargs)

    @staticmethod
    def build_context_with_time_injection(results):
        return ""

    @staticmethod
    def build_context_with_metadata(results):
        return ""


@pytest.fixture
def recorded_retriever(tower_graph_factory):
    """Retriever with recording stand-ins for every tower backend."""

    def create():
        graph = tower_graph_factory()
        recorders = {
            "hierarchical": _RecordingHierarchical(),
            "graph": _RecordingGraph(),
            "episodic": _RecordingEpisodic(),
        }
        retriever = TripleTowerRetriever(
            graph,
            config=TripleTowerConfig(enable_routing=False),
            hierarchical_retriever=recorders["hierarchical"],
            graph_retriever=recorders["graph"],
            episodic_retriever=recorders["episodic"],
        )
        return retriever, recorders

    return create


BASELINE = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("query", sorted(BASELINE))
def test_constraints_none_matches_baseline_snapshot(tower_retriever_factory, query):
    retriever, _graph = tower_retriever_factory()
    payload = retriever.tri_tower_search_quantification(query, constraints=None)
    assert [unit.uid for unit, _score in payload["results"]] == BASELINE[query]


def test_constraints_none_is_repeatable(tower_retriever_factory):
    retriever, _graph = tower_retriever_factory()
    first = retriever.tri_tower_search_quantification("alpha")
    second = retriever.tri_tower_search_quantification("alpha")
    assert [u.uid for u, _ in first["results"]] == [u.uid for u, _ in second["results"]]


def test_empty_constraint_short_circuits_dispatch(recorded_retriever):
    retriever, recorders = recorded_retriever()
    payload = retriever.tri_tower_search_quantification(
        "alpha",
        constraints=QueryConstraints(metadata_filters={"uid": {"in": ["missing"]}}),
    )

    assert payload["results"] == []
    assert payload["context"] == ""
    assert payload["quantification"]["constraint"]["branch"] == "empty"
    assert payload["quantification"]["routing"]["active_towers"] is not None
    for recorder in recorders.values():
        assert recorder.calls == []


def test_space_only_branch_selects_tower_without_materializing_uids(recorded_retriever):
    retriever, recorders = recorded_retriever()
    payload = retriever.tri_tower_search_quantification(
        "alpha", constraints=QueryConstraints(space_names=["episodic_memory"])
    )

    assert payload["quantification"]["constraint"]["branch"] == "space_only"
    assert payload["quantification"]["constraint"]["selectivity"] is None
    assert recorders["hierarchical"].calls == []
    assert recorders["graph"].calls == []
    assert len(recorders["episodic"].calls) == 1
    call = recorders["episodic"].calls[0]
    assert call["space_names"] == ["episodic_memory"]
    assert "candidate_uids" not in call["extra"]


def test_filtered_branch_pushes_candidate_uids_to_every_tower(recorded_retriever):
    retriever, recorders = recorded_retriever()
    payload = retriever.tri_tower_search_quantification(
        "alpha", constraints=QueryConstraints(metadata_filters={"speaker": {"eq": "speaker_a"}})
    )

    constraint = payload["quantification"]["constraint"]
    assert constraint["branch"] == "filtered"
    assert constraint["selectivity"] == pytest.approx(3 / 6)

    expected = ["e0", "h0", "h2"]
    assert recorders["hierarchical"].calls[0]["extra"]["candidate_uids"] == expected
    assert recorders["graph"].calls[0]["extra"]["candidate_uids"] == expected
    assert recorders["episodic"].calls[0]["extra"]["candidate_uids"] == expected
    for recorder in recorders.values():
        assert set(recorder.calls[0]["extra"]) <= TOWER_EXTRA_KWARGS


def test_tower_kwargs_never_include_pipeline_level_keys(recorded_retriever):
    retriever, recorders = recorded_retriever()
    retriever.tri_tower_search_quantification(
        "alpha",
        constraints=QueryConstraints(metadata_filters={"speaker": {"eq": "speaker_a"}}),
        methods=["bm25"],
        category="single-hop",
    )
    for recorder in recorders.values():
        for call in recorder.calls:
            assert "constraints" not in call["extra"]
            assert "constraint_space_names" not in call["extra"]
            assert "category" not in call["extra"]


def test_too_broad_gate_keeps_router_decision(recorded_retriever):
    retriever, recorders = recorded_retriever()
    payload = retriever.tri_tower_search_quantification(
        "alpha",
        constraints=QueryConstraints(
            metadata_filters={"speaker": {"in": ["speaker_a", "speaker_b"]}}
        ),
    )

    constraint = payload["quantification"]["constraint"]
    assert constraint["branch"] == "too_broad"
    assert constraint["applied"] is False
    for recorder in recorders.values():
        assert len(recorder.calls) == 1
        assert recorder.calls[0]["extra"] == {}


def test_budget_scales_per_tower_by_selectivity(recorded_retriever):
    retriever, recorders = recorded_retriever()
    payload = retriever.tri_tower_search_quantification(
        "alpha", constraints=QueryConstraints(metadata_filters={"speaker": {"eq": "speaker_a"}})
    )

    budget = payload["quantification"]["constraint"]["budget"]
    assert budget["topk_hierarchical"] == 40
    assert budget["topk_graph"] == 20
    assert budget["topk_episodic"] == 30
    assert recorders["hierarchical"].calls[0]["top_k"] == 40
    assert recorders["graph"].calls[0]["top_k"] == 20
    assert recorders["episodic"].calls[0]["top_k"] == 30


def test_budget_caps_at_four_times_base(recorded_retriever):
    retriever, recorders = recorded_retriever()
    payload = retriever.tri_tower_search_quantification(
        "alpha", constraints=QueryConstraints(metadata_filters={"uid": {"in": ["h0"]}})
    )

    budget = payload["quantification"]["constraint"]["budget"]
    assert budget["topk_hierarchical"] == 80
    assert budget["topk_graph"] == 40
    assert budget["topk_episodic"] == 60
    assert budget["selectivity"] == round(1 / 6, 6)


def test_zero_intersection_skips_declared_tower(recorded_retriever):
    retriever, recorders = recorded_retriever()
    payload = retriever.tri_tower_search_quantification(
        "alpha",
        constraints=QueryConstraints(
            metadata_filters={"uid": {"in": ["e0"]}},
            space_names=["hierarchical_memory", "episodic_memory"],
        ),
    )

    constraint = payload["quantification"]["constraint"]
    assert constraint["skipped_towers"] == ["hierarchical"]
    # Towers outside the declared spaces are switched off by the space flag,
    # not by the zero-intersection check.
    assert recorders["hierarchical"].calls == []
    assert recorders["graph"].calls == []
    assert len(recorders["episodic"].calls) == 1
    assert recorders["episodic"].calls[0]["extra"]["candidate_uids"] == ["e0"]
