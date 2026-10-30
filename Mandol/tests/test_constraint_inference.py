"""Rule-based constraint inference against the offline tower fixture."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from mandol.constraints import ConstraintInferrer

DAY = 86400.0


def _epoch(*parts) -> float:
    return datetime(*parts, tzinfo=timezone.utc).timestamp()


ANCHOR = _epoch(2024, 1, 11, 12, 0, 0)  # Thursday, 2024-01-11


@pytest.fixture
def inferrer(tower_graph_factory):
    graph = tower_graph_factory()
    return ConstraintInferrer(graph.semantic_map)


def test_schema_covers_registered_and_observed_fields(inferrer):
    schema = inferrer.schema()
    assert schema.kind("uid") == "hash"
    assert schema.kind("timestamp") == "sorted"
    assert schema.kind("speaker") == "string"
    assert schema.version.startswith("v1:")
    assert schema.has_value("speaker", "speaker_a")
    assert not schema.has_value("speaker", "ghost")


@pytest.mark.parametrize(
    "query,start,end",
    [
        ("2024-01-06 发生了什么", _epoch(2024, 1, 6), _epoch(2024, 1, 6) + DAY - 1),
        ("2024年1月6日", _epoch(2024, 1, 6), _epoch(2024, 1, 6) + DAY - 1),
        ("2024年1月", _epoch(2024, 1, 1), _epoch(2024, 2, 1) - 1),
    ],
)
def test_absolute_dates_become_time_range(inferrer, query, start, end):
    inferred = inferrer.infer(query)
    assert inferred.constraints is not None
    time_range = inferred.constraints.time_range
    assert time_range.start == start
    assert time_range.end == end
    assert inferred.confidence == pytest.approx(0.95)


@pytest.mark.parametrize(
    "query,start",
    [
        ("昨天发生了什么", _epoch(2024, 1, 10)),
        ("上周三", _epoch(2024, 1, 3)),
        ("上周", _epoch(2024, 1, 1)),
    ],
)
def test_relative_cues_use_explicit_anchor(inferrer, query, start):
    inferred = inferrer.infer(query, reference_time=ANCHOR)
    time_range = inferred.constraints.time_range
    assert time_range.start == start
    assert inferred.reference_time == ANCHOR


def test_relative_days_window_ends_at_anchor(inferrer):
    inferred = inferrer.infer("最近3天", reference_time=ANCHOR)
    time_range = inferred.constraints.time_range
    assert time_range.start == ANCHOR - 3 * DAY
    assert time_range.end == ANCHOR
    assert inferred.confidence == pytest.approx(0.8)


def test_relative_cue_without_anchor_is_dropped(inferrer):
    inferred = inferrer.infer("昨天发生了什么")
    assert inferred.constraints is None
    assert "time: no reference_time for relative cue" in inferred.dropped
    assert inferred.confidence == 0.0


def test_mounted_space_is_inferred(inferrer):
    inferred = inferrer.infer("在情景记忆里找 alpha")
    assert inferred.constraints.space_names == ["episodic_memory"]
    assert inferred.evidence["space"] == ["episodic_memory"]


def test_space_word_without_locative_is_ignored(inferrer):
    inferred = inferrer.infer("情景记忆 alpha")
    assert inferred.constraints is None


def test_unmounted_space_is_dropped():
    stub_map = SimpleNamespace(
        memory_units={}, memory_spaces={}, total_unit_count=0, _metadata_index=None
    )
    inferred = ConstraintInferrer(stub_map).infer("在情景记忆里找 alpha")
    assert inferred.constraints is None
    assert "space: episodic_memory: not mounted" in inferred.dropped


def test_attribute_filter_uses_observed_value(inferrer):
    inferred = inferrer.infer("speaker 是 speaker_a")
    assert inferred.constraints.metadata_filters == {"speaker": {"eq": "speaker_a"}}
    assert inferred.confidence == pytest.approx(0.8)


def test_attribute_filter_explicit_cue_raises_confidence(inferrer):
    inferred = inferrer.infer("只看 speaker 是 speaker_a 的记录")
    assert inferred.constraints.metadata_filters == {"speaker": {"eq": "speaker_a"}}
    assert inferred.confidence == pytest.approx(0.9)


def test_unobserved_value_is_rejected(inferrer):
    inferred = inferrer.infer("speaker 是 ghost")
    assert inferred.constraints is None
    assert "field: speaker: value 'ghost' not observed" in inferred.dropped


def test_unknown_field_is_rejected(inferrer):
    inferred = inferrer.infer("foo 是 bar")
    assert inferred.constraints is None
    assert "field: foo: not in schema" in inferred.dropped


def test_known_identifier_resolves_to_uid_filter(inferrer):
    inferred = inferrer.infer("# h0 的内容")
    assert inferred.constraints.metadata_filters == {"uid": {"in": ["h0"]}}
    assert inferred.confidence == pytest.approx(0.95)


def test_unknown_identifier_is_rejected(inferrer):
    inferred = inferrer.infer("编号为 zzz9")
    assert inferred.constraints is None
    assert any(item.startswith("identifier:") for item in inferred.dropped)


def test_relation_cue_is_reported_not_guessed(inferrer):
    inferred = inferrer.infer("Alice 的朋友")
    assert inferred.constraints is None
    assert "relation: no name->uid resolver" in inferred.dropped


@pytest.mark.parametrize("query", ["", "请总结一下"])
def test_queries_without_signals_yield_no_constraints(inferrer, query):
    inferred = inferrer.infer(query)
    assert inferred.constraints is None
    assert inferred.source == "rule"
    assert inferred.confidence == 0.0


def test_combined_constraints_take_minimum_confidence(inferrer):
    inferred = inferrer.infer("在情景记忆里找昨天的事", reference_time=ANCHOR)
    assert inferred.constraints.space_names == ["episodic_memory"]
    assert inferred.constraints.time_range is not None
    assert inferred.confidence == pytest.approx(0.8)
