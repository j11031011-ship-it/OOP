"""Rule-based constraint inference against the offline tower fixture."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from mandol.constraints import ConstraintInferrer
from mandol.core.memory_unit import MemoryUnit

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


# ---------------------------------------------------------------------------
# M1c: LLM fallback
# ---------------------------------------------------------------------------

LLM_QUERY = "帮我筛一下只属于 speaker_a 的对话"


class _FakeLLMClient:
    def __init__(self, reply: str = "{}", *, delay: float = 0.0, error: Exception = None):
        self.reply = reply
        self.delay = delay
        self.error = error
        self.calls = 0
        self.prompts = []

    def generate_answer(self, prompt, max_tokens=None, temperature=None):
        self.calls += 1
        self.prompts.append(prompt)
        assert temperature == 0.0
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.reply


def _llm_inferrer(graph, client, **kwargs):
    return ConstraintInferrer(graph.semantic_map, llm_client=client, **kwargs)


def test_rule_hit_skips_the_llm(tower_graph_factory):
    client = _FakeLLMClient('{"metadata_filters": {"speaker": {"eq": "speaker_b"}}}')
    inferrer = _llm_inferrer(tower_graph_factory(), client)
    inferred = inferrer.infer("speaker 是 speaker_a")
    assert inferred.source == "rule"
    assert inferred.constraints.metadata_filters == {"speaker": {"eq": "speaker_a"}}
    assert client.calls == 0


def test_llm_fallback_returns_validated_constraints(tower_graph_factory):
    client = _FakeLLMClient(
        '{"confidence": 0.8, "metadata_filters": {"speaker": {"eq": "speaker_a"}}}'
    )
    inferrer = _llm_inferrer(tower_graph_factory(), client)
    inferred = inferrer.infer(LLM_QUERY)
    assert inferred.source == "llm"
    assert inferred.confidence == pytest.approx(0.8)
    assert inferred.constraints.metadata_filters == {"speaker": {"eq": "speaker_a"}}
    assert inferred.evidence["attribute"] == ["llm:speaker:eq"]
    assert client.calls == 1
    prompt = client.prompts[0]
    assert "speaker" in prompt and "speaker_a" in prompt and "timestamp" in prompt


def test_llm_cache_serves_repeated_queries(tower_graph_factory):
    client = _FakeLLMClient('{"metadata_filters": {"speaker": {"eq": "speaker_a"}}}')
    inferrer = _llm_inferrer(tower_graph_factory(), client)
    first = inferrer.infer(LLM_QUERY)
    second = inferrer.infer(LLM_QUERY)
    assert client.calls == 1
    assert second.source == "llm"
    assert second.dropped == first.dropped
    second.dropped.append("mutated")
    assert first.dropped == []


def test_llm_cache_key_tracks_schema_version(tower_graph_factory):
    graph = tower_graph_factory()
    client = _FakeLLMClient('{"metadata_filters": {"speaker": {"eq": "speaker_a"}}}')
    inferrer = _llm_inferrer(graph, client)
    assert inferrer.infer(LLM_QUERY).source == "llm"
    graph.semantic_map.add_unit(
        MemoryUnit(
            "extra",
            {
                "text_content": "extra note",
                "speaker": "speaker_c",
                "timestamp": "2024-01-11T08:00:00",
            },
        ),
        explicit_content_for_embedding="extra note",
        space_names=["hierarchical_memory:L0_Observation"],
        generate_sparse_embedding=False,
    )
    assert inferrer.infer(LLM_QUERY).source == "llm"
    assert client.calls == 2


def test_llm_invalid_json_fails_open(tower_graph_factory):
    inferrer = _llm_inferrer(tower_graph_factory(), _FakeLLMClient("not json at all"))
    inferred = inferrer.infer(LLM_QUERY)
    assert inferred.constraints is None
    assert inferred.source == "rule"
    assert "llm: invalid JSON response" in inferred.dropped


def test_llm_reply_with_code_fence_is_parsed(tower_graph_factory):
    client = _FakeLLMClient(
        '```json\n{"metadata_filters": {"speaker": {"eq": "speaker_a"}}}\n```'
    )
    inferred = _llm_inferrer(tower_graph_factory(), client).infer(LLM_QUERY)
    assert inferred.source == "llm"
    assert inferred.constraints.metadata_filters == {"speaker": {"eq": "speaker_a"}}


def test_llm_timeout_fails_open(tower_graph_factory):
    client = _FakeLLMClient('{"metadata_filters": {"speaker": {"eq": "speaker_a"}}}', delay=0.4)
    inferrer = _llm_inferrer(tower_graph_factory(), client, llm_timeout=0.05)
    started = time.perf_counter()
    inferred = inferrer.infer(LLM_QUERY)
    elapsed = time.perf_counter() - started
    assert inferred.constraints is None
    assert inferred.dropped == ["llm: timeout after 0.05s"]
    assert elapsed < 0.3


def test_llm_exception_fails_open(tower_graph_factory):
    inferrer = _llm_inferrer(tower_graph_factory(), _FakeLLMClient(error=RuntimeError("boom")))
    inferred = inferrer.infer(LLM_QUERY)
    assert inferred.constraints is None
    assert inferred.dropped == ["llm: RuntimeError: boom"]


def test_llm_predicates_are_validated_against_the_schema(tower_graph_factory):
    client = _FakeLLMClient(
        '{"metadata_filters": {'
        '"ghost": {"eq": "x"}, '
        '"speaker": {"eq": "ghost"}, '
        '"timestamp": {"contain": "2024"}}}'
    )
    inferred = _llm_inferrer(tower_graph_factory(), client).infer(LLM_QUERY)
    assert inferred.constraints is None
    assert "llm: field 'ghost' not in schema" in inferred.dropped
    assert "llm: field 'speaker': value 'ghost' not observed" in inferred.dropped
    assert "llm: field 'timestamp': operator 'contain' not allowed" in inferred.dropped


def test_llm_accepts_unobserved_sorted_values(tower_graph_factory):
    client = _FakeLLMClient(
        '{"metadata_filters": {"timestamp": {"gte": "2024-01-06"}}}'
    )
    inferred = _llm_inferrer(tower_graph_factory(), client).infer(LLM_QUERY)
    assert inferred.constraints.metadata_filters == {"timestamp": {"gte": "2024-01-06"}}


def test_llm_unmounted_space_is_dropped(tower_graph_factory):
    client = _FakeLLMClient('{"space_names": ["ghost_space"]}')
    inferred = _llm_inferrer(tower_graph_factory(), client).infer(LLM_QUERY)
    assert inferred.constraints is None
    assert "llm: space: ghost_space: not mounted" in inferred.dropped


def test_llm_time_range_is_validated(tower_graph_factory):
    client = _FakeLLMClient(
        '{"time_range": {"start": "2024-01-05T00:00:00", "end": "2024-01-06T00:00:00"}}'
    )
    inferred = _llm_inferrer(tower_graph_factory(), client).infer(LLM_QUERY)
    assert inferred.constraints.time_range.start == _epoch(2024, 1, 5)
    assert inferred.constraints.time_range.end == _epoch(2024, 1, 6)
    assert inferred.evidence["time"] == ["llm"]


def test_llm_inverted_time_range_is_dropped(tower_graph_factory):
    client = _FakeLLMClient('{"time_range": {"start": 200, "end": 100}}')
    inferred = _llm_inferrer(tower_graph_factory(), client).infer(LLM_QUERY)
    assert inferred.constraints is None
    assert "llm: time: start after end" in inferred.dropped


def test_llm_fallback_is_wired_through_retriever_config(tower_retriever_factory):
    client = _FakeLLMClient('{"metadata_filters": {"speaker": {"eq": "speaker_a"}}}')
    retriever, _graph = tower_retriever_factory(
        constraint_llm_client=client, constraint_llm_timeout=1.0
    )
    payload = retriever.tri_tower_search_quantification(
        LLM_QUERY, top_k=5, constraints="auto"
    )
    info = payload["quantification"]["constraint"]
    assert info["source"] == "llm"
    assert info["branch"] == "filtered"
    assert {unit.uid for unit, _score in payload["results"]} <= {"h0", "h2", "e0"}
    assert client.calls == 1
