"""Metric and gold-set integrity tests for the M6 evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mandol.constraints import (
    QueryConstraints,
    TimeRange,
    canonical_predicates,
    empty_result_rate,
    recall_at_k,
    score_extraction,
    score_predictions,
)

GOLD_PATH = Path(__file__).resolve().parents[1] / "examples" / "data" / "constraint_gold.jsonl"


def _epoch(*parts) -> float:
    return datetime(*parts, tzinfo=timezone.utc).timestamp()


def test_canonical_predicates_normalize_shapes():
    constraints = QueryConstraints(
        metadata_filters={
            "speaker": {"eq": "speaker_a"},
            "uid": {"in": ["b", "a"]},
        },
        time_range=TimeRange(
            start="2024-01-06T00:00:00",
            end=_epoch(2024, 1, 6) + 86399,
        ),
        space_names=["episodic_memory"],
    )
    leaves = canonical_predicates(constraints)

    assert ("filter", "uid", "in", ("a", "b")) in leaves
    assert ("filter", "speaker", "eq", "speaker_a") in leaves
    assert ("space", "episodic_memory") in leaves
    time_leaf = next(leaf for leaf in leaves if leaf[0] == "time")
    assert time_leaf[1] == "timestamp"
    assert time_leaf[2] == pytest.approx(_epoch(2024, 1, 6), abs=1e-3)
    assert time_leaf[3] == pytest.approx(_epoch(2024, 1, 6) + 86399, abs=1e-3)


def test_none_constraints_have_no_predicates():
    assert canonical_predicates(None) == set()
    assert canonical_predicates(QueryConstraints()) == set()


def test_score_predictions_counts_tp_fp_fn_per_type():
    predicted = {("filter", "speaker", "eq", "speaker_a")}
    expected = {
        ("filter", "speaker", "eq", "speaker_a"),
        ("space", "episodic_memory"),
    }
    score = score_predictions([(predicted, expected)])

    assert (score.overall.tp, score.overall.fp, score.overall.fn) == (1, 0, 1)
    assert score.overall.precision == pytest.approx(1.0)
    assert score.overall.recall == pytest.approx(0.5)
    assert score.overall.f1 == pytest.approx(2 / 3)
    assert score.by_type["filter"].tp == 1
    assert score.by_type["space"].fn == 1
    assert score.by_type["space"].precision == 0.0


def test_score_predictions_penalizes_wrong_values():
    predicted = {("filter", "speaker", "eq", "speaker_b")}
    expected = {("filter", "speaker", "eq", "speaker_a")}
    score = score_predictions([(predicted, expected)])
    assert (score.overall.tp, score.overall.fp, score.overall.fn) == (0, 1, 1)
    assert score.overall.f1 == 0.0


def test_score_extraction_accepts_bundles_and_none():
    predicted = [None, QueryConstraints(space_names=["episodic_memory"])]
    expected = [QueryConstraints(), QueryConstraints(space_names=["episodic_memory"])]
    score = score_extraction(predicted, expected)
    assert (score.overall.tp, score.overall.fp, score.overall.fn) == (1, 0, 0)
    assert score.overall.f1 == 1.0


def test_extraction_score_is_json_ready():
    score = score_predictions([({("time", "timestamp", 1.0, 2.0)}, set())])
    payload = score.to_dict()
    assert payload["tp"] == 0
    assert payload["fp"] == 1
    assert payload["precision"] == 0.0
    assert payload["by_type"]["time"]["fp"] == 1
    assert payload["by_type"]["time"]["f1"] == 0.0


def test_empty_result_rate():
    assert empty_result_rate([0, 3, 0]) == pytest.approx(2 / 3)
    assert empty_result_rate([1, 2]) == 0.0
    assert empty_result_rate([]) == 0.0


def test_recall_at_k():
    retrieved = ["a", "b", "c"]
    relevant = {"b", "d"}
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(0.5)
    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert recall_at_k(retrieved, relevant, 0) == 0.0
    assert recall_at_k(retrieved, set(), 3) == 1.0


def test_gold_file_has_fifty_well_formed_entries():
    entries = [
        json.loads(line)
        for line in GOLD_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(entries) == 50
    assert len({entry["id"] for entry in entries}) == 50

    allowed_types = {"time", "metadata", "space", "mixed", "uid", "none"}
    allowed_leaves = {"time", "filter", "space"}
    for entry in entries:
        assert entry["type"] in allowed_types
        assert entry["query"].strip()
        assert isinstance(entry["expected"], list)
        for leaf in entry["expected"]:
            assert leaf["type"] in allowed_leaves
            if leaf["type"] == "filter":
                assert leaf["field"] and leaf["value"] is not None
            elif leaf["type"] == "space":
                assert leaf["name"]
        if entry["type"] == "none":
            assert entry["expected"] == []
        else:
            assert entry["expected"]
