"""Measure the unified triple-tower entry against the legacy constrained entry.

The unified entry (``TripleTowerRetriever.tri_tower_search_quantification`` with
``constraints=None | "auto" | QueryConstraints``) routes constraints *before*
retrieval and then runs the full S4-S9 pipeline. The legacy entry
(``ConstraintAwarePlanner.execute_query``) constraints the same retriever but
applies filtering around it. This script runs both over the same queries and
emits a JSON report with hit UIDs, counts, and elapsed time so the two can be
compared offline.

Without ``--graph-dir`` the script builds a small synthetic graph and patches in
deterministic dummy models, so it needs no network access and downloads nothing.

Run from the repository root:

    .venv/Scripts/python examples/eval_unified_entry.py --mode none
    .venv/Scripts/python examples/eval_unified_entry.py --mode explicit
    .venv/Scripts/python examples/eval_unified_entry.py --mode auto

An explicit query file is a JSON list of objects::

    [{"query": "speaker 是 speaker_a 的记录",
      "reference_time": "2024-01-11T00:00:00",
      "constraints": {"metadata_filters": {"speaker": {"eq": "speaker_a"}}}}]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from mandol.constraints import ConstraintAwarePlanner, ConstraintInferrer, QueryConstraints, TimeRange
from mandol.core.memory_space_registry import MemorySpaceRegistry
from mandol.core.memory_unit import MemoryUnit
from mandol.core.semantic_graph import SemanticGraph
from mandol.core.semantic_map import SemanticMap
from mandol.indexes.timestamp_normalizer import TimestampNormalizer
from mandol.triple_retrieval.triple_tower_retriever import (
    RerankStrategy,
    TripleTowerConfig,
    TripleTowerRetriever,
)
from mandol.utils.model_manager import global_model_manager

SYNTHETIC_UNITS = [
    ("h0", "alpha standup notes", "2024-01-05T08:00:00", "speaker_a", "hierarchical_memory:L0_Observation"),
    ("h1", "alpha weekly summary", "2024-01-06T09:00:00", "speaker_b", "hierarchical_memory:L1_Summary"),
    ("h2", "alpha insight pattern", "2024-01-07T10:00:00", "speaker_a", "hierarchical_memory:L2_Insight"),
    ("g0", "alpha project entity", "2024-01-08T11:00:00", "speaker_b", "entity_relation:mentions"),
    ("e0", "alpha conversation episode", "2024-01-09T12:00:00", "speaker_a", "episodic_memory"),
    ("e1", "beta conversation episode", "2024-01-10T13:00:00", "speaker_b", "episodic_memory"),
]

DEFAULT_QUERIES: List[Dict[str, Any]] = [
    {"query": "alpha"},
    {"query": "beta"},
    {"query": "昨天的 alpha", "reference_time": "2024-01-11T00:00:00"},
    {
        "query": "alpha 的记录，speaker 是 speaker_a",
        "constraints": {"metadata_filters": {"speaker": {"eq": "speaker_a"}}},
    },
    {"query": "在情景记忆里找 alpha"},
]


class _DummyEmbeddingModel:
    """Deterministic 2-dim encoder; matches the offline test fixtures."""

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


def _build_synthetic_graph() -> SemanticGraph:
    original_loader = global_model_manager.get_or_load_model

    def _load_model(**kwargs):
        if kwargs.get("model_type") == "text_embedding":
            return _DummyEmbeddingModel()
        return original_loader(**kwargs)

    global_model_manager.get_or_load_model = _load_model
    global_model_manager.get_splade_model = lambda *args, **kwargs: _DummySpladeModel()
    SemanticMap._incremental_aux_retriever_add = lambda self, units: None

    graph = SemanticGraph(
        SemanticMap(
            embedding_model_name="test/dummy",
            embedding_dim=2,
            use_flash_attention=False,
        )
    )
    MemorySpaceRegistry.initialize_spaces(graph)
    for uid, text, timestamp, speaker, space in SYNTHETIC_UNITS:
        raw_data = {"text_content": text, "speaker": speaker, "timestamp": timestamp}
        graph.semantic_map.add_unit(
            MemoryUnit(uid, raw_data),
            explicit_content_for_embedding=text,
            space_names=[space],
            generate_sparse_embedding=False,
        )
    return graph


def _build_retriever(graph: SemanticGraph) -> TripleTowerRetriever:
    return TripleTowerRetriever(
        graph,
        config=TripleTowerConfig(
            enable_routing=False,
            dispatch_mode="sequential",
            enable_second_stage_rerank=False,
            rerank_strategy=RerankStrategy.NONE,
            hierarchical_methods=["bm25"],
            hierarchical_rerank_method=None,
            graph_rerank_method=None,
            episodic_rerank_method=None,
        ),
    )


_NORMALIZER = TimestampNormalizer()


def _reference_epoch(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if float(value) > 0 else None
    return _NORMALIZER.normalize(value)


def _parse_constraints(spec: Optional[Dict[str, Any]]) -> Optional[QueryConstraints]:
    if not spec:
        return None
    time_spec = spec.get("time_range")
    time_range = TimeRange(**time_spec) if time_spec else None
    return QueryConstraints(
        metadata_filters=spec.get("metadata_filters"),
        time_range=time_range,
        space_names=spec.get("space_names"),
    )


def _load_queries(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return DEFAULT_QUERIES
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Query file must contain a JSON list of query objects.")
    return payload


def _run_unified(
    retriever: TripleTowerRetriever,
    item: Dict[str, Any],
    mode: str,
    top_k: int,
) -> Optional[Dict[str, Any]]:
    if mode == "none":
        declared: Any = None
    elif mode == "auto":
        declared = "auto"
    else:
        declared = _parse_constraints(item.get("constraints"))
        if declared is None:
            return {"skipped": "no explicit constraints declared"}

    pipeline_kwargs: Dict[str, Any] = {"top_k": top_k, "constraints": declared}
    reference_epoch = _reference_epoch(item.get("reference_time"))
    if reference_epoch is not None:
        pipeline_kwargs["query_timestamp"] = reference_epoch

    started = time.perf_counter()
    payload = retriever.tri_tower_search_quantification(item["query"], **pipeline_kwargs)
    elapsed = time.perf_counter() - started
    uids = [unit.uid for unit, _score in payload["results"]]
    return {
        "uids": uids,
        "count": len(uids),
        "elapsed_s": round(elapsed, 4),
        "branch": payload["quantification"]["constraint"]["branch"],
        "constraint": payload["quantification"]["constraint"],
    }


def _run_legacy(
    graph: SemanticGraph,
    item: Dict[str, Any],
    mode: str,
    top_k: int,
) -> Optional[Dict[str, Any]]:
    if mode == "none":
        return {"skipped": "mode none declares no constraints"}
    constraints = _parse_constraints(item.get("constraints"))
    if mode == "auto":
        inferred = ConstraintInferrer(graph.semantic_map).infer(
            item["query"], reference_time=item.get("reference_time")
        )
        constraints = inferred.constraints
    if constraints is None or constraints.is_empty():
        return {"skipped": "no constraints to apply"}

    planner = ConstraintAwarePlanner(graph)
    started = time.perf_counter()
    results = planner.execute_query(item["query"], constraints, top_k=top_k)
    elapsed = time.perf_counter() - started
    uids = [unit.uid for unit, _score in results]
    return {"uids": uids, "count": len(uids), "elapsed_s": round(elapsed, 4)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("none", "explicit", "auto"), default="none")
    parser.add_argument("--queries", help="JSON file with query objects; defaults to a small built-in set.")
    parser.add_argument("--graph-dir", help="Directory of a graph saved by SemanticGraph.save_graph.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", help="Output JSON path; defaults to evaluation/unified_entry_<mode>.json")
    args = parser.parse_args(argv)

    if args.graph_dir:
        graph = SemanticGraph.load_graph(args.graph_dir)
        source = args.graph_dir
    else:
        graph = _build_synthetic_graph()
        source = "synthetic"

    retriever = _build_retriever(graph)
    queries = _load_queries(args.queries)
    records = []
    for item in queries:
        record: Dict[str, Any] = {"query": item["query"], "mode": args.mode}
        try:
            record["unified"] = _run_unified(retriever, item, args.mode, args.top_k)
            record["legacy"] = _run_legacy(graph, item, args.mode, args.top_k)
        except Exception as exc:  # keep the report usable when one query fails
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "graph": source,
        "top_k": args.top_k,
        "queries": len(records),
        "records": records,
    }

    out_path = Path(args.out) if args.out else REPO_ROOT / "evaluation" / f"unified_entry_{args.mode}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for record in records:
        unified = record.get("unified") or {}
        legacy = record.get("legacy") or {}
        branch = unified.get("branch", "-")
        print(
            f"{record['query'][:32]:<32} branch={branch:<11} "
            f"unified={unified.get('count', '-')}/{unified.get('elapsed_s', '-')}s "
            f"legacy={legacy.get('count', '-')}/{legacy.get('elapsed_s', '-')}s"
        )
    print(f"\nReport written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
