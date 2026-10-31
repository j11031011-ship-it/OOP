"""Gold-set evaluation for constraint inference and the unified entry.

Runs fully offline against a deterministic synthetic corpus: the embedding and
SPLADE models are dummies and every tower is searched with BM25 plus cosine, so
the numbers are reproducible on a laptop. The report covers three layers:

- extraction precision / recall / F1 of inferred leaves against the gold set
  (rule mode always; LLM fallback with ``--llm``);
- the auto-mode empty-result rate, which is the early-warning gate for a wrong
  predicate emptying the candidate pool;
- end-to-end recall@k of ``constraints=None`` / ``"auto"`` / explicit gold
  constraints, so applying constraints can be compared with not applying them.

Usage::

    python examples/eval_constraint_gold.py
    python examples/eval_constraint_gold.py --top-k 10 --out evaluation/gold.json
    python examples/eval_constraint_gold.py --llm   # needs a configured provider
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np

from mandol.constraints import (
    ConstraintInferrer,
    QueryConstraints,
    TimeRange,
    canonical_predicates,
    empty_result_rate,
    recall_at_k,
    score_extraction,
)
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

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = Path(__file__).resolve().parent / "data" / "constraint_gold.jsonl"

TOPIC_DATES = {
    "apollo": "2024-01-04",
    "beacon": "2024-01-06",
    "cinder": "2024-01-08",
    "delta": "2024-01-10",
    "ember": "2024-01-12",
    "fjord": "2024-02-02",
    "glacier": "2024-02-05",
    "harbor": "2024-02-08",
}
SPACES = [
    ("obs", "hierarchical_memory:L0_Observation", "observation"),
    ("sum", "hierarchical_memory:L1_Summary", "summary"),
    ("ins", "hierarchical_memory:L2_Insight", "insight"),
    ("ent", "entity_relation:mentions", "entity"),
    ("epi", "episodic_memory", "episode"),
]
SPEAKERS = ["speaker_a", "speaker_b", "speaker_c", "speaker_d"]

_NORMALIZER = TimestampNormalizer()


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


def build_corpus() -> List[Dict[str, Any]]:
    """Return the 40-unit synthetic corpus behind the gold queries."""
    corpus: List[Dict[str, Any]] = []
    for index, (topic, base) in enumerate(TOPIC_DATES.items()):
        start = datetime.fromisoformat(base).replace(tzinfo=timezone.utc)
        for offset, (label, space, word) in enumerate(SPACES):
            timestamp = start + timedelta(days=offset)
            speaker = SPEAKERS[(index + offset) % len(SPEAKERS)]
            corpus.append(
                {
                    "uid": f"{topic}-{label}",
                    "topic": topic,
                    "speaker": speaker,
                    "timestamp": timestamp.isoformat(),
                    "space": space,
                    "text": f"{topic} {word} notes from {speaker}",
                }
            )
    return corpus


def build_graph(corpus: Sequence[Dict[str, Any]]) -> SemanticGraph:
    """Build the graph with dummy models so no download is attempted."""
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
    for spec in corpus:
        raw_data = {
            "text_content": spec["text"],
            "speaker": spec["speaker"],
            "timestamp": spec["timestamp"],
            "topic": spec["topic"],
        }
        graph.semantic_map.add_unit(
            MemoryUnit(spec["uid"], raw_data),
            explicit_content_for_embedding=spec["text"],
            space_names=[spec["space"]],
            generate_sparse_embedding=False,
        )
    return graph


def build_retriever(
    graph: SemanticGraph,
    llm_client: Optional[Any] = None,
) -> TripleTowerRetriever:
    return TripleTowerRetriever(
        graph,
        config=TripleTowerConfig(
            enable_routing=False,
            dispatch_mode="sequential",
            enable_second_stage_rerank=False,
            rerank_strategy=RerankStrategy.NONE,
            hierarchical_methods=["bm25", "cosine_similarity"],
            hierarchical_rerank_method=None,
            graph_rerank_method=None,
            episodic_rerank_method=None,
            constraint_llm_client=llm_client,
        ),
    )


def _load_gold(path: Path) -> List[Dict[str, Any]]:
    entries = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not entries:
        raise ValueError(f"Gold file {path} is empty.")
    return entries


def _reference_epoch(value: Any) -> Optional[float]:
    if value is None:
        return None
    return _NORMALIZER.normalize(value)


def _constraints_from_expected(leaves: Iterable[Dict[str, Any]]) -> QueryConstraints:
    metadata_filters: Dict[str, Dict[str, Any]] = {}
    time_range = None
    spaces: List[str] = []
    for leaf in leaves:
        kind = leaf["type"]
        if kind == "time":
            time_range = TimeRange(
                start=leaf.get("start"),
                end=leaf.get("end"),
                field=leaf.get("field", "timestamp"),
            )
        elif kind == "filter":
            metadata_filters.setdefault(leaf["field"], {})[leaf.get("op", "eq")] = leaf.get(
                "value"
            )
        elif kind == "space":
            spaces.append(leaf["name"])
        else:
            raise ValueError(f"Unknown gold leaf type: {kind!r}")
    return QueryConstraints(
        metadata_filters=metadata_filters or None,
        time_range=time_range,
        space_names=spaces or None,
    )


def _predicates_json(predicates: Set[tuple]) -> List[List[Any]]:
    return [list(leaf) for leaf in sorted(predicates, key=repr)]


def _field_value(spec: Dict[str, Any], name: str) -> Any:
    return spec["uid"] if name == "uid" else spec.get(name)


def _matches(value: Any, operator: str, target: Any) -> bool:
    operator = str(operator or "eq").lower()
    if operator == "eq":
        return value == target
    if operator == "ne":
        return value != target
    if operator == "in":
        return value in target
    if operator == "nin":
        return value not in target
    if operator in {"gt", "gte", "lt", "lte"}:
        if operator == "gt":
            return value > target
        if operator == "gte":
            return value >= target
        if operator == "lt":
            return value < target
        return value <= target
    if operator == "contain":
        return str(target) in str(value)
    if operator == "not_contain":
        return str(target) not in str(value)
    raise ValueError(f"Unsupported operator in gold leaf: {operator!r}")


def _satisfies(spec: Dict[str, Any], leaves: Iterable[Dict[str, Any]]) -> bool:
    for leaf in leaves:
        kind = leaf["type"]
        if kind == "filter":
            if not _matches(_field_value(spec, leaf["field"]), leaf.get("op", "eq"), leaf["value"]):
                return False
        elif kind == "space":
            name = leaf["name"]
            if not (spec["space"] == name or spec["space"].startswith(f"{name}:")):
                return False
        elif kind == "time":
            epoch = _NORMALIZER.normalize(spec["timestamp"])
            start = _NORMALIZER.normalize(leaf["start"]) if leaf.get("start") is not None else None
            end = _NORMALIZER.normalize(leaf["end"]) if leaf.get("end") is not None else None
            if start is not None and epoch < start:
                return False
            if end is not None and epoch > end:
                return False
    return True


def relevant_uids(entry: Dict[str, Any], corpus: Sequence[Dict[str, Any]]) -> Set[str]:
    topic = entry.get("topic")
    return {
        spec["uid"]
        for spec in corpus
        if (not topic or spec["topic"] == topic) and _satisfies(spec, entry["expected"])
    }


def evaluate_extraction(
    entries: Sequence[Dict[str, Any]],
    graph: SemanticGraph,
    llm_client: Optional[Any] = None,
) -> tuple:
    """Score inferred leaves against the gold leaves."""
    inferrer = ConstraintInferrer(graph.semantic_map, llm_client=llm_client)
    predicted: List[Optional[QueryConstraints]] = []
    expected: List[QueryConstraints] = []
    records: List[Dict[str, Any]] = []
    for entry in entries:
        inferred = inferrer.infer(
            entry["query"], reference_time=_reference_epoch(entry.get("reference_time"))
        )
        gold = _constraints_from_expected(entry["expected"])
        predicted.append(inferred.constraints)
        expected.append(gold)
        gold_predicates = canonical_predicates(gold)
        inferred_predicates = canonical_predicates(inferred.constraints)
        records.append(
            {
                "id": entry["id"],
                "type": entry["type"],
                "query": entry["query"],
                "source": inferred.source,
                "confidence": round(inferred.confidence, 4),
                "predicted": _predicates_json(inferred_predicates),
                "expected": _predicates_json(gold_predicates),
                "match": inferred_predicates == gold_predicates,
                "dropped": list(inferred.dropped),
            }
        )
    return score_extraction(predicted, expected), records


def evaluate_end_to_end(
    retriever: TripleTowerRetriever,
    entries: Sequence[Dict[str, Any]],
    corpus: Sequence[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in entries:
        reference = _reference_epoch(entry.get("reference_time"))
        relevant = relevant_uids(entry, corpus)
        row: Dict[str, Any] = {
            "id": entry["id"],
            "type": entry["type"],
            "query": entry["query"],
            "relevant": len(relevant),
        }
        for name, declared in (
            ("none", None),
            ("auto", "auto"),
            ("explicit", _constraints_from_expected(entry["expected"])),
        ):
            kwargs: Dict[str, Any] = {"top_k": top_k, "constraints": declared}
            if reference is not None:
                kwargs["query_timestamp"] = reference
            payload = retriever.tri_tower_search_quantification(entry["query"], **kwargs)
            uids = [unit.uid for unit, _score in payload["results"]]
            constraint = payload["quantification"]["constraint"]
            row[name] = {
                "count": len(uids),
                "recall": round(recall_at_k(uids, relevant, top_k), 4),
                "mode": constraint["mode"],
                "branch": constraint["branch"],
            }
        rows.append(row)
    return rows


def summarize(rows: Sequence[Dict[str, Any]], top_k: int) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for mode in ("none", "auto", "explicit"):
        counts = [row[mode]["count"] for row in rows]
        recalls = [row[mode]["recall"] for row in rows]
        summary[mode] = {
            f"mean_recall@{top_k}": round(sum(recalls) / len(recalls), 4),
            "empty_result_rate": round(empty_result_rate(counts), 4),
        }
    return summary


def _print_score(label: str, score: Any) -> None:
    overall = score.overall
    print(
        f"{label}: P={overall.precision:.4f} R={overall.recall:.4f} "
        f"F1={overall.f1:.4f} (tp={overall.tp} fp={overall.fp} fn={overall.fn})"
    )
    for name, kind in score.by_type.items():
        print(
            f"  {name:<7} P={kind.precision:.4f} R={kind.recall:.4f} "
            f"F1={kind.f1:.4f} (tp={kind.tp} fp={kind.fp} fn={kind.fn})"
        )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=str(DEFAULT_GOLD), help="Gold JSONL path.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Also run the LLM fallback; needs a configured provider in .env.",
    )
    parser.add_argument("--out", help="Report path; defaults to evaluation/constraint_gold.json.")
    args = parser.parse_args(argv)

    entries = _load_gold(Path(args.gold))
    corpus = build_corpus()
    graph = build_graph(corpus)

    llm_client = None
    if args.llm:
        from mandol.llm.llm_client import LLMClient

        llm_client = LLMClient()

    retriever = build_retriever(graph, llm_client)

    rule_score, extraction_records = evaluate_extraction(entries, graph)
    extraction: Dict[str, Any] = {"rule": rule_score.to_dict()}
    if llm_client is not None:
        llm_score, _llm_records = evaluate_extraction(entries, graph, llm_client)
        extraction["llm"] = llm_score.to_dict()

    rows = evaluate_end_to_end(retriever, entries, corpus, args.top_k)
    summary = summarize(rows, args.top_k)
    mismatches = [record["id"] for record in extraction_records if not record["match"]]
    empty_auto = [row["id"] for row in rows if row["auto"]["count"] == 0]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gold": str(Path(args.gold)),
        "queries": len(entries),
        "corpus_units": len(corpus),
        "top_k": args.top_k,
        "extraction": extraction,
        "end_to_end": summary,
        "auto_empty_result_rate": summary["auto"]["empty_result_rate"],
        "extraction_mismatches": mismatches,
        "auto_empty_queries": empty_auto,
        "extraction_records": extraction_records,
        "end_to_end_records": rows,
    }

    out_path = Path(args.out) if args.out else REPO_ROOT / "evaluation" / "constraint_gold.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Gold queries: {len(entries)}  corpus units: {len(corpus)}  top_k: {args.top_k}")
    _print_score("rule extraction", rule_score)
    if "llm" in extraction:
        print(
            "llm extraction: "
            f"P={extraction['llm']['precision']} R={extraction['llm']['recall']} "
            f"F1={extraction['llm']['f1']}"
        )
    print("end-to-end:")
    for mode, values in summary.items():
        print(
            f"  {mode:<8} mean_recall@{args.top_k}={values[f'mean_recall@{args.top_k}']:.4f} "
            f"empty={values['empty_result_rate']:.2%}"
        )
    print(
        f"auto empty-result rate: {summary['auto']['empty_result_rate']:.2%} "
        f"(target < 2.00%)"
    )
    if mismatches:
        print(f"extraction mismatches ({len(mismatches)}): {', '.join(mismatches)}")
    if empty_auto:
        print(f"auto returned nothing for: {', '.join(empty_auto)}")
    print(f"\nReport written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
