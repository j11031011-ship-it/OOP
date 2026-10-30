"""Top-level triple-tower retrieval orchestrator.

The three tower classes are intentionally thin wrappers. This module owns routing,
dispatch, cross-tower pruning, unified reranking, and quantification packaging.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from ..constraints import DEFAULT_MAX_SELECTIVITY, ConstraintAwarePlanner, ConstraintInferrer, QueryConstraints
from ..core.memory_space_registry import TowerSpace
from ..core.memory_unit import MemoryUnit
from ..entity_relation.entity_relation_retriever import (
    EntityRelationGraphRetriever,
    GraphRetrievalConfig,
    create_graph_retriever,
)
from ..episodic.episodic_memory_retriever import (
    EpisodicMemoryRetriever,
    EpisodicRetrievalConfig,
    create_episodic_retriever,
)
from ..hierarchical.hierarchical_retriever import (
    HierarchicalRetriever,
    HierarchicalSearchConfig,
    MemoryLevel,
    create_hierarchical_retriever,
)
from ..memory_router.locomo_tower_router import LocomoTowerRouter
from ..memory_router.longmemeval_tower_router import LongMemEvalTowerRouter
from ..quantification import EnhancedCandidateChunk, TowerSource, create_cascade_pruner
from ..retrieval.rerank_manager import RerankerManager
from ..retrieval.retrieval_interface import RetrievalMethod
from ..retrieval.score_fusion import ScoreFusion
from ..utils.logging_config import create_module_logger

logger = create_module_logger("triple_retrieval.triple_tower_retriever")

_TOWER_BY_SPACE_ROOT = {
    "hierarchical_memory": "hierarchical",
    "entity_relation": "graph",
    "episodic_memory": "episodic",
}


class TripleTowerSearchMode(Enum):
    ALL_TOWERS = "all_towers"
    HIERARCHICAL_ONLY = "hierarchical_only"
    GRAPH_ONLY = "graph_only"
    EPISODIC_ONLY = "episodic_only"
    HIERARCHICAL_GRAPH = "hierarchical_graph"
    HIERARCHICAL_EPISODIC = "hierarchical_episodic"
    GRAPH_EPISODIC = "graph_episodic"


class RerankStrategy(Enum):
    TOWER_SEPARATE = "tower_separate"
    UNIFIED_RERANK = "unified_rerank"
    NONE = "none"


@dataclass
class TripleTowerConfig:
    search_mode: TripleTowerSearchMode = TripleTowerSearchMode.ALL_TOWERS

    hierarchical_top_k: int = 20
    hierarchical_methods: Optional[List[str]] = None
    hierarchical_fusion_method: str = "rrf"
    hierarchical_rerank_method: str = "baai"

    graph_topk_similarity: int = 15
    graph_topk_graph: int = 0
    graph_final_topk: int = 10
    graph_use_entity_relation: bool = False
    graph_rerank_method: str = "baai"
    graph_target_space: str = TowerSpace.GRAPH_ROOT.value

    episodic_top_k: int = 15
    episodic_rerank_method: str = "baai"
    episodic_target_space: str = TowerSpace.EPISODIC_ROOT.value
    episodic_enable_time_injection: bool = True

    enable_second_stage_rerank: bool = True
    second_stage_rerank_method: str = "baai"
    rerank_strategy: RerankStrategy = RerankStrategy.UNIFIED_RERANK
    final_top_k: int = 15
    rerank_threshold: float = 0.0

    fusion_weights: Optional[Dict[str, float]] = None

    enable_routing: bool = True
    router_dataset: str = "locomo"
    router_model_name: str = "gpt-4.1-mini"
    router_strategy: str = "aggressive"
    routing_enable_guidance: bool = False

    dispatch_mode: str = "auto"  # auto | sequential | parallel
    enable_parallel: bool = False
    max_workers: int = 3

    pruner_max_tokens: int = 2000
    pruner_mode: str = "dynamic_adaptive"
    pruner_use_raw_score: bool = False
    pruner_lambda_mmr: float = 0.6

    def __post_init__(self) -> None:
        if self.hierarchical_methods is None:
            self.hierarchical_methods = ["bm25", "cosine_similarity", "splade"]
        if self.fusion_weights is None:
            self.fusion_weights = {"hierarchical": 0.34, "graph": 0.33, "episodic": 0.33}


@dataclass
class TowerRetrievalStats:
    enabled: bool = False
    success: bool = False
    retrieved_count: int = 0
    retrieval_time: float = 0.0
    error_message: str = ""


@dataclass
class HierarchicalTowerResult:
    l0_results: List[Tuple[MemoryUnit, float]] = field(default_factory=list)
    l1_results: List[Tuple[MemoryUnit, float]] = field(default_factory=list)
    l2_results: List[Tuple[MemoryUnit, float]] = field(default_factory=list)
    context_text: str = ""
    stats: TowerRetrievalStats = field(default_factory=TowerRetrievalStats)

    @property
    def total_count(self) -> int:
        return len(self.l0_results) + len(self.l1_results) + len(self.l2_results)


@dataclass
class GraphTowerResult:
    results: List[Tuple[MemoryUnit, float]] = field(default_factory=list)
    context_text: str = ""
    retrieval_details: Dict[str, Any] = field(default_factory=dict)
    stats: TowerRetrievalStats = field(default_factory=TowerRetrievalStats)


@dataclass
class EpisodicTowerResult:
    results: List[Tuple[MemoryUnit, float]] = field(default_factory=list)
    context_with_time: str = ""
    context_with_metadata: str = ""
    retrieval_stats: Dict[str, Any] = field(default_factory=dict)
    stats: TowerRetrievalStats = field(default_factory=TowerRetrievalStats)


@dataclass
class SecondStageRerankResult:
    reranked_l0: List[Tuple[MemoryUnit, float]] = field(default_factory=list)
    reranked_graph: List[Tuple[MemoryUnit, float]] = field(default_factory=list)
    reranked_episodic: List[Tuple[MemoryUnit, float]] = field(default_factory=list)
    rerank_time: float = 0.0
    strategy_used: str = "none"
    first_stage_counts: Dict[str, int] = field(default_factory=dict)
    final_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class TripleTowerResult:
    query: str
    search_mode: TripleTowerSearchMode
    hierarchical: HierarchicalTowerResult = field(default_factory=HierarchicalTowerResult)
    graph: GraphTowerResult = field(default_factory=GraphTowerResult)
    episodic: EpisodicTowerResult = field(default_factory=EpisodicTowerResult)
    rerank_result: Optional[SecondStageRerankResult] = None
    fused_context: str = ""
    total_retrieval_time: float = 0.0
    towers_used: List[str] = field(default_factory=list)
    quantification: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_flat_results(self) -> List[Tuple[MemoryUnit, float]]:
        results: List[Tuple[MemoryUnit, float]] = []
        results.extend(self.hierarchical.l0_results)
        results.extend(self.hierarchical.l1_results)
        results.extend(self.hierarchical.l2_results)
        results.extend(self.graph.results)
        results.extend(self.episodic.results)
        return results


@dataclass
class _RouteDecision:
    enable_hierarchical: bool
    enable_graph: bool
    enable_episodic: bool
    topk_hierarchical: int
    topk_graph: int
    topk_episodic: int
    final_top_k: int
    active_towers: List[str]
    dataset: str
    category: Optional[Union[str, int]]
    confidence: float
    reason: str
    raw_config: Any = None
    constraint_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _ConstraintRouteContext:
    """Resolved constraint decision, shared by routing and dispatch."""

    branch: str = "none"
    candidate_uids: Optional[List[str]] = None
    space_names: List[str] = field(default_factory=list)
    tower_flags: Optional[Dict[str, bool]] = None
    selectivity: Optional[float] = None
    skipped_towers: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)


class TripleTowerRetriever:
    """Top-level triple-tower orchestrator."""

    def __init__(
        self,
        semantic_graph: Any,
        config: Optional[TripleTowerConfig] = None,
        hierarchical_interface: Optional[Any] = None,
        hierarchical_retriever: Optional[HierarchicalRetriever] = None,
        graph_retriever: Optional[EntityRelationGraphRetriever] = None,
        episodic_retriever: Optional[EpisodicMemoryRetriever] = None,
        reranker_manager: Optional[RerankerManager] = None,
        reranker_configs: Optional[Dict[str, str]] = None,
    ):
        del hierarchical_interface
        self.semantic_graph = semantic_graph
        self.config = config or TripleTowerConfig()
        self.reranker_configs = reranker_configs or {
            "baai": "BAAI/bge-reranker-v2-m3",
            "qwen": "Qwen/Qwen3-Reranker-0.6B",
            "jina": "jinaai/jina-reranker-v2-base-multilingual",
        }
        self.reranker_manager = reranker_manager
        self.hierarchical_retriever = hierarchical_retriever or create_hierarchical_retriever(semantic_graph)
        self.graph_retriever = graph_retriever or create_graph_retriever(
            semantic_graph=semantic_graph,
            topk_similarity=self.config.graph_topk_similarity,
            topk_graph=self.config.graph_topk_graph,
            final_topk=self.config.graph_final_topk,
            use_entity_relation=self.config.graph_use_entity_relation,
            rerank_method=self.config.graph_rerank_method,
            target_space=self.config.graph_target_space,
            reranker_manager=reranker_manager,
        )
        self.episodic_retriever = episodic_retriever or create_episodic_retriever(
            semantic_graph=semantic_graph,
            top_k=self.config.episodic_top_k,
            rerank_method=self.config.episodic_rerank_method,
            target_space=self.config.episodic_target_space,
            enable_time_injection=self.config.episodic_enable_time_injection,
            reranker_manager=reranker_manager,
        )
        self.stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "tower_usage": {"hierarchical": 0, "graph": 0, "episodic": 0},
            "avg_retrieval_time": 0.0,
        }

    def search(
        self,
        query: str,
        config: Optional[TripleTowerConfig] = None,
        constraints: Union[str, QueryConstraints, None] = None,
        **kwargs,
    ) -> TripleTowerResult:
        start = time.time()
        payload = self._execute_pipeline(
            query=query, cfg=config or self.config, constraints=constraints, **kwargs
        )
        tower_results = payload["tower_results"]
        result = TripleTowerResult(query=query, search_mode=(config or self.config).search_mode)
        result.hierarchical = tower_results["hierarchical"]
        result.graph = tower_results["graph"]
        result.episodic = tower_results["episodic"]
        result.fused_context = payload["context"]
        result.quantification = payload["quantification"]
        result.rerank_result = payload["rerank_result"]
        result.towers_used = payload["quantification"]["routing"]["active_towers"]
        result.total_retrieval_time = time.time() - start
        self._update_stats(result)
        return result

    async def search_async(
        self,
        query: str,
        config: Optional[TripleTowerConfig] = None,
        constraints: Union[str, QueryConstraints, None] = None,
        **kwargs,
    ) -> TripleTowerResult:
        start = time.time()
        payload = await self._execute_pipeline_async(
            query=query, cfg=config or self.config, constraints=constraints, **kwargs
        )
        tower_results = payload["tower_results"]
        result = TripleTowerResult(query=query, search_mode=(config or self.config).search_mode)
        result.hierarchical = tower_results["hierarchical"]
        result.graph = tower_results["graph"]
        result.episodic = tower_results["episodic"]
        result.fused_context = payload["context"]
        result.quantification = payload["quantification"]
        result.rerank_result = payload["rerank_result"]
        result.towers_used = payload["quantification"]["routing"]["active_towers"]
        result.total_retrieval_time = time.time() - start
        self._update_stats(result)
        return result

    def tri_tower_search(self, query: str, top_k: Optional[int] = None, **kwargs) -> List[Tuple[MemoryUnit, float]]:
        payload = self.tri_tower_search_quantification(query=query, top_k=top_k, **kwargs)
        return payload["results"]

    async def tri_tower_search_async(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[Tuple[MemoryUnit, float]]:
        payload = await self.tri_tower_search_quantification_async(query=query, top_k=top_k, **kwargs)
        return payload["results"]

    def tri_tower_search_quantification(
        self,
        query: str,
        methods: Optional[Union[str, List[str], List[RetrievalMethod]]] = None,
        top_k: Optional[int] = None,
        rerank_method: Optional[str] = None,
        rerank_params: Optional[Dict[str, Any]] = None,
        constraints: Union[str, QueryConstraints, None] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        cfg = kwargs.pop("config", self.config)
        if methods is not None:
            kwargs["methods"] = methods
        if rerank_method is not None:
            cfg = self._copy_config_with(cfg, second_stage_rerank_method=rerank_method)
        payload = self._execute_pipeline(
            query=query,
            cfg=cfg,
            top_k=top_k,
            rerank_params=rerank_params,
            constraints=constraints,
            **kwargs,
        )
        return {
            "results": payload["results"],
            "context": payload["context"],
            "quantification": payload["quantification"],
        }

    async def tri_tower_search_quantification_async(
        self,
        query: str,
        methods: Optional[Union[str, List[str], List[RetrievalMethod]]] = None,
        top_k: Optional[int] = None,
        rerank_method: Optional[str] = None,
        rerank_params: Optional[Dict[str, Any]] = None,
        constraints: Union[str, QueryConstraints, None] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        cfg = kwargs.pop("config", self.config)
        if methods is not None:
            kwargs["methods"] = methods
        if rerank_method is not None:
            cfg = self._copy_config_with(cfg, second_stage_rerank_method=rerank_method)
        payload = await self._execute_pipeline_async(
            query=query,
            cfg=cfg,
            top_k=top_k,
            rerank_params=rerank_params,
            constraints=constraints,
            **kwargs,
        )
        return {
            "results": payload["results"],
            "context": payload["context"],
            "quantification": payload["quantification"],
        }

    def smart_search_with_quantification(
        self,
        query: str,
        methods: Optional[Union[str, List[str], List[RetrievalMethod]]] = None,
        top_k: int = 10,
        rerank_method: Optional[str] = "baai",
        rerank_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return self.tri_tower_search_quantification(
            query=query,
            methods=methods,
            top_k=top_k,
            rerank_method=rerank_method,
            rerank_params=rerank_params,
            **kwargs,
        )

    async def smart_search_with_quantification_async(
        self,
        query: str,
        methods: Optional[Union[str, List[str], List[RetrievalMethod]]] = None,
        top_k: int = 10,
        rerank_method: Optional[str] = "baai",
        rerank_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return await self.tri_tower_search_quantification_async(
            query=query,
            methods=methods,
            top_k=top_k,
            rerank_method=rerank_method,
            rerank_params=rerank_params,
            **kwargs,
        )

    def smart_search(
        self,
        query: str,
        methods: Optional[Union[str, List[str], List[RetrievalMethod]]] = None,
        top_k: int = 10,
        rerank_method: Optional[str] = "baai",
        rerank_params: Optional[Dict[str, Any]] = None,
        return_detailed: bool = False,
        constraints: Union[str, QueryConstraints, None] = None,
        **kwargs,
    ) -> Union[List[Tuple[MemoryUnit, float]], Dict[str, Any]]:
        payload = self.tri_tower_search_quantification(
            query=query,
            methods=methods,
            top_k=top_k,
            rerank_method=rerank_method,
            rerank_params=rerank_params,
            constraints=constraints,
            **kwargs,
        )
        if not return_detailed:
            return payload["results"]
        return self._format_smart_search_payload(payload)

    async def smart_search_async(
        self,
        query: str,
        methods: Optional[Union[str, List[str], List[RetrievalMethod]]] = None,
        top_k: int = 10,
        rerank_method: Optional[str] = "baai",
        rerank_params: Optional[Dict[str, Any]] = None,
        return_detailed: bool = False,
        constraints: Union[str, QueryConstraints, None] = None,
        **kwargs,
    ) -> Union[List[Tuple[MemoryUnit, float]], Dict[str, Any]]:
        payload = await self.tri_tower_search_quantification_async(
            query=query,
            methods=methods,
            top_k=top_k,
            rerank_method=rerank_method,
            rerank_params=rerank_params,
            constraints=constraints,
            **kwargs,
        )
        if not return_detailed:
            return payload["results"]
        return self._format_smart_search_payload(payload)

    def benchmark_dispatch_modes(
        self,
        query: str,
        repeats: int = 1,
        update_config: bool = False,
        config: Optional[TripleTowerConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Measure sequential vs parallel tower dispatch for this graph and query."""
        cfg = config or self.config
        category = kwargs.pop("category", kwargs.pop("query_category", None))
        dataset = kwargs.pop("dataset", None)
        kwargs.pop("constraints", None)
        route = self._route_towers(query, cfg, category=category, dataset=dataset)
        self._ensure_sync_rerank_allowed_for_route(
            cfg,
            route,
            context="TripleTowerRetriever.benchmark_dispatch_modes",
        )
        active_count = len(route.active_towers)
        timings: Dict[str, Dict[str, Any]] = {}
        runs_per_mode = max(1, repeats)

        for mode in ("sequential", "parallel"):
            mode_cfg = self._copy_config_with(
                cfg,
                dispatch_mode=mode,
                enable_parallel=(mode == "parallel"),
            )
            mode_runs = []
            for _ in range(runs_per_mode):
                _, dispatch_info = self._dispatch_towers(query, mode_cfg, route, **kwargs)
                mode_runs.append(float(dispatch_info["elapsed"]))
            timings[mode] = {
                "runs": mode_runs,
                "mean": sum(mode_runs) / len(mode_runs),
            }

        if active_count <= 1:
            fastest_mode = "sequential"
        else:
            fastest_mode = min(timings, key=lambda mode: timings[mode]["mean"])

        if update_config:
            self.config.dispatch_mode = fastest_mode
            self.config.enable_parallel = fastest_mode == "parallel"

        return {
            "fastest_mode": fastest_mode,
            "active_towers": route.active_towers,
            "timings": timings,
            "updated_config": update_config,
        }

    def _execute_pipeline(
        self,
        query: str,
        cfg: TripleTowerConfig,
        top_k: Optional[int] = None,
        rerank_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        started = time.time()
        category = kwargs.pop("category", kwargs.pop("query_category", None))
        dataset = kwargs.pop("dataset", None)
        query_timestamp = float(kwargs.pop("query_timestamp", 0.0) or 0.0)
        declared = kwargs.pop("constraints", None)
        constraint_ctx = self._resolve_constraints(query, declared, query_timestamp or None)
        route = self._route_towers(query, cfg, category=category, dataset=dataset)
        route = self._apply_constraint_route(route, cfg, constraint_ctx)
        self._ensure_sync_rerank_allowed_for_route(
            cfg,
            route,
            context="TripleTowerRetriever._execute_pipeline",
        )
        if self._constraint_short_circuits(route, constraint_ctx):
            return self._empty_payload(route, constraint_ctx, cfg, time.time() - started)
        tower_results, dispatch_info = self._dispatch_towers(
            query,
            cfg,
            route,
            candidate_uids=constraint_ctx.candidate_uids,
            constraint_space_names=constraint_ctx.space_names,
            **kwargs,
        )
        candidates, candidate_units = self._build_candidates(tower_results)
        selected, context, prune_info = self._prune_candidates(
            candidates=candidates,
            candidate_units=candidate_units,
            cfg=cfg,
            route=route,
            query_timestamp=query_timestamp,
        )
        final_top_k = top_k or route.final_top_k or cfg.final_top_k
        reranked = self._unified_rerank(
            query=query,
            results=selected,
            cfg=cfg,
            top_k=final_top_k,
            rerank_params=rerank_params,
        )
        quantification = self._build_quantification(
            route=route,
            dispatch_info=dispatch_info,
            prune_info=prune_info,
            tower_results=tower_results,
            candidate_count=len(candidates),
            selected_count=len(selected),
            final_count=len(reranked),
            elapsed=time.time() - started,
        )
        quantification["constraint"] = constraint_ctx.info
        return {
            "results": reranked,
            "context": context,
            "quantification": quantification,
            "tower_results": tower_results,
            "rerank_result": self._build_rerank_result(tower_results, selected, reranked, cfg),
        }

    async def _execute_pipeline_async(
        self,
        query: str,
        cfg: TripleTowerConfig,
        top_k: Optional[int] = None,
        rerank_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        started = time.time()
        category = kwargs.pop("category", kwargs.pop("query_category", None))
        dataset = kwargs.pop("dataset", None)
        query_timestamp = float(kwargs.pop("query_timestamp", 0.0) or 0.0)
        declared = kwargs.pop("constraints", None)
        constraint_ctx = self._resolve_constraints(query, declared, query_timestamp or None)
        route = self._route_towers(query, cfg, category=category, dataset=dataset)
        route = self._apply_constraint_route(route, cfg, constraint_ctx)
        if self._constraint_short_circuits(route, constraint_ctx):
            return self._empty_payload(route, constraint_ctx, cfg, time.time() - started)
        tower_results, dispatch_info = await self._dispatch_towers_async(
            query,
            cfg,
            route,
            candidate_uids=constraint_ctx.candidate_uids,
            constraint_space_names=constraint_ctx.space_names,
            **kwargs,
        )
        candidates, candidate_units = self._build_candidates(tower_results)
        selected, context, prune_info = self._prune_candidates(
            candidates=candidates,
            candidate_units=candidate_units,
            cfg=cfg,
            route=route,
            query_timestamp=query_timestamp,
        )
        final_top_k = top_k or route.final_top_k or cfg.final_top_k
        reranked = await self._unified_rerank_async(
            query=query,
            results=selected,
            cfg=cfg,
            top_k=final_top_k,
            rerank_params=rerank_params,
        )
        quantification = self._build_quantification(
            route=route,
            dispatch_info=dispatch_info,
            prune_info=prune_info,
            tower_results=tower_results,
            candidate_count=len(candidates),
            selected_count=len(selected),
            final_count=len(reranked),
            elapsed=time.time() - started,
        )
        quantification["constraint"] = constraint_ctx.info
        return {
            "results": reranked,
            "context": context,
            "quantification": quantification,
            "tower_results": tower_results,
            "rerank_result": self._build_rerank_result(tower_results, selected, reranked, cfg),
        }

    def _route_towers(
        self,
        query: str,
        cfg: TripleTowerConfig,
        category: Optional[Union[str, int]],
        dataset: Optional[str],
    ) -> _RouteDecision:
        enable_h, enable_g, enable_e = self._resolve_tower_flags(cfg)
        if not cfg.enable_routing or cfg.search_mode != TripleTowerSearchMode.ALL_TOWERS:
            active = self._active_tower_names(enable_h, enable_g, enable_e)
            return _RouteDecision(
                enable_h, enable_g, enable_e,
                cfg.hierarchical_top_k, cfg.graph_final_topk, cfg.episodic_top_k,
                cfg.final_top_k, active,
                dataset or cfg.router_dataset,
                category,
                1.0,
                "search_mode_override",
            )

        resolved_dataset = self._infer_dataset(dataset or cfg.router_dataset, category)
        try:
            if resolved_dataset == "longmemeval":
                router = LongMemEvalTowerRouter(cfg.router_model_name, cfg.router_strategy)
                raw = router.route(str(query), category=str(category) if category is not None else None,
                                   enable_guidance=cfg.routing_enable_guidance)
                enable_h = bool(raw.enable_sentence)
                enable_e = bool(raw.enable_episodic)
                enable_g = bool(raw.enable_entity)
                topk_h = int(raw.sentence_top_k)
                topk_g = int(raw.entity_top_k)
                topk_e = int(raw.episodic_top_k)
                final_top_k = int(raw.final_top_k)
            else:
                router = LocomoTowerRouter(cfg.router_model_name, cfg.router_strategy)
                locomo_category = int(category) if self._is_int_like(category) else None
                raw = router.route(str(query), category=locomo_category, enable_guidance=cfg.routing_enable_guidance)
                enable_h = bool(raw.enable_hierarchical)
                enable_g = bool(raw.enable_graph)
                enable_e = bool(raw.enable_episodic)
                topk_h = int(raw.topk_hierarchical)
                topk_g = int(raw.topk_similarity)
                topk_e = int(raw.topk_episodic)
                final_top_k = int(raw.final_top_k)
            confidence = 0.9 if category is not None else 0.65
            reason = "category_router" if category is not None else "router_inferred_category"
            active = self._active_tower_names(enable_h, enable_g, enable_e)
            return _RouteDecision(
                enable_h, enable_g, enable_e,
                topk_h or cfg.hierarchical_top_k,
                topk_g or cfg.graph_final_topk,
                topk_e or cfg.episodic_top_k,
                final_top_k or cfg.final_top_k,
                active,
                resolved_dataset,
                category,
                confidence,
                reason,
                raw,
            )
        except Exception as exc:
            logger.warning("Tower routing failed, falling back to all towers: %s", exc)
            active = self._active_tower_names(True, True, True)
            return _RouteDecision(
                True, True, True,
                cfg.hierarchical_top_k, cfg.graph_final_topk, cfg.episodic_top_k,
                cfg.final_top_k, active,
                resolved_dataset,
                category,
                0.5,
                "router_fallback_all_towers",
            )

    # ------------------------------------------------------------- constraints
    def _resolve_constraints(
        self,
        query: str,
        declared: Any,
        reference_time: Optional[float],
    ) -> _ConstraintRouteContext:
        """Resolve ``constraints`` into a route context.

        ``constraints`` accepts ``"auto"`` (infer from the query), an explicit
        ``QueryConstraints`` (evaluation and debugging), or ``None`` (off).
        """
        info: Dict[str, Any] = {
            "mode": "off",
            "branch": "none",
            "confidence": 0.0,
            "source": "none",
            "dropped": [],
            "selectivity": None,
            "applied": False,
            "spaces": [],
            "skipped_towers": [],
            "budget": None,
        }
        context = _ConstraintRouteContext(info=info)
        if declared is None:
            return context

        if isinstance(declared, str):
            if declared.strip().lower() != "auto":
                raise ValueError(
                    f"constraints must be 'auto', QueryConstraints(...), or None; got {declared!r}."
                )
            inferrer = ConstraintInferrer(self.semantic_graph.semantic_map)
            inferred = inferrer.infer(query, reference_time=reference_time)
            info.update(
                mode="auto",
                confidence=round(inferred.confidence, 4),
                source=inferred.source,
                dropped=list(inferred.dropped),
            )
            constraints = inferred.constraints
            if constraints is None or constraints.is_empty():
                info["gap"] = "no_constraint"
                return context
            if inferred.confidence < inferrer.min_confidence:
                info["gap"] = "low_confidence"
                return context
            max_selectivity = inferrer.max_selectivity
        elif isinstance(declared, QueryConstraints):
            constraints = declared
            info.update(mode="explicit", confidence=1.0, source="declared")
            max_selectivity = DEFAULT_MAX_SELECTIVITY
        else:
            raise TypeError(
                "constraints must be 'auto', QueryConstraints(...), or None; "
                f"got {type(declared).__name__}."
            )

        if constraints.is_empty():
            return context

        info["applied"] = True
        spaces = [str(name) for name in (constraints.space_names or [])]
        context.space_names = spaces
        info["spaces"] = spaces

        only_space = bool(spaces) and not (
            constraints.metadata_filters
            or constraints.time_range
            or constraints.relation
        )
        if only_space:
            addressable = self._addressable_spaces()
            if all(name in addressable for name in spaces):
                context.branch = "space_only"
                info["branch"] = "space_only"
                context.tower_flags = self._tower_flags_for_spaces(spaces)
                return context
            info["gap"] = "space_not_addressable"

        planner = ConstraintAwarePlanner(self.semantic_graph)
        candidate_set = planner.resolve_candidates(constraints)
        if candidate_set is None:
            info.update(applied=False, gap="unresolved")
            return context
        if len(candidate_set) == 0:
            context.branch = "empty"
            info.update(branch="empty", applied=False)
            return context

        selectivity = candidate_set.selectivity
        if selectivity is None:
            total = int(getattr(self.semantic_graph.semantic_map, "total_unit_count", 0) or 0)
            selectivity = len(candidate_set) / total if total else 1.0
        info["selectivity"] = round(float(selectivity), 6)
        if selectivity > max_selectivity:
            context.branch = "too_broad"
            info.update(branch="too_broad", applied=False, gap="too_broad")
            return context

        context.branch = "filtered"
        context.selectivity = float(selectivity)
        context.candidate_uids = sorted(candidate_set.uids)
        info["branch"] = "filtered"
        if spaces:
            context.tower_flags = self._tower_flags_for_spaces(spaces)
        return context

    def _apply_constraint_route(
        self,
        route: _RouteDecision,
        cfg: TripleTowerConfig,
        context: _ConstraintRouteContext,
    ) -> _RouteDecision:
        """Fold a resolved constraint context into the router decision."""
        info = context.info
        if context.branch not in ("space_only", "filtered"):
            return replace(route, constraint_info=info)

        flags = {
            "hierarchical": route.enable_hierarchical,
            "graph": route.enable_graph,
            "episodic": route.enable_episodic,
        }
        if context.tower_flags is not None:
            flags = {name: flags[name] and context.tower_flags[name] for name in flags}

        if context.candidate_uids and context.space_names:
            semantic_map = self.semantic_graph.semantic_map
            for name in ("hierarchical", "graph", "episodic"):
                tower_spaces = self._spaces_for_tower(name, context.space_names)
                if not tower_spaces or not flags[name]:
                    continue
                intersection = semantic_map._get_candidate_uids_set(
                    tower_spaces, context.candidate_uids
                )
                if intersection is not None and len(intersection) == 0:
                    flags[name] = False
                    context.skipped_towers.append(name)

        topk = {
            "hierarchical": route.topk_hierarchical,
            "graph": route.topk_graph,
            "episodic": route.topk_episodic,
        }
        if context.selectivity:
            for name in topk:
                if not flags[name]:
                    continue
                topk[name] = max(
                    1,
                    min(
                        math.ceil(topk[name] / context.selectivity),
                        4 * topk[name],
                    ),
                )
            info["budget"] = {
                "selectivity": round(context.selectivity, 6),
                **{f"topk_{name}": value for name, value in topk.items()},
            }
        info["skipped_towers"] = list(context.skipped_towers)

        return replace(
            route,
            enable_hierarchical=flags["hierarchical"],
            enable_graph=flags["graph"],
            enable_episodic=flags["episodic"],
            topk_hierarchical=topk["hierarchical"],
            topk_graph=topk["graph"],
            topk_episodic=topk["episodic"],
            active_towers=self._active_tower_names(
                flags["hierarchical"], flags["graph"], flags["episodic"]
            ),
            reason=f"{route.reason}+constraint:{context.branch}",
            constraint_info=info,
        )

    @staticmethod
    def _constraint_short_circuits(
        route: _RouteDecision, context: _ConstraintRouteContext
    ) -> bool:
        if context.branch == "empty":
            return True
        return context.branch in ("space_only", "filtered") and not route.active_towers

    def _empty_payload(
        self,
        route: _RouteDecision,
        context: _ConstraintRouteContext,
        cfg: TripleTowerConfig,
        elapsed: float,
    ) -> Dict[str, Any]:
        """Return the full pipeline payload for a query with no candidates."""
        tower_results = self._empty_tower_results()
        prune_info = {
            "stage1_input_count": 0,
            "stage1_output_count": 0,
            "selected_count": 0,
            "error": "constraint_empty",
        }
        quantification = self._build_quantification(
            route=route,
            dispatch_info=self._dispatch_info("skipped", 0.0, tower_results),
            prune_info=prune_info,
            tower_results=tower_results,
            candidate_count=0,
            selected_count=0,
            final_count=0,
            elapsed=elapsed,
        )
        quantification["constraint"] = context.info
        return {
            "results": [],
            "context": "",
            "quantification": quantification,
            "tower_results": tower_results,
            "rerank_result": self._build_rerank_result(tower_results, [], [], cfg),
        }

    def _addressable_spaces(self) -> set:
        mounted = getattr(self.semantic_graph.semantic_map, "memory_spaces", {}) or {}
        return {space.value for space in TowerSpace if space.value in mounted}

    @staticmethod
    def _tower_flags_for_spaces(spaces: List[str]) -> Dict[str, bool]:
        towers = {_TOWER_BY_SPACE_ROOT.get(name.split(":", 1)[0]) for name in spaces}
        return {
            "hierarchical": "hierarchical" in towers,
            "graph": "graph" in towers,
            "episodic": "episodic" in towers,
        }

    @staticmethod
    def _spaces_for_tower(tower: str, space_names: List[str]) -> List[str]:
        root = {
            "hierarchical": "hierarchical_memory",
            "graph": "entity_relation",
            "episodic": "episodic_memory",
        }[tower]
        return [name for name in space_names if name.split(":", 1)[0] == root]

    @staticmethod
    def _tower_search_kwargs(tower: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Select only the kwargs a tower backend understands.

        ``constraints`` and ``constraint_space_names`` are pipeline-level and
        must never reach ``smart_search``; the hierarchical tower already gets
        its methods as ``retrieval_methods``, so ``methods`` is not forwarded.
        """
        search_kwargs: Dict[str, Any] = {}
        if tower != "hierarchical" and kwargs.get("methods") is not None:
            search_kwargs["methods"] = kwargs["methods"]
        candidate_uids = kwargs.get("candidate_uids")
        if candidate_uids is not None:
            search_kwargs["candidate_uids"] = list(candidate_uids)
        return search_kwargs

    def _dispatch_towers(
        self,
        query: str,
        cfg: TripleTowerConfig,
        route: _RouteDecision,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        tower_results = self._empty_tower_results()
        jobs = {}
        if route.enable_hierarchical:
            jobs["hierarchical"] = lambda: self._run_hierarchical_tower(query, cfg, route, **kwargs)
        if route.enable_graph:
            jobs["graph"] = lambda: self._run_graph_tower(query, cfg, route, **kwargs)
        if route.enable_episodic:
            jobs["episodic"] = lambda: self._run_episodic_tower(query, cfg, route, **kwargs)

        mode = self._choose_dispatch_mode(cfg, len(jobs))
        if mode == "parallel" and len(jobs) > 1:
            self._ensure_no_running_loop("TripleTowerRetriever sync parallel dispatch")
            return asyncio.run(self._dispatch_towers_async(query, cfg, route, **kwargs))

        dispatch_started = time.time()
        for name, job in jobs.items():
            try:
                tower_results[name] = job()
            except Exception as exc:
                logger.error("%s tower failed: %s", name, exc)
                tower_results[name].stats.error_message = str(exc)

        return tower_results, self._dispatch_info(mode, time.time() - dispatch_started, tower_results)

    async def _dispatch_towers_async(
        self,
        query: str,
        cfg: TripleTowerConfig,
        route: _RouteDecision,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        tower_results = self._empty_tower_results()
        jobs = {}
        if route.enable_hierarchical:
            jobs["hierarchical"] = self._run_hierarchical_tower_async(query, cfg, route, **kwargs)
        if route.enable_graph:
            jobs["graph"] = self._run_graph_tower_async(query, cfg, route, **kwargs)
        if route.enable_episodic:
            jobs["episodic"] = self._run_episodic_tower_async(query, cfg, route, **kwargs)

        mode = self._choose_async_dispatch_mode(cfg, len(jobs))
        dispatch_started = time.time()
        if mode == "parallel" and len(jobs) > 1:
            names = list(jobs.keys())
            gathered = await asyncio.gather(*(jobs[name] for name in names), return_exceptions=True)
            for name, item in zip(names, gathered):
                if isinstance(item, Exception):
                    logger.error("%s tower async failed: %s", name, item)
                    tower_results[name].stats.error_message = str(item)
                else:
                    tower_results[name] = item
        else:
            for name, coroutine in jobs.items():
                try:
                    tower_results[name] = await coroutine
                except Exception as exc:
                    logger.error("%s tower async failed: %s", name, exc)
                    tower_results[name].stats.error_message = str(exc)

        return tower_results, self._dispatch_info(mode, time.time() - dispatch_started, tower_results)

    def _run_hierarchical_tower(
        self, query: str, cfg: TripleTowerConfig, route: _RouteDecision, **kwargs
    ) -> HierarchicalTowerResult:
        started = time.time()
        result = HierarchicalTowerResult(stats=TowerRetrievalStats(enabled=True))
        methods = kwargs.get("methods", cfg.hierarchical_methods)
        tower_kwargs = self._tower_search_kwargs("hierarchical", kwargs)
        spaces = self._spaces_for_tower(
            "hierarchical", kwargs.get("constraint_space_names") or []
        )
        if spaces:
            tower_kwargs["space_names"] = spaces
        search_result = self.hierarchical_retriever.hierarchical_search(
            query=query,
            config=HierarchicalSearchConfig(
                top_k_per_layer=route.topk_hierarchical,
                fusion_method=cfg.hierarchical_fusion_method,
            ),
            top_k=route.topk_hierarchical,
            retrieval_methods=methods,
            rerank_method=cfg.hierarchical_rerank_method,
            **tower_kwargs,
        )
        result.l0_results = self._retrieval_results_to_tuples(
            search_result.results_by_layer.get(MemoryLevel.OBSERVATION, [])
        )
        result.l1_results = self._retrieval_results_to_tuples(
            search_result.results_by_layer.get(MemoryLevel.SUMMARY, [])
        )
        result.l2_results = self._retrieval_results_to_tuples(
            search_result.results_by_layer.get(MemoryLevel.INSIGHT, [])
        )
        result.context_text = self._format_context("Hierarchical", result.l0_results + result.l1_results + result.l2_results)
        result.stats.success = True
        result.stats.retrieved_count = result.total_count
        result.stats.retrieval_time = time.time() - started
        return result

    async def _run_hierarchical_tower_async(
        self, query: str, cfg: TripleTowerConfig, route: _RouteDecision, **kwargs
    ) -> HierarchicalTowerResult:
        started = time.time()
        result = HierarchicalTowerResult(stats=TowerRetrievalStats(enabled=True))
        methods = kwargs.get("methods", cfg.hierarchical_methods)
        tower_kwargs = self._tower_search_kwargs("hierarchical", kwargs)
        spaces = self._spaces_for_tower(
            "hierarchical", kwargs.get("constraint_space_names") or []
        )
        if spaces:
            tower_kwargs["space_names"] = spaces
        search_result = await self.hierarchical_retriever.hierarchical_search_async(
            query=query,
            config=HierarchicalSearchConfig(
                top_k_per_layer=route.topk_hierarchical,
                fusion_method=cfg.hierarchical_fusion_method,
            ),
            top_k=route.topk_hierarchical,
            retrieval_methods=methods,
            rerank_method=cfg.hierarchical_rerank_method,
            **tower_kwargs,
        )
        result.l0_results = self._retrieval_results_to_tuples(
            search_result.results_by_layer.get(MemoryLevel.OBSERVATION, [])
        )
        result.l1_results = self._retrieval_results_to_tuples(
            search_result.results_by_layer.get(MemoryLevel.SUMMARY, [])
        )
        result.l2_results = self._retrieval_results_to_tuples(
            search_result.results_by_layer.get(MemoryLevel.INSIGHT, [])
        )
        result.context_text = self._format_context("Hierarchical", result.l0_results + result.l1_results + result.l2_results)
        result.stats.success = True
        result.stats.retrieved_count = result.total_count
        result.stats.retrieval_time = time.time() - started
        return result

    def _run_graph_tower(self, query: str, cfg: TripleTowerConfig, route: _RouteDecision, **kwargs) -> GraphTowerResult:
        started = time.time()
        result = GraphTowerResult(stats=TowerRetrievalStats(enabled=True))
        search_kwargs = self._tower_search_kwargs("graph", kwargs)
        result.results = self.graph_retriever.entity_relation_search(
            query=query,
            top_k=route.topk_graph,
            final_topk=route.topk_graph,
            rerank_method=cfg.graph_rerank_method,
            space_names=self._spaces_for_tower(
                "graph", kwargs.get("constraint_space_names") or []
            )
            or [cfg.graph_target_space],
            **search_kwargs,
        )
        result.context_text = self._format_context("Knowledge Graph", result.results)
        result.stats.success = True
        result.stats.retrieved_count = len(result.results)
        result.stats.retrieval_time = time.time() - started
        return result

    async def _run_graph_tower_async(
        self, query: str, cfg: TripleTowerConfig, route: _RouteDecision, **kwargs
    ) -> GraphTowerResult:
        started = time.time()
        result = GraphTowerResult(stats=TowerRetrievalStats(enabled=True))
        search_kwargs = self._tower_search_kwargs("graph", kwargs)
        result.results = await self.graph_retriever.entity_relation_search_async(
            query=query,
            top_k=route.topk_graph,
            final_topk=route.topk_graph,
            rerank_method=cfg.graph_rerank_method,
            space_names=self._spaces_for_tower(
                "graph", kwargs.get("constraint_space_names") or []
            )
            or [cfg.graph_target_space],
            **search_kwargs,
        )
        result.context_text = self._format_context("Knowledge Graph", result.results)
        result.stats.success = True
        result.stats.retrieved_count = len(result.results)
        result.stats.retrieval_time = time.time() - started
        return result

    def _run_episodic_tower(self, query: str, cfg: TripleTowerConfig, route: _RouteDecision, **kwargs) -> EpisodicTowerResult:
        started = time.time()
        result = EpisodicTowerResult(stats=TowerRetrievalStats(enabled=True))
        search_kwargs = self._tower_search_kwargs("episodic", kwargs)
        result.results = self.episodic_retriever.episodic_search(
            query=query,
            top_k=route.topk_episodic,
            rerank_method=cfg.episodic_rerank_method,
            space_names=self._spaces_for_tower(
                "episodic", kwargs.get("constraint_space_names") or []
            )
            or [cfg.episodic_target_space],
            **search_kwargs,
        )
        result.context_with_time = self.episodic_retriever.build_context_with_time_injection(result.results)
        result.context_with_metadata = self.episodic_retriever.build_context_with_metadata(result.results)
        result.stats.success = True
        result.stats.retrieved_count = len(result.results)
        result.stats.retrieval_time = time.time() - started
        return result

    async def _run_episodic_tower_async(
        self, query: str, cfg: TripleTowerConfig, route: _RouteDecision, **kwargs
    ) -> EpisodicTowerResult:
        started = time.time()
        result = EpisodicTowerResult(stats=TowerRetrievalStats(enabled=True))
        search_kwargs = self._tower_search_kwargs("episodic", kwargs)
        result.results = await self.episodic_retriever.episodic_search_async(
            query=query,
            top_k=route.topk_episodic,
            rerank_method=cfg.episodic_rerank_method,
            space_names=self._spaces_for_tower(
                "episodic", kwargs.get("constraint_space_names") or []
            )
            or [cfg.episodic_target_space],
            **search_kwargs,
        )
        result.context_with_time = self.episodic_retriever.build_context_with_time_injection(result.results)
        result.context_with_metadata = self.episodic_retriever.build_context_with_metadata(result.results)
        result.stats.success = True
        result.stats.retrieved_count = len(result.results)
        result.stats.retrieval_time = time.time() - started
        return result

    def _build_candidates(
        self, tower_results: Dict[str, Any]
    ) -> Tuple[List[EnhancedCandidateChunk], Dict[str, Tuple[MemoryUnit, float]]]:
        raw_items: List[Tuple[str, TowerSource, MemoryUnit, float, int]] = []
        for rank, (unit, score) in enumerate(
            tower_results["hierarchical"].l0_results
            + tower_results["hierarchical"].l1_results
            + tower_results["hierarchical"].l2_results,
            1,
        ):
            raw_items.append(("hierarchical", TowerSource.HIERARCHICAL, unit, score, rank))
        for rank, (unit, score) in enumerate(tower_results["graph"].results, 1):
            raw_items.append(("graph", TowerSource.KG, unit, score, rank))
        for rank, (unit, score) in enumerate(tower_results["episodic"].results, 1):
            raw_items.append(("episodic", TowerSource.EPISODIC, unit, score, rank))

        if not raw_items:
            return [], {}

        scores = [float(score) for *_prefix, score, _rank in raw_items]
        min_score, max_score = min(scores), max(scores)
        candidate_units: Dict[str, Tuple[MemoryUnit, float]] = {}
        candidates: List[EnhancedCandidateChunk] = []
        for tower_name, source, unit, score, rank in raw_items:
            normalized_score = 1.0 if max_score == min_score else (float(score) - min_score) / (max_score - min_score)
            chunk_id = f"{tower_name}:{unit.uid}"
            candidate_units[chunk_id] = (unit, float(score))
            candidates.append(
                EnhancedCandidateChunk(
                    chunk_id=chunk_id,
                    text=self._extract_text(unit),
                    ce_score=max(0.0, min(1.0, normalized_score)),
                    rank_dense=rank,
                    rank_splade=rank,
                    rank_bm25=rank,
                    tower_source=source,
                    source_chunk_ids=[unit.uid],
                    timestamp=self._extract_timestamp(unit),
                    entity_ids=self._extract_entity_ids(unit),
                    memory_space=self._infer_memory_space(source),
                    raw_metadata={**(unit.metadata or {}), "uid": unit.uid, "tower": tower_name, "original_score": float(score)},
                )
            )
        return candidates, candidate_units

    def _prune_candidates(
        self,
        candidates: List[EnhancedCandidateChunk],
        candidate_units: Dict[str, Tuple[MemoryUnit, float]],
        cfg: TripleTowerConfig,
        route: _RouteDecision,
        query_timestamp: float,
    ) -> Tuple[List[Tuple[MemoryUnit, float]], str, Dict[str, Any]]:
        if not candidates:
            return [], "", {"stage1_input_count": 0, "selected_count": 0, "error": "empty_candidates"}
        try:
            pruner = create_cascade_pruner(
                max_tokens=cfg.pruner_max_tokens,
                lambda_mmr=cfg.pruner_lambda_mmr,
                mode=cfg.pruner_mode,
                adaptive_dataset=route.dataset,
                use_raw_score=cfg.pruner_use_raw_score,
            )
            pruned = pruner.prune(candidates, query_timestamp=query_timestamp, query_category=route.category)
            selected: List[Tuple[MemoryUnit, float]] = []
            for pruned_chunk in pruned.selected_chunks:
                item = candidate_units.get(pruned_chunk.chunk.chunk_id)
                if item is not None:
                    selected.append(item)
            return selected, pruned.prompt_context, {
                "stage1_input_count": pruned.stage1_input_count,
                "stage1_output_count": pruned.stage1_output_count,
                "stage2_conflicts_found": pruned.stage2_conflicts_found,
                "stage2_dropped_count": pruned.stage2_dropped_count,
                "stage2_output_count": pruned.stage2_output_count,
                "stage3_mmr_iterations": pruned.stage3_mmr_iterations,
                "stage3_diversity_penalties": pruned.stage3_diversity_penalties,
                "total_tokens_used": pruned.total_tokens_used,
                "mode_used": pruned.mode_used.value,
                "selected_count": len(selected),
            }
        except Exception as exc:
            logger.warning("Cascade pruning failed, falling back to score order: %s", exc)
            fallback = sorted(candidate_units.values(), key=lambda item: item[1], reverse=True)[: cfg.final_top_k]
            return fallback, self._format_context("Fallback", fallback), {
                "stage1_input_count": len(candidates),
                "selected_count": len(fallback),
                "error": str(exc),
            }

    def _unified_rerank(
        self,
        query: str,
        results: List[Tuple[MemoryUnit, float]],
        cfg: TripleTowerConfig,
        top_k: int,
        rerank_params: Optional[Dict[str, Any]],
    ) -> List[Tuple[MemoryUnit, float]]:
        if not results:
            return []
        if not cfg.enable_second_stage_rerank or cfg.rerank_strategy == RerankStrategy.NONE:
            return sorted(results, key=lambda item: item[1], reverse=True)[:top_k]
        ScoreFusion.ensure_sync_rerank_allowed(
            cfg.second_stage_rerank_method,
            context="TripleTowerRetriever._unified_rerank",
        )
        try:
            multi = self.semantic_graph.get_multi_retriever()
            if hasattr(multi, "_execute_reranking"):
                return multi._execute_reranking(
                    query=query,
                    base_results=results,
                    rerank_method=cfg.second_stage_rerank_method,
                    rerank_params=rerank_params or {"threshold": cfg.rerank_threshold},
                    top_k=top_k,
                )
        except Exception as exc:
            if ScoreFusion.is_vllm_sync_rerank_error(exc):
                raise
            logger.warning("Unified rerank failed, falling back to score order: %s", exc)
        return sorted(results, key=lambda item: item[1], reverse=True)[:top_k]

    async def _unified_rerank_async(
        self,
        query: str,
        results: List[Tuple[MemoryUnit, float]],
        cfg: TripleTowerConfig,
        top_k: int,
        rerank_params: Optional[Dict[str, Any]],
    ) -> List[Tuple[MemoryUnit, float]]:
        if not results:
            return []
        if not cfg.enable_second_stage_rerank or cfg.rerank_strategy == RerankStrategy.NONE:
            return sorted(results, key=lambda item: item[1], reverse=True)[:top_k]
        try:
            multi = self.semantic_graph.get_multi_retriever()
            if hasattr(multi, "_execute_reranking_async"):
                return await multi._execute_reranking_async(
                    query=query,
                    base_results=results,
                    rerank_method=cfg.second_stage_rerank_method,
                    rerank_params=rerank_params or {"threshold": cfg.rerank_threshold},
                    top_k=top_k,
                )
            if hasattr(multi, "_execute_reranking"):
                ScoreFusion.ensure_sync_rerank_allowed(
                    cfg.second_stage_rerank_method,
                    context="TripleTowerRetriever._unified_rerank_async fallback",
                )
                return await asyncio.to_thread(
                    multi._execute_reranking,
                    query=query,
                    base_results=results,
                    rerank_method=cfg.second_stage_rerank_method,
                    rerank_params=rerank_params or {"threshold": cfg.rerank_threshold},
                    top_k=top_k,
                )
            raise NotImplementedError("MultiRetriever does not provide reranking helpers")
        except Exception as exc:
            if ScoreFusion.is_vllm_sync_rerank_error(exc):
                raise
            logger.warning("Async unified rerank failed, falling back to score order: %s", exc)
        return sorted(results, key=lambda item: item[1], reverse=True)[:top_k]

    def _build_quantification(
        self,
        route: _RouteDecision,
        dispatch_info: Dict[str, Any],
        prune_info: Dict[str, Any],
        tower_results: Dict[str, Any],
        candidate_count: int,
        selected_count: int,
        final_count: int,
        elapsed: float,
    ) -> Dict[str, Any]:
        stage2_input = max(1, int(prune_info.get("stage1_output_count", candidate_count) or candidate_count or 1))
        stage2_dropped = int(prune_info.get("stage2_dropped_count", 0) or 0)
        pruning_consistency = max(0.0, 1.0 - (stage2_dropped / stage2_input))
        active_with_results = sum(
            1
            for name in route.active_towers
            if self._tower_count(tower_results, name) > 0
        )
        route_coverage = active_with_results / max(1, len(route.active_towers))
        return {
            "routing_confidence": round(route.confidence, 4),
            "pruning_consistency": round(pruning_consistency, 4),
            "route_coverage": round(route_coverage, 4),
            "final_score": round((0.45 * route.confidence) + (0.35 * pruning_consistency) + (0.20 * route_coverage), 4),
            "routing": {
                "dataset": route.dataset,
                "category": route.category,
                "active_towers": route.active_towers,
                "reason": route.reason,
            },
            "dispatch": dispatch_info,
            "pruning": prune_info,
            "counts": {
                "candidates": candidate_count,
                "selected_after_pruning": selected_count,
                "final_results": final_count,
                "hierarchical": tower_results["hierarchical"].total_count,
                "graph": len(tower_results["graph"].results),
                "episodic": len(tower_results["episodic"].results),
            },
            "elapsed": elapsed,
        }

    @staticmethod
    def _resolve_tower_flags(cfg: TripleTowerConfig) -> Tuple[bool, bool, bool]:
        mapping = {
            TripleTowerSearchMode.ALL_TOWERS: (True, True, True),
            TripleTowerSearchMode.HIERARCHICAL_ONLY: (True, False, False),
            TripleTowerSearchMode.GRAPH_ONLY: (False, True, False),
            TripleTowerSearchMode.EPISODIC_ONLY: (False, False, True),
            TripleTowerSearchMode.HIERARCHICAL_GRAPH: (True, True, False),
            TripleTowerSearchMode.HIERARCHICAL_EPISODIC: (True, False, True),
            TripleTowerSearchMode.GRAPH_EPISODIC: (False, True, True),
        }
        return mapping[cfg.search_mode]

    @staticmethod
    def _active_tower_names(enable_h: bool, enable_g: bool, enable_e: bool) -> List[str]:
        names = []
        if enable_h:
            names.append("hierarchical")
        if enable_g:
            names.append("graph")
        if enable_e:
            names.append("episodic")
        return names

    @staticmethod
    def _choose_dispatch_mode(cfg: TripleTowerConfig, active_count: int) -> str:
        if active_count <= 1:
            return "sequential"
        if cfg.dispatch_mode == "parallel":
            return "parallel"
        if cfg.dispatch_mode == "sequential":
            return "sequential"
        return "parallel" if cfg.enable_parallel else "sequential"

    @staticmethod
    def _choose_async_dispatch_mode(cfg: TripleTowerConfig, active_count: int) -> str:
        if active_count <= 1:
            return "sequential"
        if cfg.dispatch_mode == "sequential":
            return "sequential"
        return "parallel"

    @staticmethod
    def _empty_tower_results() -> Dict[str, Any]:
        return {
            "hierarchical": HierarchicalTowerResult(),
            "graph": GraphTowerResult(),
            "episodic": EpisodicTowerResult(),
        }

    @staticmethod
    def _dispatch_info(mode: str, elapsed: float, tower_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "mode": mode,
            "elapsed": elapsed,
            "tower_times": {
                "hierarchical": tower_results["hierarchical"].stats.retrieval_time,
                "graph": tower_results["graph"].stats.retrieval_time,
                "episodic": tower_results["episodic"].stats.retrieval_time,
            },
        }

    @staticmethod
    def _ensure_no_running_loop(context: str) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError(f"{context} cannot run inside an active event loop; use the async TripleTowerRetriever API.")

    @staticmethod
    def _route_rerank_methods(cfg: TripleTowerConfig, route: _RouteDecision) -> List[str]:
        methods: List[str] = []
        if route.enable_hierarchical and cfg.hierarchical_rerank_method:
            methods.append(cfg.hierarchical_rerank_method)
        if route.enable_graph and cfg.graph_rerank_method:
            methods.append(cfg.graph_rerank_method)
        if route.enable_episodic and cfg.episodic_rerank_method:
            methods.append(cfg.episodic_rerank_method)
        if cfg.enable_second_stage_rerank and cfg.rerank_strategy != RerankStrategy.NONE and cfg.second_stage_rerank_method:
            methods.append(cfg.second_stage_rerank_method)
        return methods

    def _ensure_sync_rerank_allowed_for_route(
        self,
        cfg: TripleTowerConfig,
        route: _RouteDecision,
        context: str,
    ) -> None:
        for rerank_method in self._route_rerank_methods(cfg, route):
            ScoreFusion.ensure_sync_rerank_allowed(rerank_method, context=context)

    @staticmethod
    def _infer_dataset(dataset: Optional[str], category: Optional[Union[str, int]]) -> str:
        if dataset:
            normalized = str(dataset).strip().lower()
            if normalized in {"longmemeval", "longmem", "lme"}:
                return "longmemeval"
            return "locomo"
        if category is not None and not TripleTowerRetriever._is_int_like(category):
            return "longmemeval"
        return "locomo"

    @staticmethod
    def _is_int_like(value: Any) -> bool:
        try:
            if value is None:
                return False
            int(value)
            return True
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _retrieval_results_to_tuples(results: List[Any]) -> List[Tuple[MemoryUnit, float]]:
        return [(item.unit, float(item.score)) for item in results]

    @staticmethod
    def _extract_text(unit: MemoryUnit) -> str:
        raw = unit.raw_data or {}
        for field in ("text_content", "content", "message", "description", "summary", "fact", "statement"):
            value = raw.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return str(raw)

    @staticmethod
    def _extract_timestamp(unit: MemoryUnit) -> float:
        raw = unit.raw_data or {}
        metadata = unit.metadata or {}
        for field in ("timestamp", "event_time", "time", "date", "created", "created_at"):
            value = metadata.get(field) or raw.get(field)
            if value is None:
                continue
            if isinstance(value, (int, float)):
                return float(value)
            text = str(value).strip()
            try:
                return float(text)
            except ValueError:
                pass
            try:
                return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
        return 0.0

    @staticmethod
    def _extract_entity_ids(unit: MemoryUnit) -> List[str]:
        raw = unit.raw_data or {}
        metadata = unit.metadata or {}
        entity_ids: List[str] = []
        for field in ("entity_ids", "entities", "entity", "subject", "object"):
            value = metadata.get(field) or raw.get(field)
            if isinstance(value, list):
                entity_ids.extend(str(item) for item in value if item)
            elif value:
                entity_ids.append(str(value))
        return list(dict.fromkeys(entity_ids))

    @staticmethod
    def _infer_memory_space(source: TowerSource) -> str:
        if source == TowerSource.KG:
            return TowerSpace.GRAPH_ROOT.value
        if source == TowerSource.EPISODIC:
            return TowerSpace.EPISODIC_ROOT.value
        return TowerSpace.HIERARCHICAL_ROOT.value

    @staticmethod
    def _format_context(title: str, results: List[Tuple[MemoryUnit, float]]) -> str:
        if not results:
            return f"{title}: no results."
        lines = [f"{title} results:"]
        for index, (unit, score) in enumerate(results, 1):
            lines.append(f"{index}. ({float(score):.4f}) {TripleTowerRetriever._extract_text(unit)}")
        return "\n".join(lines)

    @staticmethod
    def _format_smart_search_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        results = payload.get("results") or []
        quantification = payload.get("quantification") or {}
        return {
            "results": results,
            "context": payload.get("context", ""),
            "quantification": quantification,
            "statistics": {
                "final_results_count": len(results),
                "dispatch": quantification.get("dispatch", {}),
                "routing": quantification.get("routing", {}),
            },
        }

    @staticmethod
    def _tower_count(tower_results: Dict[str, Any], tower_name: str) -> int:
        if tower_name == "hierarchical":
            return tower_results["hierarchical"].total_count
        if tower_name == "graph":
            return len(tower_results["graph"].results)
        if tower_name == "episodic":
            return len(tower_results["episodic"].results)
        return 0

    def _build_rerank_result(
        self,
        tower_results: Dict[str, Any],
        selected: List[Tuple[MemoryUnit, float]],
        reranked: List[Tuple[MemoryUnit, float]],
        cfg: TripleTowerConfig,
    ) -> SecondStageRerankResult:
        return SecondStageRerankResult(
            reranked_l0=[item for item in reranked if item[0].uid in {unit.uid for unit, _ in tower_results["hierarchical"].l0_results}],
            reranked_graph=[item for item in reranked if item[0].uid in {unit.uid for unit, _ in tower_results["graph"].results}],
            reranked_episodic=[item for item in reranked if item[0].uid in {unit.uid for unit, _ in tower_results["episodic"].results}],
            strategy_used=cfg.rerank_strategy.value,
            first_stage_counts={
                "selected_after_pruning": len(selected),
                "hierarchical": tower_results["hierarchical"].total_count,
                "graph": len(tower_results["graph"].results),
                "episodic": len(tower_results["episodic"].results),
            },
            final_counts={"total": len(reranked)},
        )

    def _update_stats(self, result: TripleTowerResult) -> None:
        self.stats["total_searches"] += 1
        if result.all_flat_results:
            self.stats["successful_searches"] += 1
        for tower in result.towers_used:
            if tower in self.stats["tower_usage"]:
                self.stats["tower_usage"][tower] += 1
        total = self.stats["total_searches"]
        previous = self.stats["avg_retrieval_time"]
        self.stats["avg_retrieval_time"] = previous + (result.total_retrieval_time - previous) / total

    @staticmethod
    def _copy_config_with(cfg: TripleTowerConfig, **updates: Any) -> TripleTowerConfig:
        data = {field_name: getattr(cfg, field_name) for field_name in cfg.__dataclass_fields__}
        data.update(updates)
        return TripleTowerConfig(**data)

    def get_stats(self) -> Dict[str, Any]:
        return dict(self.stats)

    def update_config(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning("Unknown TripleTowerConfig field: %s", key)


def create_triple_tower_retriever(
    semantic_graph: Any,
    hierarchical_top_k: int = 20,
    graph_topk_similarity: int = 15,
    graph_topk_graph: int = 0,
    episodic_top_k: int = 15,
    rerank_method: str = "baai",
    enable_second_stage_rerank: bool = True,
    rerank_strategy: str = "unified_rerank",
    final_top_k: int = 15,
    reranker_manager: Optional[RerankerManager] = None,
) -> TripleTowerRetriever:
    strategy_map = {
        "tower_separate": RerankStrategy.TOWER_SEPARATE,
        "unified_rerank": RerankStrategy.UNIFIED_RERANK,
        "none": RerankStrategy.NONE,
    }
    config = TripleTowerConfig(
        hierarchical_top_k=hierarchical_top_k,
        graph_topk_similarity=graph_topk_similarity,
        graph_topk_graph=graph_topk_graph,
        episodic_top_k=episodic_top_k,
        hierarchical_rerank_method=rerank_method,
        graph_rerank_method=rerank_method,
        episodic_rerank_method=rerank_method,
        enable_second_stage_rerank=enable_second_stage_rerank,
        second_stage_rerank_method=rerank_method,
        rerank_strategy=strategy_map.get(rerank_strategy, RerankStrategy.UNIFIED_RERANK),
        final_top_k=final_top_k,
    )
    return TripleTowerRetriever(
        semantic_graph=semantic_graph,
        config=config,
        reranker_manager=reranker_manager,
    )
