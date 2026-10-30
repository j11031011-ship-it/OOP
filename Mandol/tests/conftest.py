"""Shared offline fixtures for the constraint and tower test suites.

Every fixture here runs without network access or real model downloads: the
embedding and SPLADE models are deterministic dummies while the FAISS and BM25
indexes are real, matching the offline paradigm of ``test_constrained_retrieval``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mandol.core.memory_space_registry import MemorySpaceRegistry
from mandol.core.memory_unit import MemoryUnit
from mandol.core.semantic_graph import SemanticGraph
from mandol.core.semantic_map import SemanticMap
from mandol.triple_retrieval.triple_tower_retriever import (
    RerankStrategy,
    TripleTowerConfig,
    TripleTowerRetriever,
)
from mandol.utils.model_manager import global_model_manager

TOWER_UNITS = [
    ("h0", "alpha standup notes", "2024-01-05T08:00:00", "speaker_a", "hierarchical_memory:L0_Observation"),
    ("h1", "alpha weekly summary", "2024-01-06T09:00:00", "speaker_b", "hierarchical_memory:L1_Summary"),
    ("h2", "alpha insight pattern", "2024-01-07T10:00:00", "speaker_a", "hierarchical_memory:L2_Insight"),
    ("g0", "alpha project entity", "2024-01-08T11:00:00", "speaker_b", "entity_relation:mentions"),
    ("e0", "alpha conversation episode", "2024-01-09T12:00:00", "speaker_a", "episodic_memory"),
    ("e1", "beta conversation episode", "2024-01-10T13:00:00", "speaker_b", "episodic_memory"),
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


@pytest.fixture
def offline_models(monkeypatch):
    """Patch model loading with deterministic dummies for the whole test."""
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
    return None


def build_semantic_graph() -> SemanticGraph:
    semantic_map = SemanticMap(
        embedding_model_name="test/dummy",
        embedding_dim=2,
        use_flash_attention=False,
    )
    return SemanticGraph(semantic_map)


def add_units(graph: SemanticGraph, units=TOWER_UNITS) -> SemanticGraph:
    for uid, text, timestamp, speaker, space in units:
        raw_data = {"text_content": text, "speaker": speaker, "timestamp": timestamp}
        graph.semantic_map.add_unit(
            MemoryUnit(uid, raw_data),
            explicit_content_for_embedding=text,
            space_names=[space],
            generate_sparse_embedding=False,
        )
    return graph


@pytest.fixture
def tower_graph_factory(offline_models):
    """Build a graph whose units live in the canonical tower spaces."""

    def create(units=TOWER_UNITS) -> SemanticGraph:
        graph = build_semantic_graph()
        MemorySpaceRegistry.initialize_spaces(graph)
        return add_units(graph, units)

    return create


@pytest.fixture
def tower_retriever_factory(tower_graph_factory):
    """Build a model-free triple-tower retriever plus its graph."""

    def create(graph: SemanticGraph = None, **overrides):
        graph = graph or tower_graph_factory()
        config_kwargs = {
            "enable_routing": False,
            "dispatch_mode": "sequential",
            "enable_second_stage_rerank": False,
            "rerank_strategy": RerankStrategy.NONE,
            "hierarchical_methods": ["bm25"],
            "hierarchical_rerank_method": None,
            "graph_rerank_method": None,
            "episodic_rerank_method": None,
        }
        config_kwargs.update(overrides)
        retriever = TripleTowerRetriever(graph, config=TripleTowerConfig(**config_kwargs))
        return retriever, graph

    return create
