"""Stage-1 MAD cutoff must not depend on which pool the candidates came from.

Stage 4 normalizes scores within the merged pool, so shrinking the pool changes
``ce_score`` without changing any tower's native ranking. The per-tower raw-score
mode is expected to keep the same per-tower decisions in both pools, while the
default global mode is expected to flip at least one of them (which is what
makes this test discriminating rather than vacuous).
"""

from __future__ import annotations

from mandol.quantification.cascade_pruner import (
    CascadeConfidencePruner,
    EnhancedCandidateChunk,
    TowerSource,
)


def _chunk(chunk_id: str, tower: TowerSource, raw: float, ce: float) -> EnhancedCandidateChunk:
    return EnhancedCandidateChunk(
        chunk_id=chunk_id,
        text=chunk_id,
        ce_score=ce,
        rank_dense=1,
        rank_splade=1,
        rank_bm25=1,
        tower_source=tower,
        raw_metadata={"original_score": raw},
    )


def _survivors(pruner: CascadeConfidencePruner, candidates) -> set:
    return {chunk.chunk_id for _, chunk in pruner._stage1_hard_filter(candidates)}


FULL_POOL = [
    _chunk("a1", TowerSource.HIERARCHICAL, 0.95, 1.0),
    _chunk("a2", TowerSource.HIERARCHICAL, 0.90, 0.923),
    _chunk("a3", TowerSource.HIERARCHICAL, 0.70, 0.615),
    _chunk("b1", TowerSource.EPISODIC, 0.80, 0.769),
    _chunk("b2", TowerSource.EPISODIC, 0.75, 0.692),
    _chunk("b3", TowerSource.EPISODIC, 0.60, 0.462),
    _chunk("b4", TowerSource.EPISODIC, 0.30, 0.0),
]

# The narrow pool drops three items the full pool already discards and
# renormalizes the remaining ce scores as if they had been the whole pool.
NARROW_POOL = [
    _chunk("a1", TowerSource.HIERARCHICAL, 0.95, 1.0),
    _chunk("a2", TowerSource.HIERARCHICAL, 0.90, 0.75),
    _chunk("b1", TowerSource.EPISODIC, 0.80, 0.25),
    _chunk("b2", TowerSource.EPISODIC, 0.75, 0.0),
]


def _pruner(use_raw_score: bool) -> CascadeConfidencePruner:
    return CascadeConfidencePruner(mad_multiplier=1.0, use_raw_score=use_raw_score)


def test_raw_score_mode_keeps_common_items_stable_across_pools():
    full = _survivors(_pruner(True), FULL_POOL)
    narrow = _survivors(_pruner(True), NARROW_POOL)

    common = {chunk.chunk_id for chunk in NARROW_POOL}
    assert full & common == {chunk.chunk_id for chunk in NARROW_POOL}
    assert narrow == full & common


def test_global_mode_flips_at_least_one_decision_across_pools():
    full = _survivors(_pruner(False), FULL_POOL)
    narrow = _survivors(_pruner(False), NARROW_POOL)

    common = {chunk.chunk_id for chunk in NARROW_POOL}
    flipped = {chunk_id for chunk_id in common if (chunk_id in narrow) != (chunk_id in full)}
    assert flipped, "the global threshold was expected to be pool-sensitive"


def test_raw_score_mode_uses_tower_native_scale():
    survivors = _survivors(_pruner(True), FULL_POOL)
    # Within its own tower b3 (0.60) survives and b4 (0.30) does not, while the
    # global pool actually keeps a3 (ce 0.615) that its own tower rejects.
    assert "b3" in survivors
    assert "b4" not in survivors
    assert "a3" not in survivors
    assert "a3" in _survivors(_pruner(False), FULL_POOL)


def test_pruner_default_keeps_global_behaviour():
    assert CascadeConfidencePruner().use_raw_score is False
