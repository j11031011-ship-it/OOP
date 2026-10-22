"""Offline unit tests for the filterable-field indexes (P1).

The suite exercises the index strategies and the registry directly, without
building a ``SemanticMap`` or loading any model: units are plain ``MemoryUnit``
objects and every expectation is derived from the ``filter_memory_units``
operator semantics that the linear resolvers mirror.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mandol.core.memory_unit import MemoryUnit
from mandol.indexes import (
    HashIndex,
    MetadataIndex,
    SortedIndex,
    TimestampNormalizer,
    resolve_field_value,
)

# 2024-01-01T00:00:00Z, 2024-01-05T00:00:00Z, and 2024-01-05T08:00:00Z in epoch seconds.
JAN_1_2024 = 1704067200.0
JAN_5_2024 = 1704412800.0
JAN_5_2024_08 = 1704441600.0


def _unit(uid: str, **raw: object) -> MemoryUnit:
    return MemoryUnit(uid, dict(raw))


def test_normalizer_handles_iso_numbers_and_objects() -> None:
    normalizer = TimestampNormalizer()

    assert normalizer.normalize("2024-01-05T08:00:00") == pytest.approx(JAN_5_2024_08)
    assert normalizer.normalize("2024-01-05T08:00:00Z") == pytest.approx(JAN_5_2024_08)
    assert normalizer.normalize("2024-01-05") == pytest.approx(JAN_5_2024)
    assert normalizer.normalize(JAN_5_2024_08) == pytest.approx(JAN_5_2024_08)
    assert normalizer.normalize("1704441600") == pytest.approx(JAN_5_2024_08)
    assert normalizer.normalize(
        datetime(2024, 1, 5, 8, 0, 0, tzinfo=timezone.utc)
    ) == pytest.approx(JAN_5_2024_08)


def test_normalizer_treats_naive_datetimes_as_utc() -> None:
    normalizer = TimestampNormalizer()

    naive = normalizer.normalize(datetime(2024, 1, 5, 8, 0, 0))
    aware = normalizer.normalize(datetime(2024, 1, 5, 8, 0, 0, tzinfo=timezone.utc))
    assert naive == pytest.approx(aware)


def test_normalizer_parses_locomo_style_timestamps() -> None:
    normalizer = TimestampNormalizer()

    parsed = normalizer.normalize("1:56 pm on 8 May, 2023")
    assert parsed is not None
    assert normalizer.normalize("2023-05-08T00:00:00") < parsed
    assert parsed < normalizer.normalize("2023-05-09T00:00:00")


def test_normalizer_returns_none_for_missing_and_unparsable() -> None:
    normalizer = TimestampNormalizer()

    assert normalizer.normalize(None) is None
    assert normalizer.normalize("") is None
    assert normalizer.normalize("   ") is None
    assert normalizer.normalize("not a date") is None
    assert normalizer.normalize(True) is None
    assert normalizer.normalize({"a": 1}) is None


def test_hash_index_equality_and_membership() -> None:
    index = HashIndex("speaker")
    index.add("u1", "speaker_a")
    index.add("u2", "speaker_b")
    index.add("u3", "speaker_a")
    index.add("u4", None)

    assert index.lookup({"eq": "speaker_a"}).uids == {"u1", "u3"}
    assert index.lookup({"in": ["speaker_a", "speaker_b"]}).uids == {"u1", "u2", "u3"}
    assert index.lookup({"ne": "speaker_a"}).uids == {"u2", "u4"}
    assert index.lookup({"nin": ["speaker_a"]}).uids == {"u2", "u4"}
    assert index.lookup({"eq": None}).uids == {"u4"}
    assert index.lookup({"in": ["speaker_z"]}).uids == frozenset()


def test_hash_index_combines_conditions_and_updates() -> None:
    index = HashIndex("speaker")
    index.add("u1", "a")
    index.add("u2", "b")

    assert index.lookup({"ne": "a", "in": ["a", "b"]}).uids == {"u2"}
    assert index.estimate_selectivity({"eq": "b"}) == pytest.approx(0.5)

    index.add("u1", "b")
    assert index.lookup({"eq": "a"}).uids == frozenset()
    assert index.lookup({"eq": "b"}).uids == {"u1", "u2"}

    index.remove("u2")
    assert index.uid_count == 1
    assert index.lookup({"eq": "b"}).uids == {"u1"}


def test_hash_index_projects_container_values() -> None:
    index = HashIndex("entity_ids")
    index.add("u1", ["alice", "bob"])
    index.add("u2", ["bob"])
    index.add("u3", ("alice", "bob"))
    index.add("u4", {"tier": "gold", "tags": ["vip"]})
    index.add("u5", {"tags": ["vip"], "tier": "gold"})
    index.add("u6", {"tier": "silver"})
    index.add("u7", {"alice", "bob"})

    assert index.lookup({"eq": ["alice", "bob"]}).uids == {"u1"}
    assert index.lookup({"eq": ("alice", "bob")}).uids == {"u3"}
    assert index.lookup({"in": [["bob"], ("alice", "bob")]}).uids == {"u2", "u3"}
    assert index.lookup({"eq": {"tags": ["vip"], "tier": "gold"}}).uids == {"u4", "u5"}
    assert index.lookup({"eq": frozenset({"bob", "alice"})}).uids == {"u7"}
    assert index.lookup({"ne": {"tier": "gold", "tags": ["vip"]}}).uids == {
        "u1",
        "u2",
        "u3",
        "u6",
        "u7",
    }


def test_hash_index_can_serve_rejects_unsupported_shapes() -> None:
    index = HashIndex("speaker")
    index.add("u1", "a")
    assert index.can_serve({"eq": "a"})
    assert index.can_serve({"eq": ["projectable"]})
    assert not index.can_serve({})
    assert not index.can_serve({"gt": 1})
    assert not index.can_serve({"in": "string"})


def test_hash_index_falls_back_for_unprojectable_values() -> None:
    class _Opaque:
        __hash__ = None

        def __eq__(self, other):
            return isinstance(other, _Opaque)

    index = HashIndex("payload")
    index.add("u1", "indexable")
    assert index.can_serve({"eq": "indexable"})

    index.add("u2", _Opaque())
    assert not index.can_serve({"eq": "indexable"})


def test_sorted_index_ordering_windows() -> None:
    index = SortedIndex("timestamp")
    index.add("u1", "2024-01-05T08:00:00")
    index.add("u2", "2024-02-10T14:30:00")
    index.add("u3", "2024-03-20T19:00:00")
    index.add("u4", "2024-04-02T07:15:00")

    assert index.lookup({"gte": "2024-02-01T00:00:00"}).uids == {"u2", "u3", "u4"}
    assert index.lookup({"gt": "2024-02-10T14:30:00"}).uids == {"u3", "u4"}
    assert index.lookup({"lt": "2024-02-10T14:30:00"}).uids == {"u1"}
    assert index.lookup(
        {"gte": "2024-01-01T00:00:00", "lte": "2024-03-31T23:59:59"}
    ).uids == {"u1", "u2", "u3"}
    assert index.lookup({"gt": "2025-01-01T00:00:00"}).uids == frozenset()
    assert index.estimate_selectivity({"gte": "2024-04-01T00:00:00"}) == pytest.approx(
        0.25
    )


def test_sorted_index_skips_unparsable_values_and_detects_servability() -> None:
    index = SortedIndex("timestamp")
    index.add("u1", "2024-01-05T08:00:00")
    index.add("u2", None)
    index.add("u3", "garbage")

    assert index.uid_count == 1
    assert index.lookup({"gte": "2024-01-01T00:00:00"}).uids == {"u1"}
    assert not index.can_serve({"eq": "2024-01-01"})
    assert not index.can_serve({"gte": "garbage"})
    assert not index.can_serve({})


def test_sorted_index_mixes_formats_and_updates() -> None:
    index = SortedIndex("timestamp")
    index.add("u1", JAN_1_2024)
    index.add("u2", "2024-01-02T00:00:00")
    assert index.lookup({"gt": JAN_1_2024}).uids == {"u2"}

    index.add("u1", "2024-01-03T00:00:00")
    assert index.lookup({"gt": JAN_1_2024}).uids == {"u1", "u2"}

    index.remove("u1")
    assert index.lookup({"gt": JAN_1_2024}).uids == {"u2"}


def test_metadata_index_registers_routes_and_serves() -> None:
    index = MetadataIndex()
    assert index.registered_fields == ()

    index.register_fields({"speaker": "hash", "timestamp": "sorted"})
    assert set(index.registered_fields) == {"speaker", "timestamp"}
    assert index.is_registered("speaker")

    index.add_units(
        [
            _unit("u1", speaker="a", timestamp="2024-01-05T08:00:00"),
            _unit("u2", speaker="b", timestamp="2024-02-10T14:30:00"),
            _unit("u3", speaker="a", timestamp="2024-03-20T19:00:00"),
        ]
    )

    assert index.lookup({"speaker": {"eq": "a"}}).uids == {"u1", "u3"}
    assert index.lookup(
        {"speaker": {"eq": "a"}, "timestamp": {"gte": "2024-02-01T00:00:00"}}
    ).uids == {"u3"}
    assert index.lookup({"speaker": {"eq": "a"}, "unknown": {"eq": 1}}) is None
    assert index.lookup({"speaker": {"contain": "a"}}) is None
    assert index.lookup({"speaker": {"eq": "a"}, "timestamp": {"eq": "2024"}}) is None
    assert index.lookup({}) is None


def test_metadata_index_serves_container_fields() -> None:
    index = MetadataIndex()
    index.register_fields(["entity_ids"])
    index.add_units(
        [
            _unit("u1", entity_ids=["alice", "bob"]),
            _unit("u2", entity_ids=["bob"]),
        ]
    )

    assert index.lookup({"entity_ids": {"eq": ["alice", "bob"]}}).uids == {"u1"}
    assert index.lookup({"entity_ids": {"in": [["bob"]]}}).uids == {"u2"}


def test_metadata_index_register_fields_accepts_bare_iterables() -> None:
    index = MetadataIndex()
    index.register_fields(["speaker"])
    assert index.is_registered("speaker")

    with pytest.raises(ValueError):
        index.register_fields({"speaker": "btree"})


def test_metadata_index_exposes_field_kinds_for_persistence() -> None:
    index = MetadataIndex()
    index.register_fields({"speaker": "hash", "timestamp": "sorted"})

    kinds = index.field_kinds
    assert kinds == {"speaker": "hash", "timestamp": "sorted"}

    kinds["speaker"] = "sorted"
    assert index.field_kinds["speaker"] == "hash"


def test_metadata_index_cache_tracks_mutations() -> None:
    index = MetadataIndex()
    index.register_fields(["speaker"])
    index.add_units([_unit("u1", speaker="a")])

    first = index.lookup({"speaker": {"eq": "a"}})
    assert index.lookup({"speaker": {"eq": "a"}}) is first

    version = index.version
    index.add_units([_unit("u2", speaker="a")])
    assert index.version > version
    assert index.lookup({"speaker": {"eq": "a"}}).uids == {"u1", "u2"}

    index.remove_units(["u1"])
    assert index.lookup({"speaker": {"eq": "a"}}).uids == {"u2"}


def test_metadata_index_registration_invalidates_cached_misses() -> None:
    index = MetadataIndex()
    assert index.lookup({"speaker": {"eq": "a"}}) is None

    index.register_fields(["speaker"])
    index.add_units([_unit("u1", speaker="a")])
    assert index.lookup({"speaker": {"eq": "a"}}).uids == {"u1"}


def test_metadata_index_rebuild_repopulates_registered_indexes() -> None:
    index = MetadataIndex()
    index.register_fields({"speaker": "hash", "timestamp": "sorted"})
    index.rebuild([_unit("u1", speaker="a", timestamp="2024-01-05T08:00:00")])
    assert index.lookup({"speaker": {"eq": "a"}}).uids == {"u1"}
    assert index.lookup({"timestamp": {"gte": "2024-01-01T00:00:00"}}).uids == {"u1"}

    index.rebuild([_unit("u2", speaker="b", timestamp="2024-02-01T00:00:00")])
    assert index.lookup({"speaker": {"eq": "a"}}).uids == frozenset()
    assert index.lookup({"timestamp": {"lt": "2024-01-31T00:00:00"}}).uids == frozenset()


def test_metadata_index_selectivity_is_servability_gated() -> None:
    index = MetadataIndex()
    index.register_fields(["speaker"])
    index.add_units([_unit("u1", speaker="a"), _unit("u2", speaker="b")])

    assert index.estimate_selectivity({"speaker": {"eq": "a"}}) == pytest.approx(0.5)
    assert index.estimate_selectivity({"speaker": {"contain": "a"}}) is None
    assert index.estimate_selectivity({}) is None


def test_metadata_index_mutations_are_noops_without_registration() -> None:
    index = MetadataIndex()
    index.add_units([_unit("u1", speaker="a")])
    index.remove_units(["u1"])

    assert index.version == 0
    assert index.lookup({"speaker": {"eq": "a"}}) is None


def test_resolve_field_value_prefers_attributes_then_raw_data_then_metadata() -> None:
    unit = _unit("u1", speaker="a")
    unit.metadata["project"] = "alpha"

    assert resolve_field_value(unit, "speaker") == "a"
    assert resolve_field_value(unit, "uid") == "u1"
    assert resolve_field_value(unit, "project") == "alpha"
    assert resolve_field_value(unit, "missing") is None

    unit.raw_data["project"] = "raw-wins"
    assert resolve_field_value(unit, "project") == "raw-wins"