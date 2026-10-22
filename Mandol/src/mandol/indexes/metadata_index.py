"""Filterable-field index registry for SemanticMap payloads.

``MetadataIndex`` is the composition root of the index milestone: it routes
decoded query filters to per-field ``BaseIndex`` strategies, keeps those
structures in sync with the L1 unit population through ``add_units`` /
``remove_units``, and versions its lookup cache so mutations invalidate cached
results the same way ``_space_membership_version`` invalidates FAISS filters.

The resolver contract is deliberately all-or-nothing: ``lookup`` returns
``None`` unless *every* field of the filter bundle is registered and servable,
which hands the query back to the linear scan with exactly the P0 semantics.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from ..constraints.candidate_set import CandidateSet
from ..utils.logging_config import create_module_logger
from .base_index import BaseIndex
from .hash_index import HashIndex
from .sorted_index import SortedIndex

if TYPE_CHECKING:
    from ..core.memory_unit import MemoryUnit

logger = create_module_logger("metadata_index")

_CACHE_MISS = object()


def resolve_field_value(unit: "MemoryUnit", field: str) -> Any:
    """Resolve a payload field with the same lookup order as ``filter_memory_units``.

    Attribute lookup comes first (mirroring ``getattr(unit, field, None)``),
    then ``raw_data``, then the ``metadata`` dictionary, so index maintenance
    observes exactly the values the linear resolver compares.
    """
    value = getattr(unit, field, None)
    if value is None:
        value = unit.raw_data.get(field)
    if value is None and isinstance(unit.metadata, dict):
        value = unit.metadata.get(field)
    return value


class MetadataIndex:
    """Registry and router over per-field index strategies.

    Bare field names register equality (hash) indexes; a mapping selects the
    strategy per field (``"hash"`` for ``eq`` / ``in`` / ``ne`` / ``nin``,
    ``"sorted"`` for normalized ordering). Registering a field replaces any
    previous index for it, and ``rebuild`` re-creates every registered strategy
    from scratch — used after bulk restores and at registration time.
    """

    _INDEX_FACTORIES: Dict[str, Callable[[str], BaseIndex]] = {
        "hash": HashIndex,
        "sorted": SortedIndex,
    }

    def __init__(self):
        """Create an empty registry with no registered fields."""
        self._indexes: Dict[str, BaseIndex] = {}
        self._kinds: Dict[str, str] = {}
        self._version = 0
        self._lookup_cache: Dict[Any, Optional[CandidateSet]] = {}

    @property
    def version(self) -> int:
        """Cache-invalidation version; increments on every mutation."""
        return self._version

    @property
    def registered_fields(self) -> Tuple[str, ...]:
        """Names of the fields currently served by an index."""
        return tuple(self._indexes)

    @property
    def field_kinds(self) -> Dict[str, str]:
        """Registered field -> strategy kind, suitable for persistence."""
        return dict(self._kinds)

    def is_registered(self, field: str) -> bool:
        """Return whether ``field`` has a registered index."""
        return field in self._indexes

    def register_fields(
        self, fields: Union[Iterable[str], Mapping[str, str]]
    ) -> None:
        """Register fields, replacing any index previously created for them.

        Args:
            fields: Iterable of field names (all defaulting to the hash
                strategy) or a mapping from field name to strategy kind.

        Raises:
            ValueError: If a mapping names an unsupported strategy kind.
        """
        if isinstance(fields, Mapping):
            items = [(str(name), str(kind)) for name, kind in fields.items()]
        else:
            items = [(str(name), "hash") for name in fields]
        if not items:
            return
        for field, kind in items:
            factory = self._INDEX_FACTORIES.get(kind)
            if factory is None:
                raise ValueError(
                    f"Unsupported index kind {kind!r} for field {field!r}; "
                    f"expected one of {sorted(self._INDEX_FACTORIES)}."
                )
            self._kinds[field] = kind
            self._indexes[field] = factory(field)
            logger.debug("Registered %s index for field %r.", kind, field)
        self._invalidate()

    def rebuild(self, units: Iterable["MemoryUnit"]) -> None:
        """Re-create every registered index from scratch and repopulate it."""
        if not self._kinds:
            return
        self._indexes = {
            field: self._INDEX_FACTORIES[kind](field)
            for field, kind in self._kinds.items()
        }
        self._add_to_indexes(units)
        self._invalidate()

    def add_units(self, units: Iterable["MemoryUnit"]) -> None:
        """Index the given units (replacement semantics per UID)."""
        if not self._indexes:
            return
        self._add_to_indexes(units)
        self._invalidate()

    def remove_units(self, uids: Iterable[str]) -> None:
        """Drop the given UIDs from every registered index."""
        if not self._indexes:
            return
        for uid in uids:
            for index in self._indexes.values():
                index.remove(str(uid))
        self._invalidate()

    def lookup(self, filters: Dict[str, Dict[str, Any]]) -> Optional[CandidateSet]:
        """Resolve a filter bundle into a candidate set, or ``None`` if unservable.

        Returns:
            The intersected candidate set when every field is registered and
            servable; ``None`` tells the resolver to fall back to the scan.
        """
        if not filters:
            return None
        try:
            cache_key: Any = self._freeze(filters)
            hash(cache_key)
        except TypeError:
            cache_key = None
        if cache_key is not None:
            cached = self._lookup_cache.get(cache_key, _CACHE_MISS)
            if cached is not _CACHE_MISS:
                return cached
        resolved = self._compute_lookup(filters)
        if cache_key is not None:
            self._lookup_cache[cache_key] = resolved
        return resolved

    def estimate_selectivity(
        self, filters: Dict[str, Dict[str, Any]]
    ) -> Optional[float]:
        """Return an upper-bound match fraction, or ``None`` when unservable."""
        if not filters:
            return None
        estimates = []
        for field, conditions in filters.items():
            index = self._indexes.get(field)
            if index is None or not conditions or not index.can_serve(conditions):
                return None
            estimate = index.estimate_selectivity(conditions)
            if estimate is not None:
                estimates.append(estimate)
        if not estimates:
            return None
        return min(estimates)

    def _compute_lookup(
        self, filters: Dict[str, Dict[str, Any]]
    ) -> Optional[CandidateSet]:
        """Intersect per-field index lookups; ``None`` if any field is unservable."""
        combined: Optional[CandidateSet] = None
        for field, conditions in filters.items():
            index = self._indexes.get(field)
            if index is None or not conditions or not index.can_serve(conditions):
                return None
            part = index.lookup(conditions)
            combined = part if combined is None else combined.intersect(part)
        return combined

    def _add_to_indexes(self, units: Iterable["MemoryUnit"]) -> None:
        """Feed the given units to every registered index."""
        for unit in units:
            for index in self._indexes.values():
                index.add(unit.uid, resolve_field_value(unit, index.field))

    def _invalidate(self) -> None:
        """Bump the version and drop cached lookups (mirrors the space cache)."""
        self._version += 1
        self._lookup_cache.clear()

    @classmethod
    def _freeze(cls, value: Any) -> Any:
        """Convert a filter bundle into a hashable cache key."""
        if isinstance(value, Mapping):
            return tuple(
                sorted((str(key), cls._freeze(item)) for key, item in value.items())
            )
        if isinstance(value, (list, tuple)):
            return tuple(cls._freeze(item) for item in value)
        if isinstance(value, (set, frozenset)):
            return frozenset(cls._freeze(item) for item in value)
        return value