"""Filterable-field indexes backing constrained retrieval (P1).

The package replaces the linear constraint scans of the P0 resolvers with
incrementally maintained per-field structures:

- ``TimestampNormalizer`` maps heterogeneous timestamp payloads (epoch numbers,
  ISO strings, benchmark date strings) onto comparable epoch floats;
- ``BaseIndex`` defines the ``add`` / ``remove`` / ``lookup`` /
  ``estimate_selectivity`` contract shared by every strategy;
- ``HashIndex`` answers equality and membership operators from value buckets;
- ``SortedIndex`` answers ordering operators from a bisect-maintained list;
- ``MetadataIndex`` registers fields, routes filters, tracks the L1 population
  incrementally, and versions its lookup cache.

``SemanticMap`` composes a ``MetadataIndex`` and exposes
``register_filterable_fields``; unregistered fields keep the linear-scan
resolver path, so existing behavior is preserved by construction.
"""

from .base_index import BaseIndex
from .hash_index import HashIndex
from .metadata_index import MetadataIndex, resolve_field_value
from .sorted_index import SortedIndex
from .timestamp_normalizer import TimestampNormalizer

__all__ = [
    "BaseIndex",
    "HashIndex",
    "MetadataIndex",
    "SortedIndex",
    "TimestampNormalizer",
    "resolve_field_value",
]