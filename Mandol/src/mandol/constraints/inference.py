"""Rule-based inference of query constraints.

The inferrer turns a natural-language query into a ``QueryConstraints`` bundle
whose leaves are restricted to fields the store can actually answer. It exists so
that callers no longer have to declare constraints: the retrieval entry asks the
inferrer first and only then decides how to narrow the candidate pool.

Two guards keep a wrong guess from silently emptying the result set:

- every produced field must appear in ``InferenceSchema``, which is the union of
  the registered filterable fields and the payload keys observed on a bounded
  sample of resident units (benchmark corpora register no fields at all, so a
  registered-only whitelist would make inference inert on real data);
- relative time cues ("昨天", "近三天") require an explicit reference instant.
  Without one the predicate is dropped rather than anchored to the wall clock,
  so a benchmark run is reproducible.

Relation constraints are deliberately not produced: resolving "Alice 的朋友"
needs a name-to-UID resolver that does not exist yet, and inventing seed UIDs
would narrow the pool to nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

from ..core.memory_space_registry import TowerSpace
from ..indexes.timestamp_normalizer import TimestampNormalizer
from .query_constraints import QueryConstraints, TimeRange

DEFAULT_MIN_CONFIDENCE = 0.7
DEFAULT_MAX_SELECTIVITY = 0.8

_SCHEMA_SAMPLE_SIZE = 200
_SECONDS_PER_DAY = 86400.0

_OPERATORS_BY_KIND: Dict[str, FrozenSet[str]] = {
    "hash": frozenset({"eq", "ne", "in", "nin"}),
    "sorted": frozenset({"eq", "ne", "in", "nin", "gt", "gte", "lt", "lte"}),
    "string": frozenset({"eq", "ne", "in", "nin", "contain", "not_contain"}),
}

_WEEKDAYS = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}

_SPACE_SYNONYMS: Tuple[Tuple[str, TowerSpace], ...] = (
    ("情境记忆", TowerSpace.EPISODIC_ROOT),
    ("情景记忆", TowerSpace.EPISODIC_ROOT),
    ("对话记录", TowerSpace.EPISODIC_ROOT),
    ("聊天记录", TowerSpace.EPISODIC_ROOT),
    ("对话", TowerSpace.EPISODIC_ROOT),
    ("聊天", TowerSpace.EPISODIC_ROOT),
    ("总结", TowerSpace.HIERARCHICAL_L1),
    ("摘要", TowerSpace.HIERARCHICAL_L1),
    ("洞察", TowerSpace.HIERARCHICAL_L2),
    ("反思", TowerSpace.HIERARCHICAL_L2),
    ("观察记录", TowerSpace.HIERARCHICAL_L0),
    ("原始记录", TowerSpace.HIERARCHICAL_L0),
    ("知识图谱", TowerSpace.GRAPH_ROOT),
    ("实体关系", TowerSpace.GRAPH_ROOT),
    ("分层记忆", TowerSpace.HIERARCHICAL_ROOT),
)

_LOCATIVE = r"(?:里|中|内|里面|当中)"

_RELATIVE_DAYS_RE = re.compile(r"(?:近|最近|过去)\s*(\d+)\s*天")
_YESTERDAY_RE = re.compile(r"昨天")
_LAST_WEEK_DAY_RE = re.compile(r"上周([一二三四五六日天])")
_LAST_WEEK_RE = re.compile(r"上周(?!\s*[一二三四五六日天])")
_ISO_DATE_RE = re.compile(r"(\d{4})\s*[-/]\s*(\d{1,2})\s*[-/]\s*(\d{1,2})")
_CN_DATE_RE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月(?:\s*(\d{1,2})\s*日)?")
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_HASH_ID_RE = re.compile(r"#\s*([A-Za-z0-9_\-]{1,64})")
_ASSIGNED_ID_RE = re.compile(r"编号\s*(?:为|是|:)?\s*([A-Za-z0-9_\-]{1,64})")
_FIELD_EQ_RE = re.compile(r"(?i)\b([A-Za-z_][A-Za-z0-9_]{2,})\b\s*(?:是|为|=)")


@dataclass(frozen=True)
class InferenceSchema:
    """Field vocabulary an inferred predicate may reference.

    Attributes:
        fields: Field name -> operator kind (``hash`` / ``sorted`` / ``string``).
        version: Cache and cache-invalidation token for the sampled vocabulary.
        values: Bounded sample of observed values per field, used to reject
            equality guesses the store has never seen.
    """

    fields: Dict[str, str]
    version: str
    values: Dict[str, Tuple[Any, ...]] = field(default_factory=dict)

    def kind(self, name: str) -> Optional[str]:
        return self.fields.get(name)

    def allows(self, name: str, operator: str) -> bool:
        kind = self.fields.get(name)
        return kind is not None and operator in _OPERATORS_BY_KIND[kind]

    def has_value(self, name: str, value: Any) -> bool:
        observed = self.values.get(name, ())
        if not observed:
            return True
        return value in observed


@dataclass
class InferredConstraints:
    """Inference result with provenance, so a dropped guess stays auditable."""

    constraints: Optional[QueryConstraints]
    confidence: float
    source: str
    dropped: List[str] = field(default_factory=list)
    evidence: Dict[str, List[str]] = field(default_factory=dict)
    reference_time: Optional[float] = None

    @property
    def applied(self) -> bool:
        return self.constraints is not None and not self.constraints.is_empty()


class ConstraintInferrer:
    """Turn a query into ``QueryConstraints`` using rules and a bound schema."""

    def __init__(
        self,
        semantic_map: Any,
        *,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        max_selectivity: float = DEFAULT_MAX_SELECTIVITY,
    ):
        self.semantic_map = semantic_map
        self.min_confidence = float(min_confidence)
        self.max_selectivity = float(max_selectivity)
        self._normalizer = TimestampNormalizer()
        self._schema_cache: Optional[Tuple[Tuple[Any, ...], InferenceSchema]] = None
        self._uid_cache: Optional[Tuple[Tuple[Any, ...], FrozenSet[str]]] = None

    # ------------------------------------------------------------------ schema
    def schema(self) -> InferenceSchema:
        """Return the field vocabulary, rebuilding it when the store changes."""
        registered = getattr(self.semantic_map, "_metadata_index", None)
        kinds = dict(registered.field_kinds) if registered is not None else {}
        index_version = registered.version if registered is not None else 0
        total = int(getattr(self.semantic_map, "total_unit_count", 0) or 0)
        cache_key = (id(self.semantic_map), index_version, total)
        if self._schema_cache and self._schema_cache[0] == cache_key:
            return self._schema_cache[1]

        observed: Dict[str, List[Any]] = {}
        for unit in list(getattr(self.semantic_map, "memory_units", {}).values())[
            :_SCHEMA_SAMPLE_SIZE
        ]:
            payload = getattr(unit, "raw_data", None) or {}
            if isinstance(payload, dict):
                for name, value in payload.items():
                    observed.setdefault(str(name), []).append(value)
            metadata = getattr(unit, "metadata", None)
            if isinstance(metadata, dict):
                for name, value in metadata.items():
                    observed.setdefault(str(name), []).append(value)

        fields: Dict[str, str] = {"uid": "hash"}
        values: Dict[str, Tuple[Any, ...]] = {"uid": ()}
        for name in sorted(set(kinds) | set(observed)):
            if name in kinds:
                fields[name] = kinds[name]
            else:
                fields[name] = self._infer_kind(observed.get(name, ()))
            values[name] = tuple(observed.get(name, ())[:_SCHEMA_SAMPLE_SIZE])

        schema = InferenceSchema(fields=fields, version=f"v1:{index_version}:{total}", values=values)
        self._schema_cache = (cache_key, schema)
        return schema

    def _infer_kind(self, values: Iterable[Any]) -> str:
        samples = [value for value in values if value is not None][:32]
        if not samples:
            return "hash"
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in samples
        ):
            return "sorted"
        if all(isinstance(value, str) for value in samples) and all(
            self._normalizer.normalize(value) is not None for value in samples
        ):
            return "sorted"
        return "string"

    # ------------------------------------------------------------------- infer
    def infer(
        self,
        query: str,
        *,
        reference_time: Any = None,
    ) -> InferredConstraints:
        """Extract constraints from ``query``.

        Args:
            query: Natural-language query text.
            reference_time: Anchor for relative dates; epoch seconds, an ISO
                string, or a ``datetime``. When absent, relative cues are
                dropped instead of silently using the wall clock.
        """
        text = str(query or "")
        schema = self.schema()
        anchor = self._reference_epoch(reference_time)
        dropped: List[str] = []
        evidence: Dict[str, List[str]] = {}
        scores: List[float] = []

        time_range = self._infer_time(text, anchor, schema, dropped, evidence, scores)
        metadata_filters = self._infer_metadata(text, schema, dropped, evidence, scores)
        space_names = self._infer_spaces(text, dropped, evidence, scores)

        if _looks_relational(text):
            dropped.append("relation: no name->uid resolver")

        constraints = None
        if time_range or metadata_filters or space_names:
            constraints = QueryConstraints(
                metadata_filters=metadata_filters or None,
                time_range=time_range,
                relation=None,
                space_names=space_names or None,
            )
        confidence = min(scores) if scores else 0.0
        return InferredConstraints(
            constraints=constraints,
            confidence=confidence,
            source="rule",
            dropped=dropped,
            evidence=evidence,
            reference_time=anchor,
        )

    # -------------------------------------------------------------------- time
    def _infer_time(
        self,
        text: str,
        anchor: Optional[float],
        schema: InferenceSchema,
        dropped: List[str],
        evidence: Dict[str, List[str]],
        scores: List[float],
    ) -> Optional[TimeRange]:
        if not schema.kind("timestamp"):
            if re.search(r"昨天|上周|近\s*\d+\s*天|\d{4}\s*[-/年]", text):
                dropped.append("time: timestamp field not observed")
            return None

        relative_match = None
        confidence = 0.8
        window: Optional[Tuple[float, float]] = None

        match = _RELATIVE_DAYS_RE.search(text)
        if match:
            relative_match = match.group(0)
            days = int(match.group(1))
            window = (anchor - days * _SECONDS_PER_DAY, anchor) if anchor is not None else None
        elif _YESTERDAY_RE.search(text):
            relative_match = _YESTERDAY_RE.search(text).group(0)
            confidence = 0.85
            window = self._day_window(anchor - _SECONDS_PER_DAY) if anchor is not None else None
        else:
            match = _LAST_WEEK_DAY_RE.search(text)
            if match:
                relative_match = match.group(0)
                window = self._last_week_day_window(anchor, _WEEKDAYS[match.group(1)]) if anchor is not None else None
            else:
                match = _LAST_WEEK_RE.search(text)
                if match:
                    relative_match = match.group(0)
                    if anchor is not None:
                        monday = _day_start(anchor) - weekdays(anchor) * _SECONDS_PER_DAY
                        window = (monday - 7 * _SECONDS_PER_DAY, monday)

        if relative_match is not None:
            if window is None:
                dropped.append("time: no reference_time for relative cue")
                return None
            evidence["time"] = [relative_match]
            scores.append(confidence)
            return TimeRange(start=window[0], end=window[1])

        for pattern, confidence, whole_month in (
            (_ISO_DATE_RE, 0.95, False),
            (_CN_DATE_RE, 0.95, True),
        ):
            match = pattern.search(text)
            if not match:
                continue
            year, month = int(match.group(1)), int(match.group(2))
            day_text = match.group(3) if match.re.groups >= 3 else None
            day = int(day_text) if day_text else None
            try:
                if day is None and whole_month:
                    start = _epoch(year, month, 1)
                    next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
                    end = _epoch(next_year, next_month, 1) - 1
                else:
                    start = _epoch(year, month, day or 1)
                    end = start + _SECONDS_PER_DAY - 1
            except ValueError:
                dropped.append(f"time: unparsable date {match.group(0)!r}")
                return None
            evidence["time"] = [match.group(0)]
            scores.append(confidence)
            return TimeRange(start=start, end=end)
        return None

    @staticmethod
    def _day_window(day_echo: Optional[float]) -> Optional[Tuple[float, float]]:
        if day_echo is None:
            return None
        start = _day_start(day_echo)
        return (start, start + _SECONDS_PER_DAY - 1)

    @staticmethod
    def _last_week_day_window(anchor: Optional[float], weekday: int) -> Optional[Tuple[float, float]]:
        if anchor is None:
            return None
        monday_this_week = _day_start(anchor) - weekdays(anchor) * _SECONDS_PER_DAY
        start = monday_this_week - 7 * _SECONDS_PER_DAY + weekday * _SECONDS_PER_DAY
        return (start, start + _SECONDS_PER_DAY - 1)

    # ---------------------------------------------------------------- metadata
    def _infer_metadata(
        self,
        text: str,
        schema: InferenceSchema,
        dropped: List[str],
        evidence: Dict[str, List[str]],
        scores: List[float],
    ) -> Dict[str, Dict[str, Any]]:
        filters: Dict[str, Dict[str, Any]] = {}

        identifiers = self._infer_identifiers(text, schema, dropped, evidence, scores)
        if identifiers:
            filters["uid"] = {"in": identifiers}

        for name, kind in schema.fields.items():
            if name == "uid" or len(name) < 3:
                continue
            pattern = re.compile(
                rf"(?i)\b{re.escape(name)}\b\s*(?:是|为|=|:|：)\s*([^\s，。；;？?、]+)"
            )
            match = pattern.search(text)
            if not match:
                continue
            value = _strip_quotes(match.group(1))
            if not value or not schema.allows(name, "eq"):
                continue
            if kind != "sorted" and not schema.has_value(name, value):
                dropped.append(f"field: {name}: value {value!r} not observed")
                continue
            prefix = text[max(0, match.start() - 6) : match.start()]
            explicit = any(token in prefix for token in ("只", "仅", "必须", "只要"))
            filters[name] = {"eq": value}
            evidence.setdefault("attribute", []).append(match.group(0))
            scores.append(0.9 if explicit else 0.8)
            break

        for match in _FIELD_EQ_RE.finditer(text):
            name = match.group(1)
            if name.lower() != "uid" and name not in schema.fields:
                dropped.append(f"field: {name}: not in schema")
                break
        return filters

    def _infer_identifiers(
        self,
        text: str,
        schema: InferenceSchema,
        dropped: List[str],
        evidence: Dict[str, List[str]],
        scores: List[float],
    ) -> List[str]:
        if not schema.kind("uid"):
            return []
        candidates: List[str] = []
        for pattern in (_UUID_RE, _HASH_ID_RE, _ASSIGNED_ID_RE):
            for match in pattern.findall(text):
                token = match if isinstance(match, str) else match[0]
                candidates.append(token)
        if not candidates:
            return []
        known = self._known_uids()
        resolved = [token for token in candidates if token in known]
        if not resolved:
            for token in candidates:
                if f"u{token}" in known:
                    resolved.append(f"u{token}")
        if not resolved:
            dropped.append(f"identifier: {candidates[0]!r} not found in store")
            return []
        evidence["identifier"] = resolved
        scores.append(0.95)
        return list(dict.fromkeys(resolved))

    def _known_uids(self) -> FrozenSet[str]:
        total = int(getattr(self.semantic_map, "total_unit_count", 0) or 0)
        cache_key = (id(self.semantic_map), total)
        if self._uid_cache and self._uid_cache[0] == cache_key:
            return self._uid_cache[1]
        getter = getattr(self.semantic_map, "_all_known_uids", None)
        uids = getter() if callable(getter) else set(getattr(self.semantic_map, "memory_units", {}))
        frozen = frozenset(str(uid) for uid in uids)
        self._uid_cache = (cache_key, frozen)
        return frozen

    # ------------------------------------------------------------------- space
    def _infer_spaces(
        self,
        text: str,
        dropped: List[str],
        evidence: Dict[str, List[str]],
        scores: List[float],
    ) -> List[str]:
        mounted = getattr(self.semantic_map, "memory_spaces", {}) or {}
        found: List[str] = []
        for synonym, space in _SPACE_SYNONYMS:
            if not re.search(re.escape(synonym) + _LOCATIVE, text):
                continue
            name = space.value
            if name not in mounted:
                dropped.append(f"space: {name}: not mounted")
                continue
            if name not in found:
                found.append(name)
        if found:
            evidence["space"] = found
            scores.append(0.8)
        return found

    # ----------------------------------------------------------------- anchors
    def _reference_epoch(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value) if float(value) > 0 else None
        return self._normalizer.normalize(value)


def _strip_quotes(value: str) -> str:
    return value.strip().strip("\"'“”‘’「」『』")


def _day_start(epoch: float) -> float:
    return float(int(epoch // _SECONDS_PER_DAY) * _SECONDS_PER_DAY)


def weekdays(epoch: float) -> int:
    """Return the UTC weekday index (Monday=0) for an epoch timestamp."""
    from datetime import datetime, timezone

    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).weekday()


def _epoch(year: int, month: int, day: int) -> float:
    from datetime import datetime, timezone

    return datetime(year, month, day, tzinfo=timezone.utc).timestamp()


def _looks_relational(text: str) -> bool:
    return bool(re.search(r"的朋友|的同事|的家人|认识的人|相关的人|的关系人", text))
