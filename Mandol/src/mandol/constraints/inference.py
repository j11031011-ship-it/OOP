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

When rules find nothing and an ``llm_client`` is configured, a JSON-only prompt
asks the model for the same leaves. The reply is validated through the same
``InferenceSchema`` guards before it can narrow anything, and a timeout or a
malformed answer fails open (no constraints) so retrieval still runs.
"""

from __future__ import annotations

import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Tuple

from ..core.memory_space_registry import TowerSpace
from ..indexes.timestamp_normalizer import TimestampNormalizer
from .query_constraints import QueryConstraints, TimeRange

DEFAULT_MIN_CONFIDENCE = 0.7
DEFAULT_MAX_SELECTIVITY = 0.8
DEFAULT_LLM_TIMEOUT = 2.0
DEFAULT_LLM_CACHE_SIZE = 512

_SCHEMA_SAMPLE_SIZE = 200
_SECONDS_PER_DAY = 86400.0
_LLM_DEFAULT_CONFIDENCE = 0.75
_LLM_SAMPLE_VALUES = 8
_LLM_MAX_TOKENS = 256

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
        llm_client: Optional[Any] = None,
        llm_timeout: float = DEFAULT_LLM_TIMEOUT,
        llm_cache_size: int = DEFAULT_LLM_CACHE_SIZE,
    ):
        self.semantic_map = semantic_map
        self.min_confidence = float(min_confidence)
        self.max_selectivity = float(max_selectivity)
        self.llm_client = llm_client
        self.llm_timeout = float(llm_timeout)
        self.llm_cache_size = int(llm_cache_size)
        self._normalizer = TimestampNormalizer()
        self._schema_cache: Optional[Tuple[Tuple[Any, ...], InferenceSchema]] = None
        self._uid_cache: Optional[Tuple[Tuple[Any, ...], FrozenSet[str]]] = None
        self._llm_cache: "OrderedDict[Tuple[str, str], InferredConstraints]" = OrderedDict()

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

        if constraints is None and self.llm_client is not None and text.strip():
            llm_result = self._infer_with_llm(text, schema, anchor)
            if llm_result.constraints is not None:
                llm_result.dropped = dropped + llm_result.dropped
                llm_result.evidence = {**evidence, **llm_result.evidence}
                return llm_result
            dropped.extend(llm_result.dropped)

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

    # -------------------------------------------------------------------- llm
    def _infer_with_llm(
        self,
        text: str,
        schema: InferenceSchema,
        anchor: Optional[float],
    ) -> InferredConstraints:
        key = (text, schema.version)
        cached = self._llm_cache.get(key)
        if cached is not None:
            self._llm_cache.move_to_end(key)
            return _copy_inference(cached)

        raw, failure = self._call_llm(self._build_llm_prompt(text, schema, anchor))
        if failure is not None:
            return InferredConstraints(None, 0.0, "rule", [failure], {}, anchor)

        dropped: List[str] = []
        payload = _parse_llm_json(raw, dropped)
        if payload is None:
            return InferredConstraints(None, 0.0, "rule", dropped, {}, anchor)

        constraints, evidence, confidence = self._validate_llm_payload(payload, schema, dropped)
        applied = constraints is not None
        result = InferredConstraints(
            constraints=constraints,
            confidence=confidence if applied else 0.0,
            source="llm" if applied else "rule",
            dropped=dropped,
            evidence=evidence if applied else {},
            reference_time=anchor,
        )
        self._remember_llm(key, result)
        return _copy_inference(result)

    def _remember_llm(self, key: Tuple[str, str], result: InferredConstraints) -> None:
        if self.llm_cache_size <= 0:
            return
        self._llm_cache[key] = result
        self._llm_cache.move_to_end(key)
        while len(self._llm_cache) > self.llm_cache_size:
            self._llm_cache.popitem(last=False)

    def _call_llm(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """Ask the client for one completion, bounded by ``llm_timeout``.

        A daemon thread enforces the wall-clock bound because the OpenAI and
        transformers clients expose no cancellation; an abandoned call dies with
        the process instead of blocking retrieval.
        """
        client = self.llm_client
        if client is None:
            return None, "llm: unavailable"
        outcome: Dict[str, Any] = {}

        def run() -> None:
            try:
                if hasattr(client, "generate_answer"):
                    outcome["reply"] = client.generate_answer(
                        prompt, max_tokens=_LLM_MAX_TOKENS, temperature=0.0
                    )
                else:
                    outcome["reply"] = client.generate(
                        prompt, max_tokens=_LLM_MAX_TOKENS, temperature=0.0
                    )
            except Exception as exc:  # fail open: keep the unfiltered retrieval path
                outcome["error"] = f"llm: {type(exc).__name__}: {exc}"

        worker = threading.Thread(target=run, name="mandol-constraint-llm", daemon=True)
        worker.start()
        worker.join(self.llm_timeout)
        if worker.is_alive():
            return None, f"llm: timeout after {self.llm_timeout:g}s"
        if "error" in outcome:
            return None, str(outcome["error"])
        reply = outcome.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            return None, "llm: empty response"
        return reply, None

    def _build_llm_prompt(
        self,
        text: str,
        schema: InferenceSchema,
        anchor: Optional[float],
    ) -> str:
        field_lines: List[str] = []
        for name, kind in schema.fields.items():
            samples = [
                repr(value)
                for value in schema.values.get(name, ())[:_LLM_SAMPLE_VALUES]
            ]
            line = f"- {name}: {kind}; 可用操作符 {', '.join(sorted(_OPERATORS_BY_KIND[kind]))}"
            if samples:
                line += f"; 已观测值 {', '.join(samples)}"
            field_lines.append(line)
        mounted = sorted(str(name) for name in (getattr(self.semantic_map, "memory_spaces", {}) or {}))
        anchor_text = _iso_utc(anchor) if anchor is not None else "未提供（不要输出相对时间）"
        return (
            "你是检索约束抽取器。只输出一个 JSON 对象，不要输出解释、Markdown 或代码块。\n"
            f"查询：{text}\n"
            "可用字段：\n"
            + ("\n".join(field_lines) if field_lines else "- 无")
            + "\n"
            f"已挂载空间：{', '.join(mounted) if mounted else '无'}\n"
            f"参考时间：{anchor_text}\n"
            '输出格式：{"confidence": 0-1 的小数, '
            '"metadata_filters": {"字段": {"操作符": 值}} 或 null, '
            '"time_range": {"start": epoch 秒, "end": epoch 秒} 或 null, '
            '"space_names": ["空间名"] 或 null}\n'
            "只有当查询明确表达过滤条件时才输出谓词；没有约束时输出 {}。"
        )

    def _validate_llm_payload(
        self,
        payload: Dict[str, Any],
        schema: InferenceSchema,
        dropped: List[str],
    ) -> Tuple[Optional[QueryConstraints], Dict[str, List[str]], float]:
        """Keep only leaves the schema can answer, mirroring the rule guards."""
        evidence: Dict[str, List[str]] = {}
        metadata_filters: Dict[str, Dict[str, Any]] = {}

        raw_filters = payload.get("metadata_filters")
        if isinstance(raw_filters, dict):
            for name, spec in raw_filters.items():
                field = str(name)
                predicate = _coerce_predicate(spec)
                if predicate is None:
                    dropped.append(f"llm: field '{field}': malformed predicate")
                    continue
                operator, value = predicate
                if schema.kind(field) is None:
                    dropped.append(f"llm: field '{field}' not in schema")
                    continue
                if operator in {"in", "nin"} and not isinstance(value, (list, tuple)):
                    dropped.append(f"llm: field '{field}': operator '{operator}' needs a list")
                    continue
                if not schema.allows(field, operator):
                    dropped.append(f"llm: field '{field}': operator '{operator}' not allowed")
                    continue
                if schema.kind(field) != "sorted" and not _values_observed(schema, field, operator, value):
                    dropped.append(f"llm: field '{field}': value {value!r} not observed")
                    continue
                metadata_filters[field] = {operator: list(value) if isinstance(value, tuple) else value}
                evidence.setdefault("attribute", []).append(f"llm:{field}:{operator}")
        elif raw_filters is not None:
            dropped.append("llm: metadata_filters must be an object or null")

        time_range = self._validate_llm_time(payload.get("time_range"), schema, dropped)
        if time_range is not None:
            evidence["time"] = ["llm"]

        spaces: List[str] = []
        raw_spaces = payload.get("space_names")
        mounted = getattr(self.semantic_map, "memory_spaces", {}) or {}
        if isinstance(raw_spaces, list):
            for name in raw_spaces:
                if not isinstance(name, str) or not name.strip():
                    dropped.append("llm: space: invalid name")
                    continue
                if name not in mounted:
                    dropped.append(f"llm: space: {name}: not mounted")
                    continue
                if name not in spaces:
                    spaces.append(name)
            if spaces:
                evidence["space"] = list(spaces)
        elif raw_spaces is not None:
            dropped.append("llm: space_names must be a list or null")

        confidence = _LLM_DEFAULT_CONFIDENCE
        raw_confidence = payload.get("confidence")
        if isinstance(raw_confidence, (int, float)) and not isinstance(raw_confidence, bool):
            confidence = min(1.0, max(0.0, float(raw_confidence)))

        constraints = None
        if metadata_filters or time_range or spaces:
            constraints = QueryConstraints(
                metadata_filters=metadata_filters or None,
                time_range=time_range,
                relation=None,
                space_names=spaces or None,
            )
        return constraints, evidence, confidence

    def _validate_llm_time(
        self,
        raw_time: Any,
        schema: InferenceSchema,
        dropped: List[str],
    ) -> Optional[TimeRange]:
        if raw_time is None:
            return None
        if not isinstance(raw_time, dict):
            dropped.append("llm: time_range must be an object or null")
            return None
        if schema.kind("timestamp") is None:
            dropped.append("llm: time: timestamp field not observed")
            return None

        start = _coerce_epoch(raw_time.get("start"), self._normalizer)
        end = _coerce_epoch(raw_time.get("end"), self._normalizer)
        for name, bound in (("start", start), ("end", end)):
            value = raw_time.get(name)
            if value is not None and bound is None:
                dropped.append(f"llm: time: unparsable {name} {value!r}")
                return None
        if start is None and end is None:
            dropped.append("llm: time: empty range")
            return None
        if start is not None and end is not None and start > end:
            dropped.append("llm: time: start after end")
            return None
        return TimeRange(start=start, end=end)

    # ----------------------------------------------------------------- anchors
    def _reference_epoch(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value) if float(value) > 0 else None
        return self._normalizer.normalize(value)


def _strip_quotes(value: str) -> str:
    return value.strip().strip("\"'“”‘’「」『』")


def _copy_inference(result: InferredConstraints) -> InferredConstraints:
    return InferredConstraints(
        constraints=result.constraints,
        confidence=result.confidence,
        source=result.source,
        dropped=list(result.dropped),
        evidence={key: list(values) for key, values in result.evidence.items()},
        reference_time=result.reference_time,
    )


def _parse_llm_json(reply: str, dropped: List[str]) -> Optional[Dict[str, Any]]:
    body = str(reply or "").strip()
    if body.startswith("```"):
        body = body.strip("`").strip()
        if body[:4].lower() == "json":
            body = body[4:].strip()
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        dropped.append("llm: invalid JSON response")
        return None
    try:
        payload = json.loads(body[start : end + 1])
    except (TypeError, ValueError):
        dropped.append("llm: invalid JSON response")
        return None
    if not isinstance(payload, dict):
        dropped.append("llm: invalid JSON response")
        return None
    return payload


def _coerce_predicate(spec: Any) -> Optional[Tuple[str, Any]]:
    if isinstance(spec, dict):
        if len(spec) != 1:
            return None
        operator, value = next(iter(spec.items()))
        operator = str(operator).strip().lower()
        if not operator or value is None:
            return None
        return operator, value
    if spec is None:
        return None
    return "eq", spec


def _values_observed(
    schema: InferenceSchema,
    name: str,
    operator: str,
    value: Any,
) -> bool:
    if operator in {"eq", "ne"}:
        return schema.has_value(name, value)
    if operator in {"in", "nin"}:
        if not isinstance(value, (list, tuple)) or not value:
            return False
        return all(schema.has_value(name, item) for item in value)
    return True


def _coerce_epoch(value: Any, normalizer: TimestampNormalizer) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if float(value) > 0 else None
    return normalizer.normalize(value)


def _iso_utc(epoch: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).isoformat()


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
