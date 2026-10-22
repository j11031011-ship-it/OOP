"""Timestamp normalization for the sorted value index.

Payload timestamps arrive in heterogeneous shapes (epoch numbers, ISO strings,
human-readable benchmark strings), and comparing them raw is only correct while
every value shares one format. The normalizer converts both indexed values and
query bounds to a single comparable representation — epoch seconds as floats —
so ordering stays meaningful across formats.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Optional, Tuple

from ..utils.logging_config import create_module_logger

logger = create_module_logger("timestamp_normalizer")

# Fallback formats for strings that ``datetime.fromisoformat`` cannot parse.
_DATETIME_FORMATS: Tuple[str, ...] = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
    "%d/%m/%Y %H:%M:%S",
    "%d %B, %Y",
    "%d %b, %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    # LoCoMo-style episodic timestamps, e.g. "1:56 pm on 8 May, 2023".
    "%I:%M %p on %d %B, %Y",
    "%I:%M%p on %d %B, %Y",
    "%I:%M %p on %d %b, %Y",
    "%I:%M%p on %d %b, %Y",
)


class TimestampNormalizer:
    """Normalize heterogeneous timestamp payloads to epoch floats.

    Units whose value is missing, empty, or unparsable normalize to ``None``.
    Ordering indexes skip such units, so they simply never match an ordering
    predicate — the same treatment the linear resolver applies to missing
    values. Naive datetimes are interpreted as UTC so normalization stays
    deterministic across machines.
    """

    def normalize(self, value: Any) -> Optional[float]:
        """Return epoch seconds for ``value``, or ``None`` when unparsable."""
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, datetime):
            return self._to_epoch(value)
        if isinstance(value, date):
            return self._to_epoch(datetime(value.year, value.month, value.day))
        if isinstance(value, str):
            return self._normalize_str(value)
        return None

    @staticmethod
    def _normalize_str(value: str) -> Optional[float]:
        """Normalize one string payload using the ISO-first format ladder."""
        text = value.strip()
        if not text:
            return None

        parsed = TimestampNormalizer._parse_datetime(text)
        if parsed is not None:
            return TimestampNormalizer._to_epoch(parsed)

        try:
            return float(text)
        except ValueError:
            logger.debug("Unparsable timestamp payload dropped from ordering: %r", value)
            return None

    @staticmethod
    def _parse_datetime(text: str) -> Optional[datetime]:
        """Parse ISO 8601 first, then the explicit fallback formats."""
        iso_text = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
        try:
            return datetime.fromisoformat(iso_text)
        except ValueError:
            pass
        for pattern in _DATETIME_FORMATS:
            try:
                return datetime.strptime(text, pattern)
            except ValueError:
                continue
        return None

    @staticmethod
    def _to_epoch(parsed: datetime) -> float:
        """Convert a datetime to epoch seconds, treating naive input as UTC."""
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()