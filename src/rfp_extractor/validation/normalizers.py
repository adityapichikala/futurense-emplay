"""Value normalization: dates, lists, strings.

Date handling is where naive extractors fail hardest.  This corpus alone contains
five different date notations for the *same* field:

    27-JUN-2024 14:00:00        (Ariba PDF header)
    July 9, 2024 at 2:00 PM CST (Addendum 2 prose)
    07/09/2024 03:00 PM EDT      (BidNet portal)
    06/10/2024                   (PORFP form field)
    2024-07-09                   (target output)

All of them collapse to one ISO-8601 instant, which is what makes cross-document
conflict detection possible at all: 2:00 PM CST and 3:00 PM EDT are the same
moment and must not be reported as a conflict.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

#: North-American timezone abbreviations seen in procurement documents.
#: Mapped to (standard offset, DST offset, IANA zone).  Writers routinely use
#: "CST" for a July date that is really CDT, so we resolve the offset *for the
#: date in question* rather than trusting the abbreviation's literal offset.
_TZ_ABBR: dict[str, tuple[int, int, str]] = {
    "UTC": (0, 0, "UTC"),
    "GMT": (0, 0, "UTC"),
    "EST": (-5, -4, "America/New_York"),
    "EDT": (-5, -4, "America/New_York"),
    "CST": (-6, -5, "America/Chicago"),
    "CDT": (-6, -5, "America/Chicago"),
    "MST": (-7, -6, "America/Denver"),
    "MDT": (-7, -6, "America/Denver"),
    "PST": (-8, -7, "America/Los_Angeles"),
    "PDT": (-8, -7, "America/Los_Angeles"),
    "AKST": (-9, -8, "America/Anchorage"),
    "AKDT": (-9, -8, "America/Anchorage"),
    "HST": (-10, -10, "Pacific/Honolulu"),
    "AST": (-4, -4, "America/Puerto_Rico"),
}

_TZ_SUFFIX_RE = re.compile(r"\s*\b([A-Z]{2,4})\s*$")
#: "2024at 2:00PM" - PDFs lose the space around 'at'.
_AT_RE = re.compile(r"(?<=[0-9])\s*at\s+", re.I)

#: "July 9, 2024 at 2:00PM" (no space before meridiem, as in Addendum 1)
_MERIDIEM_GLUE_RE = re.compile(r"(\d{1,2}:\d{2})\s*(AM|PM)", re.I)

_FALLBACK_FORMATS: tuple[str, ...] = (
    "%d-%b-%Y %H:%M:%S",
    "%d-%b-%Y",
    "%b %d, %Y %I:%M %p",
    "%b %d, %Y",
    "%B %d, %Y at %I:%M %p",
    "%B %d, %Y %I:%M %p",
    "%B %d, %Y",
    "%m/%d/%Y %I:%M %p",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)


def _us_dst_active(dt: datetime) -> bool:
    """US DST rule: 2nd Sunday of March 02:00 -> 1st Sunday of November 02:00.

    Used only when the ``tzdata`` package is unavailable (common on Windows).
    """
    year = dt.year
    # 2nd Sunday of March
    march = datetime(year, 3, 1)
    start = march + timedelta(days=(6 - march.weekday()) % 7 + 7)
    # 1st Sunday of November
    november = datetime(year, 11, 1)
    end = november + timedelta(days=(6 - november.weekday()) % 7)
    return start <= dt.replace(tzinfo=None) < end


def _resolve_offset(abbr: str, dt: datetime) -> timezone | None:
    """Offset for ``abbr`` at the given wall-clock time, DST-aware."""
    entry = _TZ_ABBR.get(abbr.upper())
    if entry is None:
        return None
    std_offset, dst_offset, zone_name = entry
    if std_offset == dst_offset:
        return timezone(timedelta(hours=std_offset))

    try:  # preferred: real tz database
        from zoneinfo import ZoneInfo

        return dt.replace(tzinfo=ZoneInfo(zone_name)).tzinfo  # type: ignore[return-value]
    except Exception:  # noqa: BLE001 - no tzdata installed (Windows)
        offset = dst_offset if _us_dst_active(dt) else std_offset
        return timezone(timedelta(hours=offset))


class NormalizedDate:
    """Result of a successful date parse."""

    __slots__ = ("dt", "tz_assumed")

    def __init__(self, dt: datetime, tz_assumed: bool) -> None:
        self.dt = dt
        self.tz_assumed = tz_assumed

    def iso(self) -> str:
        return self.dt.isoformat()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"NormalizedDate({self.dt.isoformat()}, assumed_tz={self.tz_assumed})"


def parse_datetime(
    raw: str,
    *,
    default_tz: timezone | None = None,
) -> NormalizedDate | None:
    """Parse a procurement date string into an aware :class:`datetime`.

    Returns ``None`` when the text is not a date, so callers can fall through to
    string handling instead of raising.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    # 1. already ISO?
    iso_candidate = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso_candidate)
        return NormalizedDate(dt, tz_assumed=False)
    except ValueError:
        pass

    # 2. split off a trailing timezone abbreviation
    abbr: str | None = None
    m = _TZ_SUFFIX_RE.search(text)
    if m and m.group(1).upper() in _TZ_ABBR:
        abbr = m.group(1).upper()
        text = text[: m.start()].strip()

    # 3. tidy known quirks: "at 2:00PM" -> "at 2:00 PM", "2:00PM" -> "2:00 PM"
    text = _MERIDIEM_GLUE_RE.sub(r"\1 \2", text)
    text = _AT_RE.sub(" ", text)
    text = " ".join(text.split())

    parsed: datetime | None = None
    for fmt in _FALLBACK_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
            break
        except ValueError:
            continue
    if parsed is None:
        return None

    if abbr is not None:
        tz = _resolve_offset(abbr, parsed)
        return NormalizedDate(parsed.replace(tzinfo=tz), tz_assumed=False)
    if default_tz is not None:
        return NormalizedDate(parsed.replace(tzinfo=default_tz), tz_assumed=True)
    return NormalizedDate(parsed, tz_assumed=False)


def normalize_datetime(value: Any, *, default_tz: timezone | None = None) -> str | None:
    """Return an ISO-8601 string, or ``None`` if unparseable."""
    result = parse_datetime(value, default_tz=default_tz) if not isinstance(value, datetime) else NormalizedDate(value, False)
    return result.iso() if result else None


def to_iso_instant(value: str) -> str | None:
    """UTC-normalized ISO string, used for comparing values across documents."""
    parsed = parse_datetime(value)
    if parsed is None:
        return None
    dt = parsed.dt
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat()


def to_list(value: Any) -> list[str]:
    """Split a multi-value capture into a cleaned list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = [str(v) for v in value]
    else:
        items = re.split(r"[,;]|\s{2,}|\n", str(value))
    out: list[str] = []
    for item in items:
        cleaned = " ".join(str(item).split()).strip(" .,;")
        if cleaned and cleaned.lower() not in {"n/a", "none", "-", "not stated"}:
            out.append(cleaned)
    return dedupe_preserving_order(out)


def prune_subsumed(items: list[str]) -> list[str]:
    """Drop entries contained in a longer entry ('Latitude 5550' vs 'Dell Latitude 5550').

    Also drops very short tokens that are almost always parse noise.
    """
    survivors = [i for i in items if len(i) >= 4]
    out: list[str] = []
    for item in survivors:
        low = item.lower()
        if any(low != other.lower() and low in other.lower() for other in survivors):
            continue
        out.append(item)
    return out or survivors


def dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


#: BidNet truncates long field bodies with a "See more" UI affordance, which is
#: page furniture rather than solicitation content.
_PORTAL_NOISE_RE = re.compile(r"\s*(?:…|\.\.\.)?\s*See more\s*$", re.I)


def strip_portal_noise(text: str) -> str:
    """Drop trailing portal UI chrome such as "… See more"."""
    return _PORTAL_NOISE_RE.sub("", text).strip()


def clean_text(value: Any, *, max_length: int | None = None) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    text = strip_portal_noise(text)
    if max_length and len(text) > max_length:
        text = text[:max_length].rstrip() + "..."
    return text or None


_LEADING_LABEL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 /#()\-]{1,38}:\s*")


def strip_leading_label(text: str) -> str:
    """Remove a 'Description: ' style prefix left by label+value blocks."""
    return _LEADING_LABEL_RE.sub("", text, count=1).strip()


def sentence_containing(text: str, start: int, end: int, *, max_chars: int = 400) -> str:
    """Extract the sentence around ``[start, end)`` within ``text``."""
    left = max(0, text.rfind(".", 0, start) + 1)
    left = max(left, text.rfind("\n", 0, start) + 1)
    right_candidates = [p for p in (text.find(".", end), text.find("\n", end)) if p != -1]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    snippet = text[left:right].strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + "..."
    return " ".join(snippet.split())
