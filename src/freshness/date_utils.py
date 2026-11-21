"""Date extraction and normalization to UTC, for the strict 24h filter.

Priority when resolving a page's publish time (`resolve_published`):
  1. feed value (RSS pubDate / API field)    -> method "feed"
  2. <meta article:published_time / og / itemprop / name=date ...>   -> "meta"
  3. JSON-LD datePublished (incl. @graph)     -> "jsonld"
  4. <time datetime=...>                      -> "time_tag"
  5. relative text ("2 hours ago")            -> "relative"
  6. fallback heuristic: first_seen_at from the dedup store -> "first_seen"
     (all we can say is "not seen before our first sighting"; never invents an older date)
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import dateparser
from lxml import html as lxml_html

UTC = timezone.utc
_META_KEYS = (
    "article:published_time", "og:article:published_time", "og:published_time", "datepublished",
    "date", "dc.date", "dc.date.issued", "pubdate", "publish-date", "publishdate", "parsely-pub-date",
    "sailthru.date", "article:published",
)
_DP_SETTINGS = {"TIMEZONE": "UTC", "RETURN_AS_TIMEZONE_AWARE": True, "TO_TIMEZONE": "UTC",
                "PREFER_DATES_FROM": "past"}
_RELATIVE_RE = re.compile(r"\b(\d+|an?|one)\s+(second|minute|hour|day|week|month|year)s?\s+ago\b|\b(yesterday|just now|today)\b", re.I)


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def parse_date(value: Any, now: datetime | None = None) -> datetime | None:
    """Anything date-like -> aware UTC datetime, or None. Naive times are assumed UTC.
    Accepts datetime, epoch seconds/ms (int/float/numeric str), ISO-8601, RFC-822, relative phrases."""
    if value is None:
        return None
    now = now or datetime.now(UTC)
    if isinstance(value, datetime):
        return _utc(value)
    if isinstance(value, (int, float)) or (isinstance(value, str) and re.fullmatch(r"\d{9,13}(\.\d+)?", value.strip())):
        v = float(value)
        if v > 1e11:  # milliseconds
            v /= 1000
        try:
            return datetime.fromtimestamp(v, UTC)
        except (OverflowError, OSError, ValueError):
            return None
    s = str(value).strip()
    if not s:
        return None
    try:  # ISO-8601 (handles Z and offsets on 3.11+)
        return _utc(datetime.fromisoformat(s.replace("Z", "+00:00")))
    except ValueError:
        pass
    try:  # RFC-822 / RSS pubDate
        return _utc(parsedate_to_datetime(s))
    except (TypeError, ValueError, IndexError):
        pass
    # relative / free-form: dateparser, anchored at `now`
    dt = dateparser.parse(s, settings={**_DP_SETTINGS, "RELATIVE_BASE": now})
    return _utc(dt) if dt else None


def is_within_hours(dt: datetime | None, hours: float = 24, now: datetime | None = None,
                    future_tolerance: timedelta = timedelta(minutes=10)) -> bool:
    """True if now-hours <= dt <= now+tolerance (small forward skew allowed; far-future = bad data)."""
    if dt is None:
        return False
    now = now or datetime.now(UTC)
    return now - timedelta(hours=hours) <= _utc(dt) <= now + future_tolerance


def to_iso(dt: datetime) -> str:
    return _utc(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- HTML extraction
def _jsonld_dates(tree) -> list[str]:
    out: list[str] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            for k in ("datePublished", "dateCreated", "uploadDate"):
                if isinstance(o.get(k), str):
                    out.append(o[k])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for el in tree.xpath('//script[@type="application/ld+json"]'):
        try:
            walk(json.loads(el.text_content()))
        except (ValueError, TypeError):
            continue
    return out


def extract_from_html(html: str, now: datetime | None = None) -> tuple[datetime | None, str | None]:
    """Best publish datetime found in the HTML and which signal gave it."""
    if not html or not html.strip():
        return None, None
    try:
        tree = lxml_html.fromstring(html)
    except (ValueError, lxml_html.etree.ParserError):
        return None, None
    metas = {}
    for m in tree.xpath("//meta"):
        key = (m.get("property") or m.get("name") or m.get("itemprop") or "").lower()
        if key and m.get("content"):
            metas.setdefault(key, m.get("content"))
    for k in _META_KEYS:
        if k in metas and (dt := parse_date(metas[k], now)):
            return dt, "meta"
    for raw in _jsonld_dates(tree):
        if dt := parse_date(raw, now):
            return dt, "jsonld"
    for t in tree.xpath("//time[@datetime]"):
        if dt := parse_date(t.get("datetime"), now):
            return dt, "time_tag"
    for t in tree.xpath("//time"):
        txt = t.text_content().strip()
        if _RELATIVE_RE.search(txt) and (dt := parse_date(txt, now)):
            return dt, "relative"
    return None, None


def resolve_published(
    feed_value: Any = None,
    html: str | None = None,
    first_seen_at: datetime | None = None,
    now: datetime | None = None,
) -> tuple[datetime | None, str]:
    """Normalized UTC publish time + the method used. Method 'none' if nothing usable."""
    if dt := parse_date(feed_value, now):
        return dt, "feed"
    if html:
        dt, method = extract_from_html(html, now)
        if dt:
            return dt, method or "html"
    if first_seen_at:
        return _utc(first_seen_at), "first_seen"
    return None, "none"
