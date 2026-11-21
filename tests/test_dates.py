from datetime import datetime, timedelta, timezone

import pytest

from src.freshness.date_utils import extract_from_html, is_within_hours, parse_date, resolve_published, to_iso
from src.freshness.dedup import DedupStore, canonical_url, hamming, simhash, url_hash
from src.storage.db import Database

NOW = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("val,expected", [
    ("2026-11-21T10:30:00Z", datetime(2026, 11, 21, 10, 30, tzinfo=timezone.utc)),
    ("2026-11-21T10:30:00+05:30", datetime(2026, 11, 21, 5, 0, tzinfo=timezone.utc)),
    ("2026-11-21T10:30:00", datetime(2026, 11, 21, 10, 30, tzinfo=timezone.utc)),  # naive -> UTC
    ("Sat, 21 Nov 2026 10:30:00 GMT", datetime(2026, 11, 21, 10, 30, tzinfo=timezone.utc)),
    ("Sat, 21 Nov 2026 16:00:00 +0530", datetime(2026, 11, 21, 10, 30, tzinfo=timezone.utc)),
    (1795257000, datetime(2026, 11, 21, 10, 30, tzinfo=timezone.utc)),
    (1795257000000, datetime(2026, 11, 21, 10, 30, tzinfo=timezone.utc)),  # ms epoch
    ("1795257000", datetime(2026, 11, 21, 10, 30, tzinfo=timezone.utc)),
    ("2 hours ago", NOW - timedelta(hours=2)),
    ("45 minutes ago", NOW - timedelta(minutes=45)),
    ("1 day ago", NOW - timedelta(days=1)),
])
def test_parse_date(val, expected):
    got = parse_date(val, now=NOW)
    assert got is not None and abs((got - expected).total_seconds()) < 2 and got.utcoffset() == timedelta(0)


@pytest.mark.parametrize("bad", [None, "", "   ", "not a date at all", "soon-ish"])
def test_parse_date_garbage_is_none(bad):
    assert parse_date(bad, now=NOW) is None


def test_24h_window_boundaries():
    assert is_within_hours(NOW - timedelta(hours=23, minutes=59), now=NOW)
    assert not is_within_hours(NOW - timedelta(hours=24, minutes=1), now=NOW)
    assert is_within_hours(NOW + timedelta(minutes=5), now=NOW)       # small clock skew ok
    assert not is_within_hours(NOW + timedelta(days=2), now=NOW)       # bogus future date
    assert not is_within_hours(None, now=NOW)


def test_meta_tag():
    html = '<html><head><meta property="article:published_time" content="2026-11-21T09:00:00-05:00"></head></html>'
    dt, m = extract_from_html(html, NOW)
    assert m == "meta" and to_iso(dt) == "2026-11-21T14:00:00Z"


def test_jsonld_graph():
    html = ('<html><head><script type="application/ld+json">{"@graph":[{"@type":"WebSite"},'
            '{"@type":"NewsArticle","datePublished":"2026-11-21T08:15:00Z"}]}</script></head></html>')
    dt, m = extract_from_html(html, NOW)
    assert m == "jsonld" and to_iso(dt) == "2026-11-21T08:15:00Z"


def test_malformed_jsonld_is_skipped_then_time_tag():
    html = ('<html><script type="application/ld+json">{broken</script>'
            '<body><time datetime="2026-11-21T07:00:00Z">Nov 21</time></body></html>')
    dt, m = extract_from_html(html, NOW)
    assert m == "time_tag" and to_iso(dt) == "2026-11-21T07:00:00Z"


def test_relative_time_text():
    dt, m = extract_from_html("<html><body><time>3 hours ago</time></body></html>", NOW)
    assert m == "relative" and abs((NOW - dt).total_seconds() - 3 * 3600) < 5


def test_no_signals_returns_none_and_first_seen_fallback():
    assert extract_from_html("<html><body>hi</body></html>", NOW) == (None, None)
    assert extract_from_html("", NOW) == (None, None)
    seen = datetime(2026, 11, 21, 6, 0, tzinfo=timezone.utc)
    dt, m = resolve_published(None, "<html></html>", first_seen_at=seen, now=NOW)
    assert (dt, m) == (seen, "first_seen")
    assert resolve_published(None, None, None, NOW) == (None, "none")


def test_feed_value_wins_over_html():
    html = '<meta property="article:published_time" content="2020-01-01T00:00:00Z">'
    dt, m = resolve_published("Sat, 21 Nov 2026 10:30:00 GMT", html, now=NOW)
    assert m == "feed" and dt.year == 2026


# ---- dedup ------------------------------------------------------------------
def test_canonical_url_strips_tracking_and_normalizes():
    a = "http://www.Example.com/post/1/?utm_source=x&b=2&a=1#frag"
    b = "https://example.com/post/1?a=1&b=2"
    assert canonical_url(a) == canonical_url(b) and url_hash(a) == url_hash(b)
    assert url_hash("https://example.com/post/2") != url_hash(b)


def test_simhash_near_duplicates():
    base = " ".join(f"word{i % 50} topic{i % 7} sentence{i}" for i in range(300))
    edited = base.replace("sentence10 ", "sentence-ten ", 1)
    other = " ".join(f"completely{i} different{i % 11} material{i % 13}" for i in range(300))
    assert hamming(simhash(base), simhash(edited)) <= 6
    assert hamming(simhash(base), simhash(other)) > 15


async def test_dedup_store_url_and_content():
    db = Database("sqlite:///:memory:")
    store = DedupStore(db)
    body = " ".join(f"alpha{i} beta{i % 9} gamma{i % 5}" for i in range(200))
    s1, first, _ = await store.check_and_add("https://a.com/x?utm_source=t", body)
    assert s1 == "new"
    s2, first2, _ = await store.check_and_add("https://www.a.com/x", body)
    assert s2 == "dup_url" and first2 == first
    s3, _, dup_of = await store.check_and_add("https://syndicated.com/copy", body + " tiny")
    assert s3 == "dup_content" and dup_of == "https://a.com/x?utm_source=t"
    s4, _, _ = await store.check_and_add("https://b.com/other", " ".join(f"zzz{i} qqq{i % 3}" for i in range(200)))
    assert s4 == "new"


async def test_dedup_reconcile_drops_entries_without_stored_record():
    db = Database("sqlite:///:memory:")
    store = DedupStore(db)
    body = " ".join(f"alpha{i} beta{i % 9}" for i in range(200))
    await store.check_and_add("https://a.com/stored", body)
    await store.check_and_add("https://a.com/crashed", " ".join(f"omega{i} sigma{i % 7}" for i in range(200)))
    n = await store.reconcile({url_hash("https://a.com/stored")})
    assert n == 1
    assert (await store.check_and_add("https://a.com/crashed", "z " * 300))[0] == "new"
    assert (await store.check_and_add("https://a.com/stored", body))[0] == "dup_url"
