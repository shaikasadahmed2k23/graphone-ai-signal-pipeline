"""Dedup keys: canonical-URL hash + 64-bit content simhash, and a SQLite-backed seen-set that
also records first_seen_at (used as the fallback publish date when a page has no date signals).
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

_TRACKING = re.compile(r"^(utm_.*|fbclid|gclid|mc_cid|mc_eid|ref|ref_src|source|cmpid|igshid|_hsenc|_hsmi)$", re.I)
_WORD = re.compile(r"[a-z0-9]{2,}")


def canonical_url(url: str) -> str:
    p = urlparse(url.strip())
    host = p.netloc.lower().removeprefix("www.")
    q = sorted((k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not _TRACKING.match(k))
    path = re.sub(r"/{2,}", "/", p.path).rstrip("/") or "/"
    scheme = "https" if p.scheme in ("http", "https", "") else p.scheme
    return urlunparse((scheme, host, path, "", urlencode(q), ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode()).hexdigest()[:32]


def simhash(text: str, bits: int = 64) -> int:
    """Charikar simhash over word 3-shingles (robust to small edits / boilerplate changes)."""
    words = _WORD.findall(text.lower())
    shingles = [" ".join(words[i:i + 3]) for i in range(max(1, len(words) - 2))] or words
    v = [0] * bits
    for sh in shingles:
        h = int.from_bytes(hashlib.md5(sh.encode()).digest()[:8], "big")
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    return sum(1 << i for i in range(bits) if v[i] > 0)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class DedupStore:
    """Seen-set keyed by canonical-URL hash, plus near-duplicate content detection.

    Near-dup lookup uses 4 x 16-bit bands: two 64-bit hashes within hamming <= 3 must share at least
    one identical band (pigeonhole), so candidates come from an indexed equality lookup, not a scan.
    """

    def __init__(self, db, max_distance: int = 3) -> None:
        self.db, self.max_distance = db, max_distance
        with db._lock:
            db._conn.executescript(
                """CREATE TABLE IF NOT EXISTS seen (
                       url_hash TEXT PRIMARY KEY, url TEXT, simhash TEXT,
                       b0 INTEGER, b1 INTEGER, b2 INTEGER, b3 INTEGER, first_seen_at TEXT NOT NULL);
                   CREATE INDEX IF NOT EXISTS seen_b0 ON seen(b0);
                   CREATE INDEX IF NOT EXISTS seen_b1 ON seen(b1);
                   CREATE INDEX IF NOT EXISTS seen_b2 ON seen(b2);
                   CREATE INDEX IF NOT EXISTS seen_b3 ON seen(b3);"""
            )

    def _check_and_add(self, url: str, content: str | None) -> tuple[str, str, str | None]:
        """-> (status, first_seen_at, duplicate_of_url). status: new | dup_url | dup_content"""
        uh, now = url_hash(url), datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self.db._lock:
            c = self.db._conn
            row = c.execute("SELECT first_seen_at FROM seen WHERE url_hash=?", (uh,)).fetchone()
            if row:
                return "dup_url", row["first_seen_at"], url
            sh = simhash(content) if content and len(content) > 200 else None
            if sh is not None:
                bands = [(sh >> (16 * i)) & 0xFFFF for i in range(4)]
                cand = c.execute(
                    "SELECT url, simhash, first_seen_at FROM seen WHERE simhash IS NOT NULL AND "
                    "(b0=? OR b1=? OR b2=? OR b3=?)", bands).fetchall()
                for r in cand:
                    if hamming(sh, int(r["simhash"])) <= self.max_distance:
                        return "dup_content", r["first_seen_at"], r["url"]
            b = [(sh >> (16 * i)) & 0xFFFF for i in range(4)] if sh is not None else [None] * 4
            c.execute("INSERT INTO seen VALUES (?,?,?,?,?,?,?,?)",
                      (uh, url, str(sh) if sh is not None else None, *b, now))
            c.commit()
            return "new", now, None

    def _reconcile(self, stored_hashes: set[str]) -> int:
        with self.db._lock:
            rows = self.db._conn.execute("SELECT url_hash FROM seen").fetchall()
            stale = [r["url_hash"] for r in rows if r["url_hash"] not in stored_hashes]
            self.db._conn.executemany("DELETE FROM seen WHERE url_hash=?", [(h,) for h in stale])
            self.db._conn.commit()
        return len(stale)

    async def reconcile(self, stored_hashes: set[str]) -> int:
        """Crash safety: a URL is added to `seen` before its record is persisted. Drop seen-entries that
        have no stored record, so an interrupted run never makes unprocessed articles look like duplicates."""
        return await asyncio.to_thread(self._reconcile, stored_hashes)

    async def check_and_add(self, url: str, content: str | None = None) -> tuple[str, str, str | None]:
        return await asyncio.to_thread(self._check_and_add, url, content)
