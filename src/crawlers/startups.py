"""YC directory via the Algolia index that ycombinator.com/companies itself uses.

* The Algolia search-only key is embedded in the page (window.AlgoliaOpts) and expires, so we
  scrape it at runtime and refresh it on 403 / near expiry.
* Algolia caps any single query at 1000 hits and YC has ~6.4k companies, so we partition by the
  `batch` facet (largest batch is <400) and page inside each partition.
* Checkpoint = list of completed batches; a crash resumes at the first incomplete batch.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote, urlencode

from src.crawlers.base_crawler import BaseCrawler, BlockedError, FetchError
from src.logging_setup import get_logger
from src.metrics import metrics
from src.models import Source, StartupContent, StartupData, StartupRecord, to_payload
from src.storage.db import Database

log = get_logger("startups")

YC_PAGE = "https://www.ycombinator.com/companies"
INDEX = "YCCompany_production"
JOB = "startups:yc"
HITS_PER_PAGE = 1000  # Algolia max


_AI_LABELS = {
    "ai", "artificial intelligence", "machine learning", "generative ai", "nlp", "computer vision", "llm",
    "deep learning", "ai assistant", "conversational ai", "ml", "robotics", "ai enhanced drug discovery",
}
_AI_LABEL_SUBSTR = ("artificial intelligence", "machine learning", "generative ai", "computer vision")
_AI_LABEL_RE = re.compile(r"\bai\b|\bllm\b|\bnlp\b", re.I)
_AI_TEXT = re.compile(
    r"\b(AI|A\.I\.|LLMs?|GPT|machine[- ]learning|deep[- ]learning|generative|neural|computer vision|NLP|"
    r"foundation models?|copilots?|agents?)\b",
    re.I,
)


def is_ai_startup(h: dict) -> bool:
    """AI if a YC tag/industry says so, or the one-liner mentions AI terms (description is too noisy)."""
    labels = [*(h.get("tags") or []), *(h.get("industries") or []), h.get("subindustry") or ""]
    for lab in labels:
        low = lab.lower()
        if low in _AI_LABELS or any(x in low for x in _AI_LABEL_SUBSTR) or _AI_LABEL_RE.search(low):
            return True
    return bool(_AI_TEXT.search(h.get("one_liner") or ""))


class YCStartupsCrawler(BaseCrawler):
    def __init__(self, db: Database, **kw: Any) -> None:
        super().__init__(**kw)
        self.db = db
        self._app = self._key = ""
        self._valid_until = 0

    # ---- Algolia credentials ----------------------------------------------
    async def _refresh_key(self) -> None:
        res = await self.fetch(YC_PAGE)
        m = re.search(r"AlgoliaOpts\s*=\s*(\{.*?\})", res.text)
        if not m:
            raise FetchError("AlgoliaOpts not found on YC page (page layout changed?)")
        opts = json.loads(m.group(1))
        self._app, self._key, self._valid_until = opts["app"], opts["key"], int(opts.get("valid_until", 0))
        log.info("algolia_key_refreshed", valid_until=self._valid_until)

    async def _query(self, params: dict[str, Any], _retried: bool = False) -> dict:
        if not self._key or (self._valid_until and time.time() > self._valid_until - 60):
            await self._refresh_key()
        body = {"requests": [{"indexName": INDEX, "params": urlencode(params)}]}
        try:
            res = await self.fetch(
                f"https://{self._app}-dsn.algolia.net/1/indexes/*/queries",
                method="POST", json=body, escalate=False,
                headers={"X-Algolia-Application-Id": self._app, "X-Algolia-API-Key": self._key},
            )
        except BlockedError:
            if _retried:
                raise
            await self._refresh_key()
            return await self._query(params, _retried=True)
        if not res.ok:
            raise FetchError(f"algolia {res.status}: {res.text[:200]}")
        return res.json()["results"][0]

    # ---- crawl -------------------------------------------------------------
    async def list_batches(self) -> list[str]:
        r = await self._query({"query": "", "hitsPerPage": 0, "facets": json.dumps(["batch"]), "maxValuesPerFacet": 200})
        return sorted(r["facets"]["batch"], key=lambda b: -r["facets"]["batch"][b])  # biggest first

    async def run(self, limit: int | None = None) -> dict[str, int]:
        """Crawl all YC companies (or stop after `limit` records). Returns upsert stats."""
        ckpt = await self.db.get_checkpoint(JOB)
        done: list[str] = ckpt.get("done_batches", [])
        batches = await self.list_batches()
        total = {"inserted": 0, "updated": 0, "unchanged": 0}
        stored = 0
        log.info("yc_start", batches=len(batches), already_done=len(done))
        for batch in batches:
            if batch in done:
                continue
            page = 0
            while True:
                r = await self._query({
                    "query": "", "hitsPerPage": HITS_PER_PAGE, "page": page,
                    "facetFilters": json.dumps([[f"batch:{batch}"]]),
                })
                hits = r["hits"]
                metrics.inc("startups.fetched", len(hits))
                items = []
                for h in hits:
                    rec = self.to_record(h)
                    if rec is None:
                        metrics.inc("startups.parse_errors")
                        continue
                    metrics.inc("startups.parsed")
                    items.append(("STARTUP", f"yc:{h['id']}", to_payload(rec), rec.source.name))
                stats = await self.db.upsert_many(items)
                for k, v in stats.items():
                    total[k] += v
                    if k == "unchanged":
                        metrics.inc("dedup.hits", v)
                stored += len(items)
                page += 1
                if page >= r.get("nbPages", 0) or (limit and stored >= limit):
                    break
            if limit and stored >= limit:
                log.info("yc_limit_reached", stored=stored)
                break
            done.append(batch)  # checkpoint only fully-finished batches
            await self.db.set_checkpoint(JOB, {"done_batches": done})
            log.info("yc_batch_done", batch=batch, n=len(hits), total=stored)
        else:
            await self.db.set_checkpoint(JOB, {"done_batches": done, "complete": True})
        return total

    # ---- mapping -----------------------------------------------------------
    @staticmethod
    def to_record(h: dict) -> StartupRecord | None:
        name = (h.get("name") or "").strip()
        if not name:
            return None
        launched = h.get("launched_at")
        launched_iso = (
            datetime.fromtimestamp(launched, timezone.utc).strftime("%Y-%m-%d") if isinstance(launched, (int, float)) else None
        )
        slug = h.get("slug") or str(h["id"])
        return StartupRecord(
            source=Source(name="YC Directory", url=f"https://www.ycombinator.com/companies/{quote(slug)}"),
            content=StartupContent(
                entityName=name,
                data=StartupData(
                    employeeCount=h.get("team_size") if isinstance(h.get("team_size"), int) else None,
                    website=h.get("website") or None,
                    oneLiner=h.get("one_liner") or None,
                    description=h.get("long_description") or None,
                    batch=h.get("batch"),
                    location=h.get("all_locations") or None,
                    industry=h.get("industry"),
                    subindustry=h.get("subindustry"),
                    industries=h.get("industries") or [],
                    tags=h.get("tags") or [],
                    stage=h.get("stage"),
                    status=h.get("status"),
                    isHiring=bool(h.get("isHiring")),
                    launchedAt=launched_iso,
                    formerNames=h.get("former_names") or [],
                    ycId=h.get("id"),
                    is_ai=is_ai_startup(h),
                ),
            ),
        )
