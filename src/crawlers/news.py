"""News: TechCrunch AI, VentureBeat, The Verge AI, MIT Technology Review AI (RSS) + Hacker News (Algolia API).

* Strict 24h filter on normalized UTC publish time (feed date; HTML meta/JSON-LD/<time> if the feed has none).
* Full text via trafilatura (escalating fetch tiers for bot-protected sites); feed body is the fallback.
* Dedup: canonical-URL hash + simhash of the text (syndicated copies / HN reposts), via freshness.dedup.
* LLM only for structuring (summary + entities), cached by content hash; entities go through the resolver.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from typing import Any

import feedparser
from pydantic import BaseModel, Field

from src.crawlers.base_crawler import BaseCrawler, FetchError
from src.freshness.date_utils import is_within_hours, parse_date, resolve_published, to_iso
from src.freshness.dedup import DedupStore, url_hash
from src.llm.chunking import compress_text
from src.llm.orchestrator import AllProvidersFailed, LLMOrchestrator
from src.logging_setup import get_logger
from src.metrics import metrics
from src.models import NewsContent, NewsRecord, Source, to_payload
from src.resolver.entity_resolver import EntityResolver
from src.storage.db import Database

log = get_logger("news")

# (name, url, needs_ai_filter)
FEEDS = [
    ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/", False),
    ("VentureBeat", "https://venturebeat.com/feed/", True),  # /category/ai/feed is stale (Aug 2026)
    ("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", False),
    ("MIT Technology Review", "https://www.technologyreview.com/topic/artificial-intelligence/feed", False),
]
HN_QUERIES = ["AI", "LLM", "GPT", "OpenAI", "Anthropic", "machine learning", "neural network", "Gemini"]
_AI_RE = re.compile(r"(?<![A-Za-z0-9])(AI|A\.I\.|LLMs?|GPT|OpenAI|Anthropic|Claude|Gemini|machine learning|neural|"
                    r"generative|chatbot|copilot|agentic|Nvidia|DeepMind|robot\w*|model)(?![A-Za-z0-9])", re.I)

STRUCT_TOKENS = 1500
NEWS_SYSTEM ="You analyze tech news articles. Reply ONLY with a JSON object."
NEWS_INSTRUCTION = """Read the article and return:
{"summary": "<2-3 factual sentences>", "entities": ["<companies / AI labs / startups / products' makers explicitly named, max 8, official names>"]}
Only include organizations that are actually discussed (not authors or generic words). Article:"""


NEWS_BATCH = 15
NEWS_BATCH_INSTRUCTION = """Input is a JSON list of articles: [{"id": "...", "title": "...", "text": "..."}].
For EACH article return its id with a 2-sentence factual summary and the companies / AI labs / startups it explicitly names (max 6, official names).
Reply with: {"results": [{"id": "<same id>", "summary": "...", "entities": ["..."]}]}
Only organizations actually discussed (not authors or generic words). Articles:"""


class NewsLLM(BaseModel):
    summary: str
    entities: list[str] = Field(default_factory=list)


class NewsBatchItem(NewsLLM):
    id: str


class NewsBatch(BaseModel):
    results: list[NewsBatchItem]


def clean_text(html_or_text: str) -> str:
    import trafilatura

    if "<" in html_or_text[:2000]:
        return (trafilatura.extract(html_or_text, include_comments=False, favor_precision=True) or "").strip()
    return html_or_text.strip()


class NewsCrawler(BaseCrawler):
    def __init__(self, db: Database, resolver: EntityResolver, llm: LLMOrchestrator | None, **kw: Any) -> None:
        super().__init__(**kw)
        self.db, self.resolver, self.llm = db, resolver, llm
        self.dedup = DedupStore(db)
        self.article_timeout = 40.0
        self._article_sem = asyncio.Semaphore(12)
        self._llm_sem = asyncio.Semaphore(2)

    # ---- candidates -----------------------------------------------------------------------
    async def fetch_feed(self, name: str, url: str, ai_filter: bool) -> list[dict]:
        res = await self.fetch(url)
        if not res.ok:
            raise FetchError(f"{name}: {res.status}")
        feed = await asyncio.to_thread(feedparser.parse, res.text)
        out = []
        for e in feed.entries:
            body = ""
            if e.get("content"):
                body = e["content"][0].get("value", "")
            body = body or e.get("summary", "")
            title = re.sub(r"\s+", " ", e.get("title", "")).strip()
            tags = " ".join(t.get("term", "") for t in e.get("tags", []))
            if ai_filter and not _AI_RE.search(f"{title} {tags} {body[:500]}"):
                metrics.inc("news.off_topic")
                continue
            out.append({"source": name, "url": e.get("link", ""), "title": title,
                        "date_raw": e.get("published") or e.get("updated"), "feed_body": body})
        return out

    async def fetch_hn(self, hours: int = 24) -> list[dict]:
        since = int(time.time()) - hours * 3600
        seen: dict[str, dict] = {}
        for q in HN_QUERIES:
            page = 0
            while True:
                res = await self.fetch("https://hn.algolia.com/api/v1/search_by_date", escalate=False,
                                       params={"tags": "story", "query": q, "numericFilters": f"created_at_i>{since}",
                                               "hitsPerPage": 100, "page": page})
                if not res.ok:
                    raise FetchError(f"hn {res.status}")
                j = res.json()
                for h in j["hits"]:
                    if h.get("url") and h["objectID"] not in seen and _AI_RE.search(h.get("title", "")):
                        seen[h["objectID"]] = {"source": "Hacker News", "url": h["url"], "title": h["title"].strip(),
                                               "date_raw": h.get("created_at_i"), "feed_body": ""}
                page += 1
                if page >= j.get("nbPages", 0):
                    break
        return list(seen.values())

    # ---- run ------------------------------------------------------------------------------
    async def run(self, now: datetime | None = None, hours: float = 24) -> dict[str, int]:
        now = now or datetime.now(timezone.utc)
        dropped = await self.dedup.reconcile({k for k, _ in await self.db.get_keyed("NEWS")})
        if dropped:
            log.info("dedup_reconciled", dropped_orphans=dropped)
        jobs = {f"feed:{n}": self.fetch_feed(n, u, f) for n, u, f in FEEDS}
        jobs["hn"] = self.fetch_hn(int(hours))
        results = await asyncio.gather(*jobs.values(), return_exceptions=True)
        cands: list[dict] = []
        for (name, _), r in zip(jobs.items(), results):
            if isinstance(r, Exception):
                metrics.inc("news.source_failed")
                log.warning("news_source_failed", source=name, error=str(r)[:150])
                continue
            metrics.inc("news.fetched", len(r))
            log.info("news_source_ok", source=name, n=len(r))
            cands += r
        # cheap pre-filter on feed date before downloading any article
        todo = []
        for c in cands:
            dt = parse_date(c["date_raw"], now)
            if dt is not None and not is_within_hours(dt, hours, now):
                metrics.inc("news.stale_dropped")
                continue
            c["dt_feed"] = dt
            todo.append(c)
        log.info("news_candidates", total=len(cands), to_process=len(todo))
        stats = {"inserted": 0, "updated": 0, "unchanged": 0}
        outs = await asyncio.gather(*[self._process(c, now, hours) for c in todo], return_exceptions=True)
        items = []
        for c, o in zip(todo, outs):
            if isinstance(o, Exception):
                metrics.inc("news.errors")
                log.warning("news_item_failed", url=c["url"], error=repr(o)[:150])
            elif o is not None:
                items.append(o)
        s = await self.db.upsert_many(items)
        for k, v in s.items():
            stats[k] += v
        return stats

    async def _process(self, c: dict, now: datetime, hours: float):
        url = c["url"]
        html, full = "", ""
        try:
            # hard per-article cap: a bot-wall that escalates through all tiers must not stall the whole run
            async with self._article_sem:  # timeout starts only once this article has its slot
                res = await asyncio.wait_for(self.fetch(url, max_tier="curl_cffi"), timeout=self.article_timeout)
            if res.ok:
                html = res.text
                full = clean_text(html)
        except asyncio.TimeoutError:
            metrics.inc("news.article_timeout")
            log.info("article_timeout", url=url)
        except FetchError as e:
            metrics.inc("news.article_fetch_failed")
            log.info("article_fetch_failed", url=url, error=str(e)[:100])
        text_src = "article"
        if len(full) < 300:  # blocked / paywalled / JS page -> fall back to the feed body
            fb = clean_text(c["feed_body"]) if c["feed_body"] else ""
            if len(fb) > len(full):
                full, text_src = fb, "feed"
        if len(full) < 120:
            metrics.inc("news.no_text_dropped")
            return None
        dt, method = resolve_published(c["date_raw"], html or None, now=now)
        if dt is None or not is_within_hours(dt, hours, now):
            metrics.inc("news.stale_dropped" if dt else "news.no_date_dropped")
            return None
        status, _, dup_of = await self.dedup.check_and_add(url, full)
        if status != "new":
            metrics.inc("dedup.hits")
            metrics.inc(f"news.{status}")
            return None
        summary, entities = await self._structure(c["title"], full)
        rec = NewsRecord(
            source=Source(name=c["source"], url=url),
            content=NewsContent(title=c["title"], url=url, source=c["source"], published_date=to_iso(dt),
                                full_text=full, summary=summary, entities=entities),
        )
        metrics.inc("news.parsed")
        metrics.inc(f"news.date_via_{method}")
        metrics.inc(f"news.text_via_{text_src}")
        return ("NEWS", url_hash(url), to_payload(rec), c["source"])

    async def _structure(self, title: str, text: str) -> tuple[str | None, list[str]]:
        if self.llm is None:
            return None, []
        async with self._llm_sem:
            try:
                # lead + key sentences are enough for summary/entities and cut free-tier token spend ~3x
                res = await self.llm.complete("news_struct_v2", NEWS_SYSTEM,
                                              f"{title}\n\n{compress_text(text, STRUCT_TOKENS)}", NEWS_INSTRUCTION)
                parsed = NewsLLM.model_validate(res.json())
            except (AllProvidersFailed, ValueError) as e:
                metrics.inc("news.llm_failed")
                log.warning("news_llm_failed", error=str(e)[:150])
                return None, []
        ents: list[str] = []
        for raw in parsed.entities[:8]:
            canon = self.resolver.resolve(raw, context=f"{title} {parsed.summary}")
            if canon.endswith(" (unverified)"):
                canon = canon.removesuffix(" (unverified)")  # generic-name guard: keep the plain mention
            if canon and canon not in ents:
                ents.append(canon)
        metrics.inc("news.llm_structured")
        return parsed.summary.strip(), ents

    def _resolve_entities(self, names: list[str], context: str, cap: int = 8) -> list[str]:
        ents: list[str] = []
        for raw in names[:cap]:
            canon = self.resolver.resolve(raw, context=context).removesuffix(" (unverified)")
            if canon and canon not in ents:
                ents.append(canon)
        return ents

    async def _structure_batch(self, batch: list[tuple[str, dict]]) -> dict[str, tuple[str, list[str]]]:
        """One LLM call for up to NEWS_BATCH articles (lead + key sentences of each). -> {key: (summary, entities)}"""
        if self.llm is None or not batch:
            return {}
        payload = json.dumps([{"id": k, "title": r["content"]["title"],
                               "text": compress_text(r["content"]["full_text"], 160)} for k, r in batch], ensure_ascii=False)
        async with self._llm_sem:
            try:
                res = await self.llm.complete("news_batch_v1", NEWS_SYSTEM, payload, NEWS_BATCH_INSTRUCTION)
                parsed = NewsBatch.model_validate(res.json())
            except (AllProvidersFailed, ValueError) as e:
                metrics.inc("news.llm_failed_batches")
                log.warning("news_batch_llm_failed", error=str(e)[:150], n=len(batch))
                return {}
        by_key = dict(batch)
        out = {}
        for it in parsed.results:
            r = by_key.get(it.id)
            if r is None or not it.summary.strip():
                continue
            out[it.id] = (it.summary.strip(), self._resolve_entities(it.entities, f"{r['content']['title']} {it.summary}"))
            metrics.inc("news.llm_structured")
        return out

    async def backfill_structure(self, rounds: int = 4) -> int:
        """Re-run summary/entity extraction for stored articles that lack it (provider rate limits / outages).
        Batched: NEWS_BATCH articles per LLM call."""
        for rnd in range(rounds):
            todo = [(k, r) for k, r in await self.db.get_keyed("NEWS") if not r["content"].get("summary")]
            if not todo:
                return 0
            log.info("news_backfill", round=rnd, remaining=len(todo))
            batches = [todo[i:i + NEWS_BATCH] for i in range(0, len(todo), NEWS_BATCH)]
            results = await asyncio.gather(*[self._structure_batch(b) for b in batches])
            done = []
            todo_map = dict(todo)
            for res in results:
                for k, (summary, ents) in res.items():
                    r = todo_map[k]
                    r["content"]["summary"], r["content"]["entities"] = summary, ents
                    done.append(("NEWS", k, r, r["source"]["name"]))
            if done:
                await self.db.upsert_many(done)
            if len(done) < len(todo):
                await self._sleep(40)  # let tripped circuit breakers / free-tier windows recover
        return len([1 for _, r in await self.db.get_keyed("NEWS") if not r["content"].get("summary")])
