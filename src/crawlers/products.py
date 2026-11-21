"""Products from the Product Hunt GraphQL API (AI topic, newest first).

Deterministic: name, tagline, description, votes, created date, topics, canonical PH URL.
Resolver:      startupName = canonical entity for the product's maker/brand name.
LLM gap-fill:  pricingModel (PH exposes no pricing field), batched 10 products per call, cached by
               content hash.  A conservative regex handles unambiguous 'open source / free forever' cases.
Checkpoint:    PH pagination cursor; crash resumes at the next page.
Fallback:      if PH is unavailable, theresanaiforthat.com sitemap works via the curl_cffi tier
               (verified in scripts/playwright_demo.py) - not wired unless needed.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any
from urllib.parse import urlparse

from src.crawlers.base_crawler import BaseCrawler, FetchError
from src.llm.orchestrator import AllProvidersFailed, LLMOrchestrator
from src.llm.prompts import PRICING_GUESS_INSTRUCTION, PRICING_INSTRUCTION, PRICING_SYSTEM
from src.llm.schemas import PricingBatch
from src.logging_setup import get_logger
from src.metrics import metrics
from src.models import ProductContent, ProductRecord, Source, to_payload
from src.resolver.entity_resolver import EntityResolver
from src.storage.db import Database

log = get_logger("products")

PH_API = "https://api.producthunt.com/v2/api/graphql"
JOB = "products:ph"
PAGE = 50
LLM_BATCH = 15  # items per LLM call: fewer calls = fewer free-tier requests/tokens burned on prompt overhead

QUERY = """query($c: String, $n: Int) {
  posts(first: $n, after: $c, order: NEWEST, topic: "artificial-intelligence") {
    pageInfo { endCursor hasNextPage }
    edges { node { id name tagline description slug url createdAt votesCount
                   topics { edges { node { name } } } } }
  }
}"""

_FREE_RULE = re.compile(r"\b(open[- ]source|free forever|100% free|completely free|totally free|free and open)\b", re.I)
_PAID_SIGNALS = re.compile(r"\b(pro plan|premium|paid|pricing|subscription|per (month|seat|user)|/mo|upgrade|trial|enterprise)\b|\$\s?\d", re.I)


def rule_pricing(text: str) -> str | None:
    """Only the unambiguous case; everything else is left to the LLM."""
    return "FREE" if _FREE_RULE.search(text) and not _PAID_SIGNALS.search(text) else None


def clean_ph_url(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}{p.path}"


class ProductsCrawler(BaseCrawler):
    def __init__(self, db: Database, token: str, resolver: EntityResolver, llm: LLMOrchestrator | None,
                 **kw: Any) -> None:
        super().__init__(**kw)
        self.db, self.token, self.resolver, self.llm = db, token, resolver, llm
        self._llm_sem = asyncio.Semaphore(4)

    # ---- fetch -----------------------------------------------------------------
    async def _page(self, cursor: str | None) -> dict:
        res = await self.fetch(PH_API, method="POST", escalate=False,
                               json={"query": QUERY, "variables": {"c": cursor, "n": PAGE}},
                               headers={"Authorization": f"Bearer {self.token}"})
        if not res.ok:
            raise FetchError(f"producthunt {res.status}: {res.text[:200]}")
        body = res.json()
        if body.get("errors") and not body.get("data"):
            raise FetchError(f"producthunt graphql: {str(body['errors'])[:200]}")
        rem = res.headers.get("x-rate-limit-remaining")
        if rem is not None and int(rem) < 300:  # stay clear of the 6250/15min complexity budget
            wait = int(res.headers.get("x-rate-limit-reset", 60))
            log.warning("ph_rate_budget_low", remaining=rem, sleeping=wait)
            await self._sleep(min(wait, 900))
        return body["data"]["posts"]

    async def run(self, target: int = 1200) -> dict[str, int]:
        ckpt = await self.db.get_checkpoint(JOB)
        cursor, fetched = ckpt.get("cursor"), ckpt.get("fetched", 0)
        total = {"inserted": 0, "updated": 0, "unchanged": 0}
        # PH uses offset-style cursors, so pages overlap as new posts arrive: stop on *stored* unique count
        while await self.db.count("PRODUCT") < target:
            posts = await self._page(cursor)
            nodes = [e["node"] for e in posts["edges"]]
            metrics.inc("products.fetched", len(nodes))
            recs = await self._build(nodes)
            items = [("PRODUCT", f"ph:{n['id']}", to_payload(r), "Product Hunt") for n, r in recs]
            s = await self.db.upsert_many(items)
            for k, v in s.items():
                total[k] += v
            metrics.inc("dedup.hits", s["unchanged"])
            fetched += len(nodes)
            info = posts["pageInfo"]
            cursor = info["endCursor"]
            await self.db.set_checkpoint(JOB, {"cursor": cursor, "fetched": fetched})
            log.info("ph_page_done", fetched=fetched, stored=await self.db.count("PRODUCT"))
            if not info["hasNextPage"]:
                break
        return total

    # ---- build -----------------------------------------------------------------
    async def _build(self, nodes: list[dict]) -> list[tuple[dict, ProductRecord]]:
        pending: list[tuple[dict, ProductRecord, str]] = []
        for n in nodes:
            text = f"{n.get('name', '')}. {n.get('tagline') or ''}. {n.get('description') or ''}"
            topics = [t["node"]["name"] for t in (n.get("topics") or {}).get("edges", [])]
            rec = ProductRecord(
                source=Source(name="Product Hunt", url=clean_ph_url(n["url"])),
                content=ProductContent(
                    startupName=self.resolver.resolve(n["name"], context=f"{n.get('tagline') or ''} {n.get('description') or ''} {' '.join(topics)}"),
                    pricingModel=rule_pricing(text),
                    productName=n["name"], tagline=n.get("tagline"), description=n.get("description"),
                    votesCount=n.get("votesCount"), launchedAt=n.get("createdAt"), topics=topics,
                    pricingSource="rule" if rule_pricing(text) else None,
                ),
            )
            metrics.inc("products.parsed")
            pending.append((n, rec, text))
        await self._fill_pricing([(n, r, t) for n, r, t in pending if r.content.pricingModel is None])
        return [(n, r) for n, r, _ in pending]

    async def _fill_pricing(self, todo: list[tuple[dict, ProductRecord, str]]) -> None:
        if not todo or self.llm is None:
            return
        batches = [todo[i:i + LLM_BATCH] for i in range(0, len(todo), LLM_BATCH)]
        await asyncio.gather(*[self._llm_batch(b) for b in batches])
        # second pass: items with no explicit signal get a forced best guess, flagged as such
        unknown = [t for t in todo if t[1].content.pricingModel is None]
        if unknown:
            await asyncio.gather(*[self._llm_batch(unknown[i:i + LLM_BATCH], guess=True)
                                   for i in range(0, len(unknown), LLM_BATCH)])

    async def _llm_batch(self, batch: list[tuple[dict, ProductRecord, str]], guess: bool = False) -> None:
        payload = json.dumps([{"id": n["id"], "name": n["name"], "text": t[:600]} for n, _, t in batch],
                             ensure_ascii=False)
        async with self._llm_sem:
            try:
                res = await self.llm.complete("pricing_guess_v1" if guess else "pricing_v1", PRICING_SYSTEM, payload,
                                              PRICING_GUESS_INSTRUCTION if guess else PRICING_INSTRUCTION)
                parsed = PricingBatch.model_validate(res.json())
            except (AllProvidersFailed, ValueError) as e:  # ValueError covers bad JSON + pydantic validation
                metrics.inc("products.llm_failed_batches")
                log.warning("pricing_llm_failed", error=str(e)[:200], n=len(batch))
                return
        by_id = {it.id: it.pricingModel for it in parsed.results}
        for n, rec, _ in batch:
            pm = by_id.get(str(n["id"]))
            if pm:
                rec.content.pricingModel = pm
                rec.content.pricingSource = "llm_guess" if guess else "llm"
                metrics.inc("products.pricing_guess" if guess else "products.pricing_llm")
            else:
                metrics.inc("products.pricing_unknown")

    async def backfill_pricing(self, rounds: int = 3) -> int:
        """Retry LLM pricing for stored products still null (e.g. after a provider outage)."""
        if self.llm is None:
            return 0
        for rnd in range(rounds):
            recs = [(k, r) for k, r in await self.db.get_keyed("PRODUCT") if r["content"].get("pricingModel") is None]
            if not recs:
                return 0
            log.info("pricing_backfill", round=rnd, remaining=len(recs))
            todo = []
            for key, r in recs:
                c = r["content"]
                node = {"id": key.removeprefix("ph:"), "name": c["productName"]}
                text = f"{c['productName']}. {c.get('tagline') or ''}. {c.get('description') or ''}"
                todo.append((node, ProductRecord.model_validate(r), text))
            await self._fill_pricing(todo)
            await self.db.upsert_many([("PRODUCT", f"ph:{n['id']}", to_payload(rec), "Product Hunt") for n, rec, _ in todo
                                       if rec.content.pricingModel is not None])
            await self._sleep(35)  # let a tripped circuit breaker cool down
        return len([r for r in await self.db.get_records("PRODUCT") if r["content"].get("pricingModel") is None])
