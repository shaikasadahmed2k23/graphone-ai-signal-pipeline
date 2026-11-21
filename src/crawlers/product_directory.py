"""Second products source: theresanaiforthat.com (TAAFT), a directory with EXPLICIT pricing labels.

Verified live at the curl_cffi tier (httpx is blocked). Every tool card carries a label such as
"100% Free", "Free + from $19/mo", "From $5.90", "Contact for pricing", "No pricing":

    label -> pricingModel is a deterministic map (pricingSource="directory_label").
    Only an unmappable label ("No pricing", unknown text) goes to the LLM, and then it is flagged llm_guess.

Discovery: homepage -> /task/<slug>/ pages (each lists 30-140 tools) -> new task links found on those pages (BFS).
Dedup against products already stored (Product Hunt): same domain, or same normalised name when at most one side
knows its domain. A match upgrades the stored row's pricing to the directory label instead of inserting a twin.
startupName always goes through the entity resolver.
"""
from __future__ import annotations

import asyncio
import html
import re
from typing import Any
from urllib.parse import urlparse

from src.crawlers.base_crawler import FetchError
from src.crawlers.products import ProductsCrawler
from src.logging_setup import get_logger
from src.metrics import metrics
from src.models import ProductContent, ProductRecord, Source, to_payload
from src.resolver.entity_resolver import EntityResolver, extract_domain

log = get_logger("product_directory")

BASE = "https://theresanaiforthat.com"
JOB = "products:taaft"
SOURCE_NAME = "There's An AI For That"
NON_GUESS = ("llm_guess", None)

_CARD_SPLIT = re.compile(r'(?=<li class="li )')
_TASK_LINK = re.compile(r'href="https://theresanaiforthat\.com/task/([a-z0-9][a-z0-9-]*)/')
_ATTR = lambda name: re.compile(rf'\b{name}="([^"]*)"')  # noqa: E731
_LABEL = re.compile(r'class="ai_launch_date"[^>]*>\s*([^<]*?)\s*<')
_DESC = re.compile(r'<div class="short_desc">\s*(.*?)\s*</div>', re.S)
_SLUG = re.compile(r'href="https://theresanaiforthat\.com/ai/([^/"]+)/"')
_COMPACT = re.compile(r"[^a-z0-9]+")


def map_label(label: str | None) -> str | None:
    """Deterministic label -> pricingModel. None = unmappable (caller may fall back to the LLM)."""
    if not label:
        return None
    s = label.strip().lower()
    if s in ("no pricing", "n/a", "unknown", "-"):
        return None
    if "contact" in s or "enterprise" in s or "custom pricing" in s or "request" in s:
        return "ENTERPRISE"
    if s.startswith("free +") or s.startswith("freemium") or ("free" in s and "from" in s):
        return "FREEMIUM"
    if s in ("100% free", "free", "free forever", "open source") or s.startswith("100% free"):
        return "FREE"
    if "free trial" in s or s.startswith("from") or s.startswith("$") or "/mo" in s or "/yr" in s or "paid" in s:
        return "PAID"
    return None


def compact(name: str) -> str:
    return _COMPACT.sub("", name.lower())


def clean_site(url: str) -> str | None:
    """Strip tracking params (ref=taaft, utm_*)."""
    u = urlparse(html.unescape(url or "").strip())
    if u.scheme not in ("http", "https") or not u.netloc:
        return None
    return f"{u.scheme}://{u.netloc}{u.path}".rstrip("/")


def parse_cards(page: str) -> list[dict]:
    out = []
    body = page[page.find("<body"):] if "<body" in page else page
    for chunk in _CARD_SPLIT.split(body)[1:]:
        head = chunk[:900]
        tid, name = _ATTR("data-id").search(head), _ATTR("data-name").search(head)
        if not tid or not name:
            continue
        site = _ATTR("data-url").search(head)
        label, desc, slug = _LABEL.search(chunk), _DESC.search(chunk), _SLUG.search(chunk)
        task = _ATTR("data-task").search(head)
        out.append({
            "id": tid.group(1), "name": html.unescape(name.group(1)).strip(),
            "website": clean_site(site.group(1)) if site else None,
            "label": html.unescape(label.group(1)).strip() if label else None,
            "desc": re.sub(r"<[^>]+>|\s+", " ", html.unescape(desc.group(1))).strip() if desc else "",
            "slug": slug.group(1) if slug else None, "task": html.unescape(task.group(1)) if task else None,
        })
    return out




class Snapshot:
    """In-memory view of stored products used for dedup (by domain / normalised name)."""

    def __init__(self, keyed: list[tuple[str, dict]]) -> None:
        self.rows = dict(keyed)
        self.by_name: dict[str, str] = {}
        self.by_domain: dict[str, str] = {}
        for k, r in keyed:
            c = r["content"]
            self.by_name.setdefault(compact(c.get("productName") or ""), k)
            if c.get("domain"):
                self.by_domain.setdefault(c["domain"], k)

    @property
    def non_guess(self) -> int:
        return sum(1 for r in self.rows.values()
                   if r["content"].get("pricingModel") and r["content"].get("pricingSource") not in NON_GUESS)

    def match(self, name: str, domain: str | None) -> str | None:
        """Same domain -> match. Same normalised name -> match unless both sides have different domains."""
        if domain and domain in self.by_domain:
            return self.by_domain[domain]
        key = self.by_name.get(compact(name))
        if key is None:
            return None
        known = self.rows[key]["content"].get("domain")
        return None if (known and domain and known != domain) else key


class DirectoryCrawler(ProductsCrawler):
    """Reuses ProductsCrawler's LLM batch fallback; the Product Hunt token is not used."""

    def __init__(self, db, resolver: EntityResolver, llm, **kw: Any) -> None:
        super().__init__(db, "", resolver, llm, **kw)

    async def _get(self, url: str) -> str:
        res = await self.fetch(url, max_tier="curl_cffi")
        if not res.ok:
            raise FetchError(f"{url}: {res.status}")
        return res.text

    async def run(self, target_non_guess: int = 1300, max_tasks: int = 600) -> dict[str, int]:
        ckpt = await self.db.get_checkpoint(JOB)
        done: set[str] = set(ckpt.get("done_tasks", []))
        queue: list[str] = ckpt.get("queue") or []
        stats = {"new": 0, "upgraded": 0, "skipped_existing": 0, "llm_guess": 0, "tasks": 0}
        if not queue and not done:
            home = await self._get(BASE + "/")
            queue = sorted(set(_TASK_LINK.findall(home)))
            await self._ingest(parse_cards(home), stats)
            log.info("taaft_seed", tasks=len(queue))
        snap = Snapshot(await self.db.get_keyed("PRODUCT"))
        while queue and len(done) < max_tasks and snap.non_guess < target_non_guess:
            slug = queue.pop(0)
            if slug in done:
                continue
            try:
                page = await self._get(f"{BASE}/task/{slug}/")
            except FetchError as e:
                metrics.inc("taaft.task_failed")
                log.warning("taaft_task_failed", task=slug, error=str(e)[:100])
                done.add(slug)
                continue
            for s in _TASK_LINK.findall(page):  # BFS: task pages link to related tasks
                if s not in done and s not in queue:
                    queue.append(s)
            await self._ingest(parse_cards(page), stats)
            done.add(slug)
            stats["tasks"] += 1
            snap = Snapshot(await self.db.get_keyed("PRODUCT"))
            await self.db.set_checkpoint(JOB, {"done_tasks": sorted(done), "queue": queue})
            log.info("taaft_task_done", task=slug, queue=len(queue), non_guess=snap.non_guess,
                     new=stats["new"], upgraded=stats["upgraded"])
        return stats

    async def _ingest(self, cards: list[dict], stats: dict[str, int]) -> None:
        snap = Snapshot(await self.db.get_keyed("PRODUCT"))
        writes: list[tuple] = []
        to_llm: list[tuple[dict, ProductRecord, str]] = []
        seen: set[str] = set()
        for c in cards:
            if c["id"] in seen:
                continue
            seen.add(c["id"])
            metrics.inc("taaft.cards")
            domain = extract_domain(c["website"]) if c["website"] else None
            pm = map_label(c["label"])
            key = snap.match(c["name"], domain)
            if key is not None:  # already stored (e.g. from Product Hunt): upgrade weak pricing, never duplicate
                r = snap.rows[key]
                ec = r["content"]
                if pm is None or ec.get("pricingSource") in ("rule", "site_pricing_page", "site_homepage", "directory_label"):
                    stats["skipped_existing"] += 1
                    continue
                ec["pricingModel"], ec["pricingSource"], ec["pricingLabel"] = pm, "directory_label", c["label"]
                if not ec.get("domain") and domain:
                    ec["websiteUrl"], ec["domain"] = c["website"], domain
                writes.append(("PRODUCT", key, r, r["source"]["name"]))
                stats["upgraded"] += 1
                metrics.inc("taaft.upgraded")
                continue
            rec = ProductRecord(
                source=Source(name=SOURCE_NAME, url=f"{BASE}/ai/{c['slug']}/" if c["slug"] else BASE),
                content=ProductContent(
                    startupName=self.resolver.resolve(c["name"], domain=domain, context=f"{c['desc']} {c['task'] or ''}"),
                    pricingModel=pm, productName=c["name"], tagline=c["desc"][:200] or None,
                    description=c["desc"] or None, topics=[c["task"]] if c["task"] else [],
                    websiteUrl=c["website"], domain=domain,
                    pricingSource="directory_label" if pm else None, pricingLabel=c["label"],
                ),
            )
            if pm is None:
                to_llm.append(({"id": c["id"], "name": c["name"]}, rec, f"{c['name']}. {c['desc']}"))
            else:
                writes.append(("PRODUCT", f"taaft:{c['id']}", to_payload(rec), SOURCE_NAME))
                stats["new"] += 1
                metrics.inc("taaft.new")
        if to_llm and self.llm is not None:  # unmappable label ("No pricing") -> LLM, flagged llm_guess
            await asyncio.gather(*[self._llm_batch(to_llm[i:i + 15], guess=True) for i in range(0, len(to_llm), 15)])
        for n, rec, _ in to_llm:
            writes.append(("PRODUCT", f"taaft:{n['id']}", to_payload(rec), SOURCE_NAME))
            stats["llm_guess"] += 1
        if writes:
            await self.db.upsert_many(writes)
