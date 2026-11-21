"""Products quality pass: real website + domain, /pricing page, site-based pricing, domain-aware resolution.

For every stored product:
  1. fetch the Product Hunt redirect (/r/...) through the escalation ladder (it is Cloudflare-protected;
     httpx 403 -> curl_cffi -> playwright). The final URL is the real site -> websiteUrl + domain.
  2. for rows whose pricing is only an `llm_guess` (or null): fetch the pricing page (link discovered in the
     homepage, else /pricing), extract text with trafilatura, re-run the pricing LLM on the SITE text.
     Keep `llm_guess` only if the site gives no signal.
  3. re-resolve startupName with the real domain.
Persisted every 30 products; `siteChecked` makes it resumable.
"""
from __future__ import annotations

import asyncio
import json
from urllib.parse import urljoin, urlparse

from lxml import html as lxml_html

from src.crawlers.base_crawler import FetchError
from src.crawlers.products import PH_API, ProductsCrawler
from src.llm.orchestrator import AllProvidersFailed
from src.llm.prompts import PRICING_SITE_INSTRUCTION, PRICING_SYSTEM
from src.llm.schemas import PricingBatch
from src.logging_setup import get_logger
from src.metrics import metrics
from src.resolver.entity_resolver import EntityResolver, extract_domain

log = get_logger("product_sites")
SITE_BATCH = 15


def site_text(html: str, limit: int = 3500) -> str:
    import trafilatura

    return (trafilatura.extract(html, include_comments=False, include_tables=True) or "")[:limit]


def pricing_link(html: str, base: str) -> str | None:
    try:
        tree = lxml_html.fromstring(html)
    except Exception:
        return None
    for a in tree.xpath("//a[@href]"):
        href = a.get("href", "")
        label = (a.text_content() or "").strip().lower()
        if "pricing" in href.lower() or label in ("pricing", "plans", "plans & pricing"):
            return urljoin(base, href)
    return None


class ProductSiteEnricher(ProductsCrawler):
    async def _ph_redirects(self, ids: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for i in range(0, len(ids), 40):
            chunk = ids[i:i + 40]
            q = "query { " + " ".join(f'p{j}: post(id: "{x}") {{ id website }}' for j, x in enumerate(chunk)) + " }"
            res = await self.fetch(PH_API, method="POST", escalate=False, json={"query": q},
                                   headers={"Authorization": f"Bearer {self.token}"})
            for node in (res.json().get("data") or {}).values():
                if node and node.get("website"):
                    out[str(node["id"])] = node["website"]
        return out

    async def _enrich_one(self, rec: dict, redirect: str, sem: asyncio.Semaphore) -> str | None:
        """Mutates rec (websiteUrl/domain/siteChecked). Returns site text if pricing needs (re)extraction."""
        c = rec["content"]
        c["siteChecked"] = True
        async with sem:
            try:
                home = await self.fetch(redirect, render=True)  # /r/ links 403 on httpx+curl_cffi: go straight to the browser
            except FetchError as e:
                metrics.inc("products.site_blocked")
                c["websiteUrl"] = c["domain"] = None
                log.info("site_unreachable", product=c["productName"], error=str(e)[:80])
                return None
            final = home.url.split("?")[0]
            host = urlparse(final).netloc
            if not host or "producthunt.com" in host:  # redirect never left PH
                metrics.inc("products.site_blocked")
                c["websiteUrl"] = c["domain"] = None
                return None
            c["websiteUrl"], c["domain"] = final, extract_domain(final)
            metrics.inc("products.site_resolved")
            if c.get("pricingSource") not in ("llm_guess", None) and c.get("pricingModel"):
                return None  # already grounded in PH text (rule/llm): only the domain was needed
            link = pricing_link(home.text, home.url) or f"{urlparse(final).scheme}://{host}/pricing"
            ptext = ""
            try:
                pr = await self.fetch(link)
                if pr.ok:
                    ptext = site_text(pr.text)
                    metrics.inc("products.pricing_page_found" if ptext else "products.pricing_page_empty")
            except FetchError:
                pass
            rec["_pricing_page"] = bool(ptext)
            return (f"[PRICING PAGE]\n{ptext}\n" if ptext else "") + f"[HOMEPAGE]\n{site_text(home.text, 1800)}"

    async def _site_pricing(self, items: list[tuple[dict, str]], stats: dict[str, int]) -> None:
        if not items or self.llm is None:
            return
        for i in range(0, len(items), SITE_BATCH):
            batch = items[i:i + SITE_BATCH]
            payload = json.dumps([{"id": r["_key"], "name": r["content"]["productName"], "text": t[:900]} for r, t in batch],
                                 ensure_ascii=False)
            try:
                res = await self.llm.complete("pricing_site_v1", PRICING_SYSTEM, payload, PRICING_SITE_INSTRUCTION)
                parsed = {it.id: it.pricingModel for it in PricingBatch.model_validate(res.json()).results}
            except (AllProvidersFailed, ValueError) as e:
                log.warning("site_pricing_llm_failed", error=str(e)[:150])
                continue
            for r, _ in batch:
                pm, c = parsed.get(r["_key"]), r["content"]
                if pm:
                    c["pricingModel"] = pm
                    c["pricingSource"] = "site_pricing_page" if r.get("_pricing_page") else "site_homepage"
                    stats["pricing_from_site"] += 1
                    metrics.inc("products.pricing_site")
                else:
                    stats["kept_guess"] += 1

    async def enrich_sites(self, resolver: EntityResolver, concurrency: int = 4, only_guess: bool = True) -> dict[str, int]:
        keyed = await self.db.get_keyed("PRODUCT")
        todo = [(k, r) for k, r in keyed if not r["content"].get("siteChecked")
                and (not only_guess or r["content"].get("pricingSource") in ("llm_guess", None))]
        log.info("site_enrich_start", todo=len(todo), total=len(keyed))
        redirects = await self._ph_redirects([k.removeprefix("ph:") for k, _ in todo])
        sem = asyncio.Semaphore(concurrency)
        stats = {"with_domain": 0, "no_domain": 0, "pricing_from_site": 0, "kept_guess": 0}

        async def work(k: str, r: dict) -> tuple[str, dict, str | None]:
            r["_key"] = k
            red = redirects.get(k.removeprefix("ph:"))
            if not red:
                r["content"]["siteChecked"] = True
                return k, r, None
            return k, r, await self._enrich_one(r, red, sem)

        for i in range(0, len(todo), 30):
            results = await asyncio.gather(*[work(k, r) for k, r in todo[i:i + 30]])
            await self._site_pricing([(r, t) for _, r, t in results if t], stats)
            items = []
            for k, r, _ in results:
                c = r["content"]
                r.pop("_key", None)
                r.pop("_pricing_page", None)
                c["startupName"] = resolver.resolve(c["productName"], domain=c.get("domain"),
                                                    context=f"{c.get('tagline') or ''} {c.get('description') or ''}")
                stats["with_domain" if c.get("domain") else "no_domain"] += 1
                items.append(("PRODUCT", k, r, "Product Hunt"))
            await self.db.upsert_many(items)
            log.info("site_enrich_progress", done=min(i + 30, len(todo)), total=len(todo), **stats)
        return stats

    async def reprice_from_sites(self) -> dict[str, int]:
        """Rows with a resolved website but still `llm_guess` (LLM was rate-limited during enrichment):
        re-fetch homepage + pricing page and re-run site-based pricing. No redirect solving needed."""
        keyed = await self.db.get_keyed("PRODUCT")
        todo = [(k, r) for k, r in keyed if r["content"].get("pricingSource") == "llm_guess" and r["content"].get("websiteUrl")]
        stats = {"candidates": len(todo), "pricing_from_site": 0, "kept_guess": 0}
        log.info("reprice_start", candidates=len(todo))
        sem = asyncio.Semaphore(6)

        async def prep(k: str, r: dict) -> tuple[dict, str | None]:
            r["_key"] = k
            url = r["content"]["websiteUrl"]
            async with sem:
                try:
                    home = await asyncio.wait_for(self.fetch(url, max_tier="curl_cffi"), 40)
                except (FetchError, asyncio.TimeoutError):
                    return r, None
                link = pricing_link(home.text, home.url) or f"{url.rstrip('/')}/pricing"
                ptext = ""
                try:
                    pr = await asyncio.wait_for(self.fetch(link, max_tier="curl_cffi"), 40)
                    ptext = site_text(pr.text) if pr.ok else ""
                except (FetchError, asyncio.TimeoutError):
                    pass
                r["_pricing_page"] = bool(ptext)
                return r, (f"[PRICING PAGE]\n{ptext}\n" if ptext else "") + f"[HOMEPAGE]\n{site_text(home.text, 1800)}"

        for i in range(0, len(todo), 30):
            prepared = await asyncio.gather(*[prep(k, r) for k, r in todo[i:i + 30]])
            await self._site_pricing([(r, t) for r, t in prepared if t], stats)
            await self.db.upsert_many([("PRODUCT", r.pop("_key"), {k: v for k, v in r.items() if k != "_pricing_page"}, "Product Hunt")
                                       for r, _ in prepared])
        return stats
