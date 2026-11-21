"""CLI: python -m src.pipeline --vertical startups|products|papers|news|jobs|all"""
from __future__ import annotations

import argparse
import asyncio
import sys

from src.config import get_settings
from src.logging_setup import get_logger, setup_logging
from src.metrics import metrics
from src.storage.db import Database

log = get_logger("pipeline")
VERTICALS = ["startups", "products", "papers", "news", "jobs"]


async def run_startups(db: Database, s, limit: int | None) -> None:
    from src.crawlers.startups import YCStartupsCrawler

    async with YCStartupsCrawler(db, concurrency=s.concurrency, per_domain_rps=s.per_domain_rps) as c:
        stats = await c.run(limit=limit)
    log.info("startups_done", **stats, total_in_db=await db.count("STARTUP"))


async def run_papers(db: Database, s, limit: int | None) -> None:
    from src.crawlers.papers import PapersCrawler

    async with PapersCrawler(db, github_token=s.github_token, concurrency=s.concurrency,
                             per_domain_rps=s.per_domain_rps) as c:
        await c.run(target=limit or 1000)
        counts = await c.enrich_stars()
    log.info("papers_done", total_with_github=await db.count("RESEARCH_PAPER"), **counts)


async def build_resolver(db: Database):
    """Seed entities + AI YC startups (single-word YC names are 'ambiguous': need domain/AI context)."""
    from src.resolver.entity_resolver import EntityResolver, extract_domain

    r = EntityResolver()
    n = 0
    for rec in await db.get_records("STARTUP"):
        d = rec["content"]["data"]
        if not d.get("is_ai"):
            continue
        name = rec["content"]["entityName"]
        if name in r.canonicals:
            continue
        r.add_entity(name, d.get("formerNames") or [], extract_domain(d["website"]) if d.get("website") else None,
                     ambiguous=len(name.split()) == 1)
        n += 1
    log.info("resolver_seeded", seed=len(r.canonicals) - n, yc_ai=n)
    return r


async def persist_entity_log(db: Database, resolver) -> None:
    items = [("ENTITY_LOG", f"{d.raw}|{d.canonical}", {"raw_name": d.raw, "canonical_name": d.canonical,
             "method": d.method, "score": d.score, "timestamp": d.timestamp, "collectedAt": d.timestamp}, "resolver")
             for d in resolver.decisions]
    await db.upsert_many(items)


async def run_products(db: Database, s, limit: int | None) -> None:
    from src.crawlers.products import ProductsCrawler
    from src.llm.orchestrator import LLMOrchestrator, default_providers

    resolver = await build_resolver(db)
    llm = LLMOrchestrator(default_providers(s), cache=db)
    async with ProductsCrawler(db, s.product_hunt_token, resolver, llm, concurrency=s.concurrency,
                               per_domain_rps=s.per_domain_rps) as c:
        stats = await c.run(target=limit or 1200)
        still_null = await c.backfill_pricing()
        log.info("pricing_backfill_done", still_null=still_null)
    await persist_entity_log(db, resolver)
    log.info("products_done", **stats, total=await db.count("PRODUCT"), entity_decisions=len(resolver.decisions))


async def run_product_sites(db: Database, s, limit: int | None) -> None:
    """Quality pass over stored products: real website/domain, site-based pricing, domain-aware resolution."""
    from src.crawlers.product_sites import ProductSiteEnricher
    from src.llm.orchestrator import LLMOrchestrator, default_providers

    resolver = await build_resolver(db)
    llm = LLMOrchestrator(default_providers(s), cache=db)
    async with ProductSiteEnricher(db, s.product_hunt_token, resolver, llm, concurrency=s.concurrency,
                                   per_domain_rps=s.per_domain_rps) as c:
        stats = await c.enrich_sites(resolver)
        stats.update({f"reprice_{k}": v for k, v in (await c.reprice_from_sites()).items()})
    await persist_entity_log(db, resolver)
    log.info("product_sites_done", **stats)


async def run_product_directory(db: Database, s, limit: int | None) -> None:
    """Second products source with explicit pricing labels (theresanaiforthat.com); upgrades/dedups vs stored."""
    from src.crawlers.product_directory import DirectoryCrawler
    from src.llm.orchestrator import LLMOrchestrator, default_providers

    resolver = await build_resolver(db)
    llm = LLMOrchestrator(default_providers(s), cache=db)
    async with DirectoryCrawler(db, resolver, llm, concurrency=s.concurrency, per_domain_rps=s.per_domain_rps) as c:
        stats = await c.run(target_non_guess=limit or 1300)
    await persist_entity_log(db, resolver)
    log.info("product_directory_done", **stats, total=await db.count("PRODUCT"))


async def run_jobs(db: Database, s, limit: int | None) -> None:
    from src.crawlers.jobs import JobsCrawler
    from src.llm.orchestrator import LLMOrchestrator, default_providers

    resolver = await build_resolver(db)
    llm = LLMOrchestrator(default_providers(s), cache=db)
    async with JobsCrawler(db, resolver, llm, concurrency=s.concurrency, per_domain_rps=s.per_domain_rps) as c:
        stats = await c.run()
    await persist_entity_log(db, resolver)
    log.info("jobs_done", **stats, total_in_db=await db.count("JOB"))


async def run_news(db: Database, s, limit: int | None) -> None:
    from src.crawlers.news import NewsCrawler
    from src.llm.orchestrator import LLMOrchestrator, default_providers

    resolver = await build_resolver(db)
    llm = LLMOrchestrator(default_providers(s), cache=db)
    async with NewsCrawler(db, resolver, llm, concurrency=s.concurrency, per_domain_rps=s.per_domain_rps) as c:
        stats = await c.run()
        still = await c.backfill_structure()
        log.info("news_backfill_done", still_without_summary=still)
    await persist_entity_log(db, resolver)
    log.info("news_done", **stats, total_in_db=await db.count("NEWS"))


async def run_export(db: Database, s, limit: int | None) -> None:
    import os

    from src.storage.export_sheets import export

    dry = os.getenv("EXPORT_DRY_RUN") == "1"
    out = await export(db, s.google_sheet_id, s.google_credentials, dry_run=dry)
    log.info("export_done", **{k: v for k, v in out.items()})


RUNNERS = {"startups": run_startups, "papers": run_papers, "products": run_products,
           "product_sites": run_product_sites, "product_directory": run_product_directory, "jobs": run_jobs, "news": run_news, "export": run_export}


async def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="src.pipeline")
    ap.add_argument("--vertical", required=True, choices=VERTICALS + ["product_sites", "product_directory", "export", "all"])
    ap.add_argument("--limit", type=int, default=None, help="stop after N records per vertical (smoke tests)")
    ap.add_argument("--reset", action="store_true", help="ignore saved checkpoints for the chosen vertical(s)")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--dry-run", action="store_true", help="export: write local CSVs only, skip Google Sheets")
    args = ap.parse_args(argv)

    setup_logging(args.log_level)
    if args.dry_run:
        import os
        os.environ["EXPORT_DRY_RUN"] = "1"
    s = get_settings()
    db = Database(s.db_url)
    targets = VERTICALS if args.vertical == "all" else [args.vertical]
    rc = 0
    try:
        for v in targets:
            runner = RUNNERS.get(v)
            if runner is None:
                log.warning("vertical_not_implemented_yet", vertical=v)
                continue
            if args.reset:
                await db.clear_checkpoint({"startups": "startups:yc", "papers": "papers:arxiv", "products": "products:ph"}.get(v, v))
            try:
                await runner(db, s, args.limit)
            except Exception as e:  # one vertical failing must not kill the others
                log.error("vertical_failed", vertical=v, error=repr(e))
                rc = 1
    finally:
        print("\n=== run metrics ===\n" + metrics.report(), file=sys.stderr)
        db.close()
    return rc


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
