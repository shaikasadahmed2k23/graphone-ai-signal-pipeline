"""Export the 6 tabs to a Google Sheet (gspread) or, with dry_run, to local CSVs.

Tabs: Startups, Products, Research Papers, Jobs, News, Entity Mapping Log.
* Startups: AI startups only if there are >= 1000, otherwise filled up with non-AI (is_ai column marks them).
* Jobs / News: strict last-24h window applied again at export time (a stale DB row never leaks into the sheet).
* Each row keeps `record_json` = the exact schema-conformant JSON record, next to flattened columns.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.freshness.date_utils import is_within_hours, parse_date
from src.logging_setup import get_logger
from src.storage.db import Database

log = get_logger("export")
CELL_LIMIT = 45_000  # Sheets hard limit is 50,000 chars per cell
MIN_STARTUPS = 1000


def _cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        v = json.dumps(v, ensure_ascii=False)
    if isinstance(v, str) and len(v) > CELL_LIMIT:
        return v[:CELL_LIMIT] + "…[truncated]"
    return v


def _json_cell(rec: dict, trim: tuple[str, ...] = ()) -> str:
    """Whole record as JSON; large text fields trimmed so the cell stays under the limit."""
    r = json.loads(json.dumps(rec))
    for path in trim:
        node = r
        *parents, leaf = path.split(".")
        for p in parents:
            node = node.get(p, {})
        if isinstance(node.get(leaf), str) and len(node[leaf]) > 8000:
            node[leaf] = node[leaf][:8000] + "…[truncated in sheet]"
    return json.dumps(r, ensure_ascii=False)


COMMON = ["schemaVersion", "recordType", "source.name", "source.url", "collectedAt"]


def _get(rec: dict, path: str) -> Any:
    cur: Any = rec
    for p in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _table(recs: list[dict], cols: list[str], trim: tuple[str, ...] = ()) -> list[list[Any]]:
    header = [c.split(".")[-1] if c not in COMMON else c for c in cols] + ["record_json"]
    rows = [[_cell(_get(r, c)) for c in cols] + [_json_cell(r, trim)] for r in recs]
    return [header, *rows]


async def build_tabs(db: Database, now: datetime | None = None) -> dict[str, list[list[Any]]]:
    now = now or datetime.now(timezone.utc)
    startups = await db.get_records("STARTUP")
    ai = [s for s in startups if s["content"]["data"].get("is_ai")]
    non_ai = [s for s in startups if not s["content"]["data"].get("is_ai")]
    chosen = ai if len(ai) >= MIN_STARTUPS else ai + non_ai[: MIN_STARTUPS - len(ai)]
    chosen.sort(key=lambda s: -(s["content"]["data"].get("employeeCount") or 0))

    all_products = await db.get_records("PRODUCT")
    # sheet = products whose pricing is grounded in a source (label / site / explicit text); model guesses stay in the DB, flagged
    products = sorted((p for p in all_products if p["content"].get("pricingModel")
                       and p["content"].get("pricingSource") not in ("llm_guess", None)),
                      key=lambda p: -(p["content"].get("votesCount") or 0))
    papers = sorted(await db.get_records("RESEARCH_PAPER"), key=lambda p: -(p["content"].get("github_stars") or -1))
    jobs = [j for j in await db.get_records("JOB") if is_within_hours(parse_date(j["content"]["date"]), 24, now)]
    news = [n for n in await db.get_records("NEWS") if is_within_hours(parse_date(n["content"]["published_date"]), 24, now)]
    news.sort(key=lambda n: n["content"]["published_date"], reverse=True)
    jobs.sort(key=lambda j: j["content"]["date"], reverse=True)
    log.info("export_selection", startups=len(chosen), startups_ai=sum(1 for s in chosen if s["content"]["data"].get("is_ai")),
             products=len(products), products_in_db=len(all_products), papers=len(papers), papers_with_stars=sum(1 for p in papers if p["content"]["github_stars"] is not None),
             jobs_24h=len(jobs), news_24h=len(news))

    entity = await db.get_records("ENTITY_LOG")
    entity.sort(key=lambda e: (e["method"] == "new", e["raw_name"].lower()))

    return {
        "Startups": _table(chosen, COMMON[:2] + ["content.entityName", "content.data.employeeCount", "content.data.is_ai",
                           "content.data.website", "content.data.oneLiner", "content.data.batch", "content.data.location",
                           "content.data.industry", "content.data.stage", "content.data.status", "content.data.isHiring"]
                           + COMMON[2:], trim=("content.data.description",)),
        "Products": _table(products, COMMON[:2] + ["content.startupName", "content.pricingModel", "content.pricingSource",
                           "content.pricingLabel", "content.productName", "content.tagline", "content.websiteUrl", "content.domain",
                           "content.votesCount", "content.launchedAt"] + COMMON[2:], trim=("content.description",)),
        "Research Papers": _table(papers, COMMON[:2] + ["content.title", "content.authors", "content.paper_url",
                                  "content.github_url", "content.github_stars", "content.published_date"] + COMMON[2:]),
        "Jobs": _table(jobs, COMMON[:2] + ["content.company", "content.date", "content.is_remote", "content.role_family",
                       "content.title", "content.location", "content.department"] + COMMON[2:]),
        "News": _table(news, COMMON[:2] + ["content.title", "content.url", "content.source", "content.published_date",
                       "content.summary", "content.entities", "content.full_text"] + COMMON[2:], trim=("content.full_text",)),
        "Entity Mapping Log": [["raw_name", "canonical_name", "method", "score", "timestamp"],
                               *[[e["raw_name"], e["canonical_name"], e["method"], e["score"], e["timestamp"]] for e in entity]],
    }


def write_csvs(tabs: dict[str, list[list[Any]]], out_dir: str = "data/export") -> list[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for name, rows in tabs.items():
        p = Path(out_dir) / f"{name.replace(' ', '_')}.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        paths.append(str(p))
    return paths


def write_sheet(tabs: dict[str, list[list[Any]]], sheet_id: str, creds_path: str, make_public: bool = True,
                gspread_client: Any = None) -> str:
    import gspread

    gc = gspread_client or gspread.service_account(filename=creds_path)
    sh = gc.open_by_key(sheet_id)
    existing = {ws.title: ws for ws in sh.worksheets()}
    for name, rows in tabs.items():
        ws = existing.get(name) or sh.add_worksheet(title=name, rows=max(len(rows), 2), cols=max(len(rows[0]), 2))
        ws.clear()
        ws.resize(rows=max(len(rows), 2), cols=max(len(rows[0]), 2))
        for i in range(0, len(rows), 2000):  # chunked writes stay under the API payload limit
            ws.update(values=rows[i:i + 2000], range_name=f"A{i + 1}", value_input_option="RAW")
        ws.freeze(rows=1)
        log.info("tab_written", tab=name, rows=len(rows) - 1)
    for ws in sh.worksheets():  # drop the default empty tab
        if ws.title not in tabs and ws.title.lower().startswith(("sheet", "blad")) and len(ws.get_all_values()) <= 1:
            sh.del_worksheet(ws)
    if make_public:
        sh.share(None, perm_type="anyone", role="reader")
    return sh.url


async def export(db: Database, sheet_id: str = "", creds_path: str = "", dry_run: bool = False,
                 now: datetime | None = None) -> dict[str, Any]:
    tabs = await build_tabs(db, now)
    summary: dict[str, Any] = {"rows": {k: len(v) - 1 for k, v in tabs.items()}}
    summary["csv"] = write_csvs(tabs)  # always keep a local copy
    if not dry_run:
        summary["sheet_url"] = write_sheet(tabs, sheet_id, creds_path)
    return summary
