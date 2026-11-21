"""Research papers with GitHub repos + current star counts.

Sources
  * arXiv API (cs.AI / cs.LG / cs.CL), fetched in 7-day submittedDate windows, newest first, until
    `target` papers with a GitHub URL are stored.  Repo URL is mined from abstract + comment + journal_ref.
  * Hugging Face daily_papers (githubRepo/githubStars per paper) - adds repos arXiv text lacks and
    papers from other categories; its star count is the fallback when no GitHub token is available.
Stars
  * GitHub GraphQL, 100 repos per query (aliased `repository()` fields; 1 rate-limit point per query).
  * renamed repo -> GitHub redirects, we store the new URL and log it
  * deleted/private repo -> github_stars = null, logged
Checkpoint = windows already completed, so a crash resumes at the next window.
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

import feedparser

from src.crawlers.base_crawler import BaseCrawler, FetchError
from src.logging_setup import get_logger
from src.metrics import metrics
from src.models import PaperContent, PaperRecord, Source, to_payload
from src.storage.db import Database

log = get_logger("papers")

ARXIV_API = "https://export.arxiv.org/api/query"
HF_DAILY = "https://huggingface.co/api/daily_papers"
GH_GRAPHQL = "https://api.github.com/graphql"
JOB = "papers:arxiv"
CATS = "(cat:cs.AI OR cat:cs.LG OR cat:cs.CL)"
PAGE = 2000  # arXiv API max per call

_GH_RE = re.compile(r"github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/([A-Za-z0-9._-]+)", re.I)
_NOT_REPO_OWNERS = {"topics", "features", "orgs", "sponsors", "settings", "about", "marketplace", "login", "pricing",
                    "collections", "apps", "enterprise", "search", "pulls", "issues", "site", "readme", "blog"}


def extract_github(*texts: str | None) -> str | None:
    """First plausible https://github.com/owner/repo in the given texts (priority = argument order)."""
    for t in texts:
        for m in _GH_RE.finditer(t or ""):
            owner, repo = m.group(1), m.group(2)
            repo = re.sub(r"(\.git)?[.,;:)\]}>\"']*$", "", repo)  # trailing punctuation / .git
            repo = repo.rstrip(".")
            if owner.lower() in _NOT_REPO_OWNERS or not repo or len(repo) > 100:
                continue
            return f"https://github.com/{owner}/{repo}"
    return None


def arxiv_id_of(entry_id: str) -> str:
    return re.sub(r"v\d+$", "", entry_id.rsplit("/abs/", 1)[-1])


def _day(d: date) -> str:
    return d.strftime("%Y%m%d")


class PapersCrawler(BaseCrawler):
    def __init__(self, db: Database, github_token: str = "", **kw: Any) -> None:
        kw.setdefault("host_rps", {"export.arxiv.org": 0.33})  # arXiv ToS: <= 1 request / 3 s
        super().__init__(**kw)
        self.db, self.token = db, github_token
        self.repo_events: list[dict] = []  # renamed / missing repos, for troubleshooting + the run log

    # ---- sources ---------------------------------------------------------------
    async def fetch_arxiv_window(self, start: date, end: date) -> list[dict]:
        """All cs.AI/LG/CL papers submitted in [start, end] (inclusive days)."""
        q = f"{CATS} AND submittedDate:[{_day(start)}0000 TO {_day(end)}2359]"
        out, offset = [], 0
        while True:
            res = await self.fetch(ARXIV_API, params={"search_query": q, "start": offset, "max_results": PAGE,
                                                      "sortBy": "submittedDate", "sortOrder": "descending"},
                                   escalate=False)
            if not res.ok:
                raise FetchError(f"arxiv {res.status}")
            feed = await asyncio.to_thread(feedparser.parse, res.text)
            batch = [self.parse_arxiv_entry(e) for e in feed.entries]
            batch = [b for b in batch if b]
            metrics.inc("papers.arxiv_fetched", len(feed.entries))
            out += batch
            total = int(feed.feed.get("opensearch_totalresults", 0) or 0)
            offset += PAGE
            if not feed.entries or offset >= total:
                break
        return out

    @staticmethod
    def parse_arxiv_entry(e: Any) -> dict | None:
        title = re.sub(r"\s+", " ", e.get("title", "")).strip()
        if not title or not e.get("id"):
            return None
        aid = arxiv_id_of(e["id"])
        gh = extract_github(e.get("arxiv_comment"), e.get("summary"), e.get("arxiv_journal_ref"),
                            *[l.get("href") for l in e.get("links", [])])
        return {
            "id": aid, "title": title, "authors": [a.get("name", "").strip() for a in e.get("authors", []) if a.get("name")],
            "published": (e.get("published") or "")[:10], "github_url": gh, "hf_stars": None, "origin": "arxiv",
        }

    async def fetch_hf_day(self, d: date) -> list[dict]:
        res = await self.fetch(HF_DAILY, params={"date": d.isoformat(), "limit": 100}, escalate=False)
        if not res.ok:
            return []
        out = []
        for item in res.json():
            p = item.get("paper") or {}
            if not p.get("id") or not p.get("title"):
                continue
            out.append({
                "id": p["id"], "title": re.sub(r"\s+", " ", p["title"]).strip(),
                "authors": [a["name"] for a in p.get("authors", []) if a.get("name")],
                "published": (p.get("publishedAt") or "")[:10],
                "github_url": extract_github(p.get("githubRepo"), p.get("summary")),
                "hf_stars": p.get("githubStars") if isinstance(p.get("githubStars"), int) else None,
                "origin": "hf",
            })
        metrics.inc("papers.hf_fetched", len(out))
        return out

    # ---- crawl -----------------------------------------------------------------
    async def run(self, target: int = 1000, window_days: int = 7, max_windows: int = 16,
                  today: date | None = None) -> dict[str, int]:
        ckpt = await self.db.get_checkpoint(JOB)
        done: list[str] = ckpt.get("done_windows", [])
        end = (today or datetime.now(timezone.utc).date())
        stats = {"inserted": 0, "updated": 0, "unchanged": 0}
        for i in range(max_windows):
            have = await self.db.count("RESEARCH_PAPER")
            if have >= target:
                log.info("papers_target_reached", have=have)
                break
            w_end = end - timedelta(days=i * window_days)
            w_start = w_end - timedelta(days=window_days - 1)
            wkey = f"{w_start}:{w_end}"
            if wkey in done:
                continue
            arx = await self.fetch_arxiv_window(w_start, w_end)
            hf: list[dict] = []
            days = [w_start + timedelta(days=k) for k in range(window_days)]
            for chunk in await asyncio.gather(*[self.fetch_hf_day(d) for d in days], return_exceptions=True):
                if isinstance(chunk, Exception):
                    log.warning("hf_day_failed", error=repr(chunk))  # HF is a bonus source, never fatal
                else:
                    hf += chunk
            papers = self.merge(arx, hf)
            items = []
            for p in papers:
                if not p["github_url"]:
                    metrics.inc("papers.no_github")
                    continue
                rec = self.to_record(p)
                if rec:
                    metrics.inc("papers.parsed")
                    items.append(("RESEARCH_PAPER", p["id"], to_payload(rec), "arXiv"))
            s = await self.db.upsert_many(items)
            for k, v in s.items():
                stats[k] += v
            metrics.inc("dedup.hits", s["unchanged"])
            done.append(wkey)
            ckpt = {**ckpt, "done_windows": done, "hf_stars": {**ckpt.get("hf_stars", {}), **{
                p["id"]: p["hf_stars"] for p in papers if p["hf_stars"] is not None}}}
            await self.db.set_checkpoint(JOB, ckpt)
            log.info("papers_window_done", window=wkey, arxiv=len(arx), hf=len(hf), with_github=len(items),
                     total_with_github=await self.db.count("RESEARCH_PAPER"))
        return stats

    @staticmethod
    def merge(arxiv: list[dict], hf: list[dict]) -> list[dict]:
        by_id = {p["id"]: p for p in arxiv}
        for h in hf:
            cur = by_id.get(h["id"])
            if cur is None:
                by_id[h["id"]] = h
                continue
            cur["github_url"] = cur["github_url"] or h["github_url"]
            cur["hf_stars"] = h["hf_stars"]
        return list(by_id.values())

    @staticmethod
    def to_record(p: dict) -> PaperRecord | None:
        if not p["published"]:
            return None
        return PaperRecord(
            source=Source(name="arXiv", url=f"https://arxiv.org/abs/{p['id']}"),
            content=PaperContent(title=p["title"], authors=p["authors"], paper_url=f"https://arxiv.org/abs/{p['id']}",
                                 github_url=p["github_url"], github_stars=None, published_date=p["published"]),
        )

    # ---- stars -----------------------------------------------------------------
    @staticmethod
    def build_query(repos: list[tuple[str, str]]) -> str:
        parts = [f'r{i}: repository(owner: "{o}", name: "{n}") {{ nameWithOwner stargazerCount isArchived }}'
                 for i, (o, n) in enumerate(repos)]
        return "query { " + " ".join(parts) + " }"

    async def graphql_stars(self, repos: list[tuple[str, str]]) -> dict[tuple[str, str], dict | None]:
        """One query for <=100 repos -> {(owner,name): {"nameWithOwner","stars"} | None if missing}."""
        res = await self.fetch(GH_GRAPHQL, method="POST", json={"query": self.build_query(repos)}, escalate=False,
                               headers={"Authorization": f"Bearer {self.token}"})
        if res.status == 401:
            raise FetchError("GitHub token rejected (401)")
        body = res.json()
        if "data" not in body or body["data"] is None:
            raise FetchError(f"GraphQL error: {str(body.get('errors'))[:200]}")
        out = {}
        for i, key in enumerate(repos):
            node = body["data"].get(f"r{i}")
            out[key] = {"nameWithOwner": node["nameWithOwner"], "stars": node["stargazerCount"]} if node else None
        return out

    async def enrich_stars(self) -> dict[str, int]:
        """Fill github_stars for stored papers. Uses GraphQL if a token exists, else HF's githubStars."""
        recs = await self.db.get_records("RESEARCH_PAPER")
        todo = [r for r in recs if r["content"]["github_stars"] is None]
        ckpt = await self.db.get_checkpoint(JOB)
        hf_stars: dict[str, int] = ckpt.get("hf_stars", {})
        dead: set[str] = set(ckpt.get("dead_repos", []))
        counts = {"graphql": 0, "hf_fallback": 0, "missing": 0, "renamed": 0}
        parsed = {}
        for r in todo:
            m = _GH_RE.search(r["content"]["github_url"] or "")
            if m:
                parsed[r["content"]["paper_url"]] = (r, (m.group(1), re.sub(r"\.git$", "", m.group(2))))
        def _save(rec: dict) -> tuple:
            return ("RESEARCH_PAPER", rec["content"]["paper_url"].rsplit("/abs/", 1)[-1], rec, "arXiv")
        updates = []
        if self.token:
            unique = sorted({k for _, k in parsed.values() if f"{k[0]}/{k[1]}".lower() not in dead})
            results: dict[tuple[str, str], dict | None] = {}
            for i in range(0, len(unique), 100):
                batch = unique[i:i + 100]
                try:
                    results.update(await self.graphql_stars(batch))
                    metrics.inc("github.graphql_batches")
                except FetchError as e:
                    log.error("graphql_batch_failed", error=str(e)[:200], batch_start=i)
            for url, (rec, key) in parsed.items():
                if key not in results:
                    continue
                info = results[key]
                if info is None:
                    counts["missing"] += 1
                    metrics.inc("github.repo_missing")
                    dead.add(f"{key[0]}/{key[1]}".lower())
                    ev = {"kind": "repo_missing", "paper": url, "repo": f"{key[0]}/{key[1]}"}
                    self.repo_events.append(ev)
                    log.warning("repo_missing_or_deleted", **ev)
                    continue
                rec["content"]["github_stars"] = info["stars"]
                if info["nameWithOwner"].lower() != f"{key[0]}/{key[1]}".lower():
                    counts["renamed"] += 1
                    metrics.inc("github.repo_renamed")
                    rec["content"]["github_url"] = f"https://github.com/{info['nameWithOwner']}"
                    ev = {"kind": "repo_renamed", "paper": url, "from": f"{key[0]}/{key[1]}", "to": info["nameWithOwner"]}
                    self.repo_events.append(ev)
                    log.info("repo_renamed", **ev)
                counts["graphql"] += 1
                updates.append(_save(rec))
        else:
            log.warning("no_github_token_using_hf_star_fallback")
            for url, (rec, key) in parsed.items():
                aid = url.rsplit("/abs/", 1)[-1]
                if aid in hf_stars:
                    rec["content"]["github_stars"] = hf_stars[aid]
                    counts["hf_fallback"] += 1
                    updates.append(_save(rec))
        if updates:
            await self.db.upsert_many(updates)
        await self.db.set_checkpoint(JOB, {**ckpt, "dead_repos": sorted(dead)})
        metrics.inc("papers.stars_filled", counts["graphql"] + counts["hf_fallback"])
        log.info("stars_enriched", **counts, total=len(todo))
        return counts
