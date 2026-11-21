"""Jobs: Greenhouse / Lever / Ashby public board APIs (AI companies) + RemoteOK + Remotive + We Work Remotely.

Strict freshness: only jobs whose normalized UTC publish time is within the last 24h are stored.
  Greenhouse -> first_published (NOT updated_at: edits would let old jobs in)   Ashby -> publishedAt
  Lever      -> createdAt (ms epoch)       RemoteOK -> epoch     Remotive -> publication_date    WWR -> pubDate
Deterministic first: is_remote from structured fields / location text, role_family from title keywords.
LLM only for gaps: is_remote when location is ambiguous, role_family when no keyword matches.
Company is canonicalized through the entity resolver (board config supplies the known domain).
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Any

import feedparser

from src.crawlers.base_crawler import BaseCrawler, FetchError
from src.freshness.date_utils import is_within_hours, parse_date, to_iso
from src.llm.orchestrator import AllProvidersFailed, LLMOrchestrator
from src.logging_setup import get_logger
from src.metrics import metrics
from src.models import JobContent, JobRecord, Source, to_payload
from src.resolver.entity_resolver import EntityResolver
from src.storage.db import Database

log = get_logger("jobs")

# (platform, board slug, display name, domain). Slugs verified live on 2026-11-21.
BOARDS: list[tuple[str, str, str, str]] = [
    ("greenhouse", "anthropic", "Anthropic", "anthropic.com"),
    ("greenhouse", "databricks", "Databricks", "databricks.com"),
    ("greenhouse", "scaleai", "Scale AI", "scale.com"),
    ("greenhouse", "xai", "xAI", "x.ai"),
    ("greenhouse", "stabilityai", "Stability AI", "stability.ai"),
    ("ashby", "openai", "OpenAI", "openai.com"),
    ("ashby", "cohere", "Cohere", "cohere.com"),
    ("ashby", "perplexity", "Perplexity", "perplexity.ai"),
    ("ashby", "elevenlabs", "ElevenLabs", "elevenlabs.io"),
    ("ashby", "cursor", "Anysphere", "cursor.com"),
    ("ashby", "cognition", "Cognition", "cognition.ai"),
    ("ashby", "harvey", "Harvey", "harvey.ai"),
    ("ashby", "sierra", "Sierra", "sierra.ai"),
    ("ashby", "langchain", "LangChain", "langchain.com"),
    ("ashby", "modal", "Modal", "modal.com"),
    ("ashby", "pinecone", "Pinecone", "pinecone.io"),
    ("lever", "palantir", "Palantir", "palantir.com"),
]

ROLE_FAMILIES = ["ML/AI Research", "ML/AI Engineering", "Software Engineering", "Data", "Product", "Design", "Sales",
                 "Marketing", "Customer Success", "Operations", "Recruiting/People", "Finance/Legal", "Security", "Other"]
# order matters: first match wins
_ROLE_RULES: list[tuple[str, re.Pattern]] = [(fam, re.compile(pat, re.I)) for fam, pat in [
    ("ML/AI Research", r"\b(research(er)? scientist|research scientist|research engineer|applied scientist|research lead|member of technical staff.*research|researcher)\b"),
    ("ML/AI Engineering", r"\b(machine learning|ml|ai|llm|deep learning|nlp|computer vision|mlops|inference|model|gpu|cuda|training)\b.*\b(engineer|developer|infra|platform)\b|\b(engineer|developer)\b.*\b(machine learning|ml|llm|inference|gpu|cuda)\b"),
    ("Security", r"\b(security|infosec|appsec|trust and safety|trust & safety|threat|red team)\b"),
    ("Data", r"\b(data (scientist|analyst|engineer)|analytics|business intelligence|bi analyst|data platform)\b"),
    ("Software Engineering", r"\b(software|backend|back-end|frontend|front-end|full[- ]?stack|platform engineer|infrastructure|devops|sre|site reliability|mobile|ios|android|engineer|developer|swe|architect)\b"),
    ("Product", r"\b(product (manager|lead|owner|marketing manager)|program manager|technical program|head of product|pm)\b"),
    ("Design", r"\b(designer|design lead|ux|ui|brand design|creative)\b"),
    ("Sales", r"\b(sales|account (executive|manager)|business development|bdr|sdr|solutions? (engineer|architect)|partnerships?|go[- ]to[- ]market|gtm|revenue)\b"),
    ("Marketing", r"\b(marketing|growth|content|communications|comms|seo|community|developer relations|devrel|public relations)\b"),
    ("Customer Success", r"\b(customer (success|support|experience)|support (engineer|specialist)|technical support|implementation)\b"),
    ("Recruiting/People", r"\b(recruit\w*|talent|people (partner|operations|ops)|hr\b|human resources|sourcer)\b"),
    ("Finance/Legal", r"\b(finance|accountant|accounting|controller|tax|legal|counsel|attorney|paralegal|compliance|procurement|payroll|fp&a)\b"),
    ("Operations", r"\b(operations|ops\b|chief of staff|administrator|executive assistant|office|facilities|workplace|logistics|supply chain|strategy)\b"),
]]


def role_family_rule(title: str, dept: str = "") -> str | None:
    t = f"{title} {dept}".strip()
    for fam, pat in _ROLE_RULES:
        if pat.search(title) or (dept and fam in ("Sales", "Marketing", "Finance/Legal", "Recruiting/People", "Design") and pat.search(dept)):
            return fam
    return None


_REMOTE_YES = re.compile(r"\bremote\b|\banywhere\b|work from home|\bwfh\b|distributed", re.I)
_REMOTE_NO_ONSITE = re.compile(r"\bon[- ]?site\b|\bin[- ]office\b", re.I)


def remote_from_text(location: str, workplace: str | None = None) -> bool | None:
    """True/False when the text is decisive, None when ambiguous (-> LLM)."""
    wp = (workplace or "").lower()
    if wp in ("remote",):
        return True
    if wp in ("onsite", "on-site", "on_site"):
        return False
    if wp == "hybrid":
        return False
    loc = location or ""
    if _REMOTE_YES.search(loc):
        return True
    if _REMOTE_NO_ONSITE.search(loc):
        return False
    if re.search(r"[A-Za-z]{3,}", loc) and not re.search(r"multiple|various|flexible|hybrid", loc, re.I):
        return False  # a concrete city/office with no remote mention
    return None


class JobsCrawler(BaseCrawler):
    def __init__(self, db: Database, resolver: EntityResolver, llm: LLMOrchestrator | None, **kw: Any) -> None:
        super().__init__(**kw)
        self.db, self.resolver, self.llm = db, resolver, llm

    # ---- source adapters -> normalized dicts ------------------------------------------
    async def _json(self, url: str, **kw: Any) -> Any:
        res = await self.fetch(url, escalate=False, **kw)
        if not res.ok:
            raise FetchError(f"{url}: {res.status}")
        return res.json()

    async def fetch_greenhouse(self, slug: str, name: str, domain: str) -> list[dict]:
        data = await self._json(f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs", params={"content": "true"})
        out = []
        for j in data.get("jobs", []):
            loc = (j.get("location") or {}).get("name", "")
            dept = ", ".join(d.get("name", "") for d in j.get("departments") or [])
            out.append({"source": "Greenhouse", "id": str(j["id"]), "title": j.get("title", ""), "company": name, "domain": domain,
                        "url": j.get("absolute_url", ""), "date_raw": j.get("first_published"), "location": loc,
                        "remote": remote_from_text(loc), "dept": dept})
        return out

    async def fetch_ashby(self, slug: str, name: str, domain: str) -> list[dict]:
        data = await self._json(f"https://api.ashbyhq.com/posting-api/job-board/{slug}")
        out = []
        for j in data.get("jobs", []):
            if j.get("isListed") is False:
                continue
            loc = j.get("location") or ""
            remote = j.get("isRemote") if isinstance(j.get("isRemote"), bool) else remote_from_text(loc, j.get("workplaceType"))
            out.append({"source": "Ashby", "id": str(j["id"]), "title": j.get("title", ""), "company": name, "domain": domain,
                        "url": j.get("jobUrl", ""), "date_raw": j.get("publishedAt"), "location": loc, "remote": remote,
                        "dept": f"{j.get('department') or ''} {j.get('team') or ''}".strip()})
        return out

    async def fetch_lever(self, slug: str, name: str, domain: str) -> list[dict]:
        data = await self._json(f"https://api.lever.co/v0/postings/{slug}", params={"mode": "json"})
        out = []
        for j in data:
            cat = j.get("categories") or {}
            loc = cat.get("location") or ""
            out.append({"source": "Lever", "id": str(j["id"]), "title": j.get("text", ""), "company": name, "domain": domain,
                        "url": j.get("hostedUrl", ""), "date_raw": j.get("createdAt"), "location": loc,
                        "remote": remote_from_text(loc, j.get("workplaceType")), "dept": cat.get("team") or ""})
        return out

    async def fetch_remoteok(self) -> list[dict]:
        data = await self._json("https://remoteok.com/api")
        return [{"source": "RemoteOK", "id": str(j["id"]), "title": j.get("position", ""), "company": j.get("company", ""),
                 "domain": None, "url": j.get("url") or j.get("apply_url", ""), "date_raw": j.get("epoch") or j.get("date"),
                 "location": j.get("location") or "Remote", "remote": True, "dept": " ".join(j.get("tags") or [])}
                for j in data if isinstance(j, dict) and j.get("id")]

    async def fetch_remotive(self) -> list[dict]:  # API allows ~4 requests/day: exactly one per run
        data = await self._json("https://remotive.com/api/remote-jobs")
        return [{"source": "Remotive", "id": str(j["id"]), "title": j.get("title", ""), "company": j.get("company_name", ""),
                 "domain": None, "url": j.get("url", ""), "date_raw": j.get("publication_date"),
                 "location": j.get("candidate_required_location") or "Remote", "remote": True,
                 "dept": f"{j.get('category') or ''} {' '.join(j.get('tags') or [])}"} for j in data.get("jobs", [])]

    async def fetch_wwr(self) -> list[dict]:
        res = await self.fetch("https://weworkremotely.com/remote-jobs.rss", escalate=False)
        if not res.ok:
            raise FetchError(f"wwr {res.status}")
        feed = await asyncio.to_thread(feedparser.parse, res.text)
        out = []
        for e in feed.entries:
            title = e.get("title", "")
            company, _, role = title.partition(": ")  # WWR titles are "Company: Role"
            out.append({"source": "We Work Remotely", "id": e.get("id") or e.get("link", ""), "title": role or title,
                        "company": company if role else "", "domain": None, "url": e.get("link", ""),
                        "date_raw": e.get("published"), "location": e.get("region") or "Remote", "remote": True,
                        "dept": e.get("category", "")})
        return out

    # ---- run ------------------------------------------------------------------------------
    async def collect(self) -> list[dict]:
        tasks: dict[str, Any] = {}
        for plat, slug, name, dom in BOARDS:
            tasks[f"{plat}:{slug}"] = getattr(self, f"fetch_{plat}")(slug, name, dom)
        tasks.update({"remoteok": self.fetch_remoteok(), "remotive": self.fetch_remotive(), "wwr": self.fetch_wwr()})
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        jobs: list[dict] = []
        for (name, _), res in zip(tasks.items(), results):
            if isinstance(res, Exception):  # one dead board never kills the run
                metrics.inc("jobs.source_failed")
                log.warning("job_source_failed", source=name, error=str(res)[:150])
                continue
            metrics.inc("jobs.fetched", len(res))
            log.info("job_source_ok", source=name, n=len(res))
            jobs += res
        return jobs

    async def run(self, now: datetime | None = None, hours: float = 24) -> dict[str, int]:
        now = now or datetime.now(timezone.utc)
        jobs = await self.collect()
        fresh = []
        for j in jobs:
            dt = parse_date(j["date_raw"], now)
            if dt is None:
                metrics.inc("jobs.no_date_dropped")  # undated = cannot prove freshness -> excluded
                continue
            if not is_within_hours(dt, hours, now):
                metrics.inc("jobs.stale_dropped")
                continue
            j["dt"] = dt
            fresh.append(j)
        log.info("jobs_fresh", fresh=len(fresh), total=len(jobs))
        await self._fill_gaps(fresh)
        items = []
        for j in fresh:
            company = self.resolver.resolve(j["company"], domain=j["domain"], context=j["dept"] + " " + j["title"]) if j["company"] else "Unknown"
            rec = JobRecord(
                source=Source(name=j["source"], url=j["url"]),
                content=JobContent(company=company, date=to_iso(j["dt"]), is_remote=j["remote"], role_family=j["role_family"],
                                   title=j["title"], location=j["location"], companyRaw=j["company"],
                                   department=j["dept"] or None, jobId=j["id"]),
            )
            items.append(("JOB", f"{j['source']}:{j['id']}", to_payload(rec), j["source"]))
            metrics.inc("jobs.parsed")
        stats = await self.db.upsert_many(items)
        metrics.inc("dedup.hits", stats["unchanged"])
        return stats

    # ---- LLM gap fill ------------------------------------------------------------------
    async def _fill_gaps(self, jobs: list[dict]) -> None:
        need_role, need_remote = [], []
        for j in jobs:
            j["role_family"] = role_family_rule(j["title"], j["dept"])
            if j["role_family"] is None:
                need_role.append(j)
            if j["remote"] is None:
                need_remote.append(j)
        metrics.inc("jobs.role_rule", len(jobs) - len(need_role))
        metrics.inc("jobs.remote_structured", len(jobs) - len(need_remote))
        if (need_role or need_remote) and self.llm is not None:
            pend = {id(j): j for j in need_role + need_remote}
            lst = list(pend.values())
            await asyncio.gather(*[self._llm_batch(lst[i:i + 15]) for i in range(0, len(lst), 15)])
        for j in need_role:  # LLM unavailable / failed -> honest fallback label
            j["role_family"] = j["role_family"] or "Other"
        # remote stays None if unknown (never invent)

    async def _llm_batch(self, batch: list[dict]) -> None:
        payload = json.dumps([{"id": i, "title": j["title"], "department": j["dept"][:80], "location": j["location"][:80],
                               "need_role": j["role_family"] is None, "need_remote": j["remote"] is None}
                              for i, j in enumerate(batch)], ensure_ascii=False)
        instr = ("For each job fill only the requested fields. role_family must be exactly one of: "
                 + ", ".join(ROLE_FAMILIES) + ". is_remote is true only if the job can be done fully remotely, "
                 "false if it needs an office/hybrid, null if truly unknowable. "
                 'Return {"results":[{"id":<int>,"role_family":"..."|null,"is_remote":true|false|null}]}.\n\nJobs (JSON):')
        try:
            res = await self.llm.complete("job_gaps_v1", "You label job postings. Reply ONLY with JSON.", payload, instr)
            rows = res.json()["results"]
        except (AllProvidersFailed, ValueError, KeyError, TypeError) as e:
            metrics.inc("jobs.llm_failed_batches")
            log.warning("job_llm_failed", error=str(e)[:150])
            return
        for r in rows:
            try:
                j = batch[int(r["id"])]
            except (KeyError, ValueError, IndexError, TypeError):
                continue
            if j["role_family"] is None and r.get("role_family") in ROLE_FAMILIES:
                j["role_family"] = r["role_family"]
                metrics.inc("jobs.role_llm")
            if j["remote"] is None and isinstance(r.get("is_remote"), bool):
                j["remote"] = r["is_remote"]
                metrics.inc("jobs.remote_llm")
