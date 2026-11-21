import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from src.crawlers.jobs import JobsCrawler, remote_from_text, role_family_rule
from src.metrics import metrics
from src.resolver.entity_resolver import EntityResolver
from src.storage.db import Database

NOW = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)


def iso(h):
    return (NOW - timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.mark.parametrize("title,expected", [
    ("Research Scientist, Alignment", "ML/AI Research"),
    ("Machine Learning Engineer", "ML/AI Engineering"),
    ("Senior Software Engineer, Backend", "Software Engineering"),
    ("Data Scientist", "Data"),
    ("Product Manager, Enterprise", "Product"),
    ("Account Executive, Startups", "Sales"),
    ("Recruiting Coordinator", "Recruiting/People"),
    ("Security Engineer", "Security"),
    ("Staff Counsel", "Finance/Legal"),
    ("Product Designer", "Design"),
    ("Wizard of Whimsy", None),
])
def test_role_family_rules(title, expected):
    assert role_family_rule(title) == expected


@pytest.mark.parametrize("loc,wp,expected", [
    ("Remote - US", None, True), ("San Francisco, CA", None, False), ("Anywhere", None, True),
    ("", None, None), ("Multiple Locations", None, None), ("London", "remote", True), ("NYC", "hybrid", False),
])
def test_remote_from_text(loc, wp, expected):
    assert remote_from_text(loc, wp) is expected


def handler(req: httpx.Request):
    h, p = req.url.host, req.url.path
    if "greenhouse" in h:
        return httpx.Response(200, json={"jobs": [
            {"id": 1, "title": "ML Engineer", "absolute_url": "https://x/1", "first_published": iso(3), "updated_at": iso(1),
             "location": {"name": "Remote"}, "departments": []},
            {"id": 2, "title": "Old Job", "absolute_url": "https://x/2", "first_published": iso(30), "updated_at": iso(1),
             "location": {"name": "SF"}, "departments": []},   # recently EDITED but published 30h ago -> must drop
            {"id": 3, "title": "Wizard of Whimsy", "absolute_url": "https://x/3", "first_published": iso(2),
             "location": {"name": "Multiple Locations"}, "departments": [{"name": "Misc"}]}]})
    if "ashby" in h:
        return httpx.Response(200, json={"jobs": [{"id": "a1", "title": "Account Executive", "jobUrl": "https://a/1",
                                                   "publishedAt": iso(5), "location": "NYC", "isRemote": False}]})
    if "lever" in h:
        return httpx.Response(200, json=[{"id": "l1", "text": "SRE", "hostedUrl": "https://l/1",
                                          "createdAt": int((NOW - timedelta(hours=2)).timestamp() * 1000),
                                          "categories": {"location": "Remote"}}])
    if "remoteok" in h:
        return httpx.Response(200, json=[{"legal": "x"}, {"id": 9, "position": "Backend Dev", "company": "OpenAI, Inc.",
                                                          "url": "https://r/9", "epoch": int((NOW - timedelta(hours=1)).timestamp())}])
    if "remotive" in h:
        return httpx.Response(200, json={"jobs": [{"id": 5, "title": "Designer", "company_name": "Foo", "url": "https://m/5",
                                                   "publication_date": iso(26).replace("Z", "")}]})
    if "weworkremotely" in h:
        return httpx.Response(200, text="""<?xml version="1.0"?><rss version="2.0"><channel><item>
            <title>Acme: Growth Marketer</title><link>https://w/1</link><guid>w1</guid>
            <pubDate>Sat, 21 Nov 2026 10:00:00 +0000</pubDate></item><item><title>Beta: Ops Lead</title><link>https://w/2</link>
            <guid>w2</guid><pubDate>Mon, 16 Nov 2026 10:00:00 +0000</pubDate></item></channel></rss>""")
    return httpx.Response(404)


class GapLLM:
    """Fake orchestrator: answers every gap with Operations / remote=True."""
    async def complete(self, task, system, text, instruction=""):
        rows = [{"id": r["id"], "role_family": "Operations", "is_remote": True} for r in json.loads(text)]

        class R:
            def json(self_inner):
                return {"results": rows}
        return R()


@pytest.fixture(autouse=True)
def _reset():
    metrics.reset()


async def test_strict_24h_filter_and_field_mapping(monkeypatch):
    import src.crawlers.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "BOARDS", [("greenhouse", "anthropic", "Anthropic", "anthropic.com"),
                                             ("ashby", "openai", "OpenAI", "openai.com"),
                                             ("lever", "palantir", "Palantir", "palantir.com")])
    db = Database("sqlite:///:memory:")
    c = JobsCrawler(db, EntityResolver(), GapLLM(), transport=httpx.MockTransport(handler), per_domain_rps=0)
    await c.run(now=NOW)
    recs = {r["source"]["url"]: r for r in await db.get_records("JOB")}
    assert "https://x/2" not in recs                         # published 30h ago (even though updated 1h ago)
    assert "https://m/5" not in recs and "https://w/2" not in recs   # Remotive 26h, WWR 5 days
    assert set(recs) == {"https://x/1", "https://x/3", "https://a/1", "https://l/1", "https://r/9", "https://w/1"}
    j = recs["https://x/1"]
    assert j["recordType"] == "JOB" and j["schemaVersion"] == "1.0" and j["content"]["company"] == "Anthropic"
    assert j["content"]["date"] == iso(3) and j["content"]["is_remote"] is True and j["content"]["role_family"] == "ML/AI Engineering"
    assert recs["https://a/1"]["content"]["is_remote"] is False
    assert recs["https://l/1"]["content"]["date"] == iso(2)                       # ms-epoch parsed
    assert recs["https://r/9"]["content"]["company"] == "OpenAI"                  # 'OpenAI, Inc.' canonicalized
    g = recs["https://x/3"]["content"]                                             # gaps filled by LLM
    assert g["role_family"] == "Operations" and g["is_remote"] is True
    assert metrics.get("jobs.stale_dropped") == 3 and metrics.get("jobs.role_llm") >= 1
    await c.aclose()


async def test_one_dead_source_does_not_kill_run(monkeypatch):
    import src.crawlers.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "BOARDS", [("greenhouse", "dead", "Dead", "dead.com")])

    def h(req):
        return httpx.Response(500) if "greenhouse" in req.url.host else handler(req)

    async def nosleep(_):
        pass

    db = Database("sqlite:///:memory:")
    c = JobsCrawler(db, EntityResolver(), None, transport=httpx.MockTransport(h), per_domain_rps=0, max_retries=1, sleep=nosleep)
    await c.run(now=NOW)
    assert metrics.get("jobs.source_failed") == 1 and await db.count("JOB") >= 2
    await c.aclose()
