import json
from datetime import date

import httpx
import pytest

from src.crawlers.papers import JOB, PapersCrawler, extract_github
from src.metrics import metrics
from src.storage.db import Database


def atom(entries, total=None):
    body = "".join(
        f"""<entry><id>http://arxiv.org/abs/{i}v1</id><title>{t}</title><published>2026-11-10T01:00:00Z</published>
        <summary>{summ}</summary><author><name>Ada L</name></author><author><name>Bob K</name></author>
        {f'<arxiv:comment>{c}</arxiv:comment>' if c else ''}</entry>"""
        for i, t, summ, c in entries)
    return (f'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" '
            f'xmlns:arxiv="http://arxiv.org/schemas/atom"><opensearch:totalResults>{total or len(entries)}</opensearch:totalResults>{body}</feed>')


ARX = [
    ("2611.00001", "Paper One", "We release code at https://github.com/foo/bar.", None),
    ("2611.00002", "Paper Two", "No code here.", None),
    ("2611.00003", "Paper Three", "Abstract.", "Code: github.com/old-org/oldname."),
    ("2611.00004", "Paper Four", "Abstract.", "see https://github.com/gone/deleted_repo.git"),
]


def handler_factory(token_ok=True):
    def h(req: httpx.Request):
        if req.url.host == "export.arxiv.org":
            return httpx.Response(200, text=atom(ARX))
        if req.url.host == "huggingface.co":
            if req.url.params.get("date") == "2026-11-10":
                return httpx.Response(200, json=[{"paper": {
                    "id": "2611.00002", "title": "Paper Two", "authors": [{"name": "Zed"}],
                    "publishedAt": "2026-11-10T00:00:00.000Z", "githubRepo": "https://github.com/hf/found", "githubStars": 42}}])
            return httpx.Response(200, json=[])
        if req.url.host == "api.github.com":
            q = json.loads(req.content)["query"]
            data = {}
            if "foo" in q:
                data["r0"] = {"nameWithOwner": "foo/bar", "stargazerCount": 123, "isArchived": False}
            # sorted order of unique repos: foo/bar, gone/deleted_repo, hf/found, old-org/oldname
            return httpx.Response(200, json={"data": {
                "r0": {"nameWithOwner": "foo/bar", "stargazerCount": 123},
                "r1": None,
                "r2": {"nameWithOwner": "hf/found", "stargazerCount": 50},
                "r3": {"nameWithOwner": "old-org/NewName", "stargazerCount": 7}}})
        return httpx.Response(404)
    return h


@pytest.fixture(autouse=True)
def _reset():
    metrics.reset()


def crawler(db, token="tok"):
    async def nosleep(_):
        pass
    return PapersCrawler(db, github_token=token, transport=httpx.MockTransport(handler_factory()),
                         per_domain_rps=0, host_rps={}, sleep=nosleep, max_retries=1)


@pytest.mark.parametrize("text,expected", [
    ("code at https://github.com/foo/bar.", "https://github.com/foo/bar"),
    ("github.com/foo/bar.git, and more", "https://github.com/foo/bar"),
    ("(https://github.com/Foo-1/bar_baz)", "https://github.com/Foo-1/bar_baz"),
    ("see github.com/topics/llm", None),
    ("https://github.com/orgonly", None),
    ("nothing", None),
])
def test_extract_github(text, expected):
    assert extract_github(text) == expected


async def test_crawl_keeps_only_github_papers_merges_hf_and_checkpoints():
    db = Database("sqlite:///:memory:")
    c = crawler(db)
    await c.run(target=1000, window_days=7, max_windows=1, today=date(2026, 11, 14))
    recs = {r["content"]["paper_url"].rsplit("/", 1)[-1]: r for r in await db.get_records("RESEARCH_PAPER")}
    assert set(recs) == {"2611.00001", "2611.00002", "2611.00003", "2611.00004"}  # 00002 gets repo from HF
    assert recs["2611.00002"]["content"]["github_url"] == "https://github.com/hf/found"
    assert recs["2611.00004"]["content"]["github_url"] == "https://github.com/gone/deleted_repo"
    r = recs["2611.00001"]
    assert r["recordType"] == "RESEARCH_PAPER" and r["content"]["authors"] == ["Ada L", "Bob K"]
    assert r["content"]["published_date"] == "2026-11-10" and r["content"]["github_stars"] is None
    assert (await db.get_checkpoint(JOB))["done_windows"] == ["2026-11-08:2026-11-14"]
    await c.run(max_windows=1, today=date(2026, 11, 14))  # window already done -> skipped
    assert metrics.get("papers.arxiv_fetched") == 4
    await c.aclose()


async def test_stars_graphql_handles_renamed_and_deleted_repos():
    db = Database("sqlite:///:memory:")
    c = crawler(db)
    await c.run(max_windows=1, today=date(2026, 11, 14))
    counts = await c.enrich_stars()
    assert counts["graphql"] == 3 and counts["missing"] == 1 and counts["renamed"] == 1
    recs = {r["content"]["paper_url"].rsplit("/", 1)[-1]: r["content"] for r in await db.get_records("RESEARCH_PAPER")}
    assert recs["2611.00001"]["github_stars"] == 123
    assert recs["2611.00004"]["github_stars"] is None                     # deleted -> null
    assert recs["2611.00003"]["github_url"] == "https://github.com/old-org/NewName"  # renamed -> new url
    assert {e["kind"] for e in c.repo_events} == {"repo_missing", "repo_renamed"}
    assert metrics.get("github.graphql_batches") == 1
    again = await c.enrich_stars()  # deleted repo is remembered and not re-queried
    assert again["missing"] == 0
    await c.aclose()


async def test_no_token_falls_back_to_hf_stars():
    db = Database("sqlite:///:memory:")
    c = crawler(db, token="")
    await c.run(max_windows=1, today=date(2026, 11, 14))
    counts = await c.enrich_stars()
    assert counts["hf_fallback"] == 1
    recs = {r["content"]["paper_url"].rsplit("/", 1)[-1]: r["content"] for r in await db.get_records("RESEARCH_PAPER")}
    assert recs["2611.00002"]["github_stars"] == 50 or recs["2611.00002"]["github_stars"] == 42
    await c.aclose()


def test_graphql_query_batches_100():
    q = PapersCrawler.build_query([("o", f"r{i}") for i in range(100)])
    assert q.count("repository(") == 100 and "r99:" in q
