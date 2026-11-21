import json

import httpx
import pytest

from src.crawlers.startups import JOB, YCStartupsCrawler
from src.metrics import metrics
from src.storage.db import Database

PAGE = '<script>window.AlgoliaOpts = {"app":"APP","key":"KEY","valid_until":4102444800}</script>'


def hit(i, batch, **kw):
    return {"id": i, "name": f"Co{i}", "slug": f"co{i}", "batch": batch, "team_size": 10 + i,
            "website": f"http://co{i}.com", "launched_at": 1326790856, **kw}


def make_handler(fail_batch=None, calls=None):
    data = {"W22": [hit(1, "W22"), hit(2, "W22")], "S21": [hit(3, "S21")]}

    def h(req: httpx.Request):
        if "ycombinator.com" in req.url.host:
            return httpx.Response(200, text=PAGE)
        body = json.loads(req.content)
        params = body["requests"][0]["params"]
        if calls is not None:
            calls.append(params)
        if "facets" in params:
            return httpx.Response(200, json={"results": [{"hits": [], "facets": {"batch": {"W22": 2, "S21": 1}}}]})
        batch = [b for b in data if b in params][0]
        if batch == fail_batch:
            return httpx.Response(500)
        return httpx.Response(200, json={"results": [{"hits": data[batch], "nbPages": 1}]})

    return h


@pytest.fixture(autouse=True)
def _reset():
    metrics.reset()


async def _crawler(db, handler):
    async def nosleep(_):
        pass

    return YCStartupsCrawler(db, transport=httpx.MockTransport(handler), per_domain_rps=0, sleep=nosleep, max_retries=1)


async def test_full_run_maps_schema_and_is_idempotent():
    db = Database("sqlite:///:memory:")
    c = await _crawler(db, make_handler())
    s1 = await c.run()
    assert s1["inserted"] == 3 and await db.count("STARTUP") == 3
    rec = (await db.get_records("STARTUP"))[0]
    assert rec["recordType"] == "STARTUP" and rec["schemaVersion"] == "1.0"
    assert rec["content"]["entityName"].startswith("Co") and rec["content"]["data"]["employeeCount"] >= 11
    assert rec["source"]["name"] == "YC Directory" and rec["collectedAt"].endswith("Z")
    await db.clear_checkpoint(JOB)
    s2 = await c.run()  # re-run: no duplicates, unchanged (collectedAt excluded from hash)
    assert s2["inserted"] == 0 and s2["unchanged"] == 3 and await db.count("STARTUP") == 3
    await c.aclose()


async def test_crash_resumes_from_checkpoint():
    db = Database("sqlite:///:memory:")
    c1 = await _crawler(db, make_handler(fail_batch="S21"))  # W22 (bigger) runs first, S21 crashes
    with pytest.raises(Exception):
        await c1.run()
    assert (await db.get_checkpoint(JOB))["done_batches"] == ["W22"]
    calls = []
    c2 = await _crawler(db, make_handler(calls=calls))
    await c2.run()
    assert not any("W22" in p for p in calls if "facets" not in p)  # W22 not re-fetched
    assert await db.count("STARTUP") == 3
    await c1.aclose(); await c2.aclose()
