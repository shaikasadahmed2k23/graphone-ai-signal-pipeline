import json

import httpx
import pytest

from src.crawlers.products import JOB, ProductsCrawler, clean_ph_url, rule_pricing
from src.llm.orchestrator import LLMOrchestrator, Provider
from src.metrics import metrics
from src.resolver.entity_resolver import EntityResolver
from src.storage.db import Database


class FakePricingLLM(Provider):
    name, max_input_tokens = "fake", 10_000

    def __init__(self):
        self.calls = 0

    async def generate(self, system, prompt, json_mode=True):
        self.calls += 1
        items = json.loads(prompt[prompt.rindex("\n\n[") + 2:])
        guess = "never return null" in prompt.lower()
        out = []
        for it in items:
            pm = "PAID" if "paid" in it["text"].lower() else ("FREEMIUM" if guess else None)
            out.append({"id": it["id"], "pricingModel": pm})
        return json.dumps({"results": out})


def node(i, name, desc, tagline="t"):
    return {"id": str(i), "name": name, "tagline": tagline, "description": desc, "slug": name.lower(),
            "url": f"https://www.producthunt.com/products/{name.lower()}?utm_source=x", "createdAt": "2026-11-20T08:00:00Z",
            "votesCount": 3, "topics": {"edges": [{"node": {"name": "Artificial Intelligence"}}]}}


NODES = [node(1, "Alpha", "An open source toolkit, free forever."),
         node(2, "Beta", "Subscription is paid only."),
         node(3, "Gamma", "Does stuff with AI agents.")]


def handler(req: httpx.Request):
    cursor = json.loads(req.content)["variables"]["c"]
    nodes = NODES if cursor is None else []
    return httpx.Response(200, json={"data": {"posts": {"pageInfo": {"endCursor": "c1", "hasNextPage": cursor is None},
                                                        "edges": [{"node": n} for n in nodes]}}},
                          headers={"x-rate-limit-remaining": "6000", "x-rate-limit-reset": "900"})


@pytest.fixture(autouse=True)
def _reset():
    metrics.reset()


def test_rule_pricing_is_conservative():
    assert rule_pricing("An open source toolkit, free forever.") == "FREE"
    assert rule_pricing("Open source core, with a pro plan at $10/mo") is None
    assert rule_pricing("Nice app") is None


def test_clean_ph_url_drops_tracking():
    assert clean_ph_url("https://www.producthunt.com/products/x?utm_source=y") == "https://www.producthunt.com/products/x"


async def test_products_pipeline_rule_llm_guess_and_resolver():
    db = Database("sqlite:///:memory:")
    llm = FakePricingLLM()
    orch = LLMOrchestrator([llm], cache=db)
    res = EntityResolver()
    res.add_entity("Alpha", [], "alpha.com")
    c = ProductsCrawler(db, "tok", res, orch, transport=httpx.MockTransport(handler), per_domain_rps=0)
    await c.run(target=3)
    recs = {r["content"]["productName"]: r for r in await db.get_records("PRODUCT")}
    assert recs["Alpha"]["content"]["pricingModel"] == "FREE" and recs["Alpha"]["content"]["pricingSource"] == "rule"
    assert recs["Beta"]["content"]["pricingModel"] == "PAID" and recs["Beta"]["content"]["pricingSource"] == "llm"
    assert recs["Gamma"]["content"]["pricingModel"] == "FREEMIUM" and recs["Gamma"]["content"]["pricingSource"] == "llm_guess"
    g = recs["Gamma"]
    assert g["recordType"] == "PRODUCT" and g["schemaVersion"] == "1.0" and g["source"]["name"] == "Product Hunt"
    assert g["source"]["url"] == "https://www.producthunt.com/products/gamma"
    assert g["content"]["startupName"] == "Gamma" and (await db.get_checkpoint(JOB))["fetched"] == 3
    assert llm.calls == 2  # 1 batch + 1 guess batch; rule-matched Alpha never hit the LLM
    await c.aclose()


async def test_llm_outage_leaves_pricing_null_but_keeps_record():
    class Down(Provider):
        name, max_input_tokens = "down", 10_000

        async def generate(self, *a, **k):
            from src.llm.orchestrator import ProviderUnavailable
            raise ProviderUnavailable("x")

    async def nosleep(_):
        pass

    db = Database("sqlite:///:memory:")
    c = ProductsCrawler(db, "tok", EntityResolver(), LLMOrchestrator([Down()], sleep=nosleep),
                        transport=httpx.MockTransport(handler), per_domain_rps=0)
    await c.run(target=3)
    recs = await db.get_records("PRODUCT")
    assert len(recs) == 3 and metrics.get("products.llm_failed_batches") >= 1
    await c.aclose()
