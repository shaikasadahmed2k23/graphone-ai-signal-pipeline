import json

from src.crawlers.news import NEWS_BATCH, NewsCrawler
from src.llm.orchestrator import AllProvidersFailed, LLMOrchestrator, Provider
from src.resolver.entity_resolver import EntityResolver
from src.storage.db import Database

TEXT = "OpenAI announced a new model today. " * 20


def article(i, summary=None):
    return ("NEWS", f"k{i}", {"schemaVersion": "1.0", "recordType": "NEWS", "collectedAt": "2026-11-21T00:00:00Z",
                              "source": {"name": "HN", "url": f"u{i}"},
                              "content": {"title": f"T{i}", "url": f"u{i}", "source": "HN",
                                          "published_date": "2026-11-21T00:00:00Z", "full_text": TEXT,
                                          "summary": summary, "entities": []}}, "HN")


class BatchProvider(Provider):
    name, max_input_tokens = "fake", 100_000

    def __init__(self, skip_ids=()):
        self.calls, self.skip = [], set(skip_ids)

    async def generate(self, system, prompt, json_mode=True):
        items, _ = json.JSONDecoder().raw_decode(prompt[prompt.rindex('[{"id": "k'):])
        self.calls.append(len(items))
        return json.dumps({"results": [{"id": it["id"], "summary": f"sum {it['title']}", "entities": ["OpenAI"]}
                                       for it in items if it["id"] not in self.skip]})


class Down(Provider):
    name, max_input_tokens = "down", 100_000

    async def generate(self, *a, **k):
        from src.llm.orchestrator import ProviderUnavailable
        raise ProviderUnavailable("down")


async def nosleep(_):
    return None


async def crawler(llm):
    db = Database("sqlite:///:memory:")
    await db.upsert_many([article(i) for i in range(35)] + [article(99, summary="already")])
    return db, NewsCrawler(db, EntityResolver(), llm, per_domain_rps=0, sleep=nosleep)


async def test_backfill_batches_15_per_call_and_fills_everything():
    prov = BatchProvider()
    db, c = await crawler(LLMOrchestrator([prov], sleep=nosleep))
    left = await c.backfill_structure(rounds=1)
    assert left == 0
    assert sorted(prov.calls) == [5, 15, 15]  # 35 articles -> 3 calls, the already-summarised one is untouched
    assert max(prov.calls) == NEWS_BATCH
    rows = dict(await db.get_keyed("NEWS"))
    assert rows["k3"]["content"]["summary"] == "sum T3" and rows["k3"]["content"]["entities"] == ["OpenAI"]
    assert rows["k99"]["content"]["summary"] == "already"


async def test_items_missing_from_llm_reply_stay_unsummarised_not_corrupted():
    prov = BatchProvider(skip_ids={"k1", "k2"})
    db, c = await crawler(LLMOrchestrator([prov], sleep=nosleep))
    left = await c.backfill_structure(rounds=1)
    assert left == 2
    rows = dict(await db.get_keyed("NEWS"))
    assert not rows["k1"]["content"]["summary"] and rows["k0"]["content"]["summary"] == "sum T0"


async def test_provider_outage_does_not_crash_or_write_partial_data():
    db, c = await crawler(LLMOrchestrator([Down()], sleep=nosleep))
    out = await c._structure_batch([(k, r) for k, r in await db.get_keyed("NEWS")][:15])
    assert out == {}
    assert await c.backfill_structure(rounds=1) == 35
