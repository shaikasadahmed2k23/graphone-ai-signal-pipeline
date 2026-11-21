import json

import pytest

from src.llm.chunking import estimate_tokens, prepare_payload, split_text
from src.llm.orchestrator import (
    AllProvidersFailed, CircuitBreaker, ContextTooLargeError, LLMOrchestrator,
    Provider, ProviderUnavailable, RateLimitError,
)
from src.metrics import metrics


class Fake(Provider):
    """Scripted provider: each call pops the next behaviour (exception instance or str)."""

    def __init__(self, name, script, max_input_tokens=1000):
        self.name, self.script, self.max_input_tokens = name, list(script), max_input_tokens
        self.calls = []

    async def generate(self, system, prompt, json_mode=True):
        self.calls.append(prompt)
        b = self.script.pop(0) if self.script else json.dumps({"ok": True})
        if isinstance(b, Exception):
            raise b
        return b


@pytest.fixture(autouse=True)
def _reset():
    metrics.reset()


def make(providers, **kw):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    return LLMOrchestrator(providers, sleep=fake_sleep, **kw), sleeps


async def test_429_respects_retry_after_then_succeeds():
    p = Fake("a", [RateLimitError("429", retry_after=7), '{"v": 1}'])
    orch, sleeps = make([p])
    res = await orch.complete("t", "sys", "hello")
    assert res.json() == {"v": 1}
    assert len(sleeps) == 1 and 7 <= sleeps[0] <= 8.0  # >= Retry-After, plus tiny jitter
    assert metrics.get("llm.429.a") == 1


async def test_429_without_retry_after_uses_full_jitter_bounds():
    p = Fake("a", [RateLimitError(), RateLimitError(), RateLimitError(), '{"v": 1}'])
    orch, sleeps = make([p])
    await orch.complete("t", "s", "x")
    assert len(sleeps) == 3
    for attempt, s in enumerate(sleeps):
        assert 0 <= s <= min(60, 2**attempt)


async def test_persistent_429_falls_back_to_next_provider():
    a = Fake("a", [RateLimitError()] * 10)
    b = Fake("b", ['{"from": "b"}'])
    orch, _ = make([a, b], max_429_retries=2)
    res = await orch.complete("t", "s", "x")
    assert res.provider == "b" and res.json() == {"from": "b"}
    assert metrics.get("llm.fallbacks") == 1
    assert len(a.calls) == 3  # 1 + 2 retries


async def test_huge_retry_after_skips_waiting_and_falls_back():
    a = Fake("a", [RateLimitError(retry_after=3600)])
    b = Fake("b", ['{"x": 1}'])
    orch, sleeps = make([a, b])
    res = await orch.complete("t", "s", "x")
    assert res.provider == "b" and sleeps == []


async def test_oversized_payload_is_chunked_before_call_and_merged():
    big = "\n\n".join(f"Paragraph {i}. " + "word " * 120 for i in range(60))  # ~ 9k tokens
    p = Fake("small", [], max_input_tokens=1000)
    orch, _ = make([p])
    res = await orch.complete("t", "s", big, instruction="extract")
    assert res.chunks >= 1
    assert metrics.get("llm.413_avoided") >= 1
    for prompt in p.calls:
        assert estimate_tokens(prompt) <= p.max_input_tokens  # never exceeded provider cap


async def test_split_merge_when_compression_not_enough():
    text = "\n\n".join(f"Sentence number {i} has content." * 40 for i in range(80))
    pieces = prepare_payload(text, 800)
    assert all(estimate_tokens(c) <= 800 for c in pieces)
    chunks = split_text(text, 300)
    assert len(chunks) > 1 and all(estimate_tokens(c) <= 300 for c in chunks)


async def test_413_from_provider_halves_budget_and_retries():
    text = "\n\n".join("Some sentence about funding of $5M. " * 10 for _ in range(40))
    p = Fake("a", [ContextTooLargeError("413"), '{"ok": 1}'], max_input_tokens=2000)
    orch, _ = make([p])
    res = await orch.complete("t", "s", text)
    assert res.json() == {"ok": 1}
    assert metrics.get("llm.413.a") == 1
    assert estimate_tokens(p.calls[1]) < estimate_tokens(p.calls[0])


async def test_circuit_breaker_opens_and_skips_provider():
    now = [0.0]
    a = Fake("a", [ProviderUnavailable("500")] * 50)
    b = Fake("b", [])
    orch, _ = make([a, b], breaker_threshold=2, breaker_cooldown=30, clock=lambda: now[0])
    for i in range(4):
        await orch.complete("t", "s", f"doc {i}")
    assert orch.breakers["a"].state == "open"
    assert metrics.get("llm.circuit_skip.a") >= 2
    calls_before = len(a.calls)
    now[0] = 31  # cooldown elapsed -> half-open probe allowed
    await orch.complete("t", "s", "doc probe")
    assert len(a.calls) > calls_before


def test_breaker_half_open_single_probe():
    t = [0.0]
    br = CircuitBreaker(threshold=1, cooldown=10, clock=lambda: t[0])
    br.failure()
    assert br.state == "open" and not br.allow()
    t[0] = 11
    assert br.allow() and not br.allow()  # only one probe
    br.success()
    assert br.state == "closed"


async def test_cache_by_content_hash():
    p = Fake("a", ['{"n": 1}'])
    orch, _ = make([p])
    r1 = await orch.complete("t", "s", "same")
    r2 = await orch.complete("t", "s", "same")
    assert r2.cached and r2.text == r1.text and len(p.calls) == 1
    assert metrics.get("llm.cache_hits") == 1


async def test_all_fail_raises():
    orch, _ = make([Fake("a", [ProviderUnavailable("x")] * 9), Fake("b", [ProviderUnavailable("y")] * 9)])
    with pytest.raises(AllProvidersFailed):
        await orch.complete("t", "s", "x")


async def test_malformed_json_triggers_fallback():
    a = Fake("a", ['{"truncated": ', '{"still": bad'])
    b = Fake("b", ['{"ok": 1}'])
    orch, _ = make([a, b])
    res = await orch.complete("t", "s", "x")
    assert res.provider == "b" and res.json() == {"ok": 1}
    assert metrics.get("llm.bad_json.a") == 2 and metrics.get("llm.fallbacks") == 1


async def test_quota_exhausted_opens_circuit_for_an_hour_not_30s():
    from src.llm.orchestrator import ProviderExhausted
    now = [0.0]
    a = Fake("a", [ProviderExhausted("402 no credit")] * 5)
    b = Fake("b", [])
    orch, _ = make([a, b], clock=lambda: now[0])
    await orch.complete("t", "s", "doc 1")
    assert orch.breakers["a"].state == "open" and metrics.get("llm.exhausted.a") == 1
    now[0] = 600  # 10 min later: normal 30s cooldown would have expired, exhausted one must not
    await orch.complete("t", "s", "doc 2")
    assert len(a.calls) == 1 and metrics.get("llm.circuit_skip.a") == 1
    now[0] = 3601
    await orch.complete("t", "s", "doc 3")
    assert len(a.calls) == 2  # half-open probe after an hour


async def test_long_retry_after_holds_circuit_open_for_that_long():
    now = [0.0]
    a = Fake("a", [RateLimitError("TPD", retry_after=574)] * 3)
    b = Fake("b", [])
    orch, _ = make([a, b], clock=lambda: now[0])
    await orch.complete("t", "s", "d1")
    assert orch.breakers["a"].state == "open" and metrics.get("llm.long_wait.a") == 1
    now[0] = 300
    await orch.complete("t", "s", "d2")
    assert len(a.calls) == 1  # still skipped 5 min in (Retry-After said ~9.5 min)
    now[0] = 600
    await orch.complete("t", "s", "d3")
    assert len(a.calls) == 2
