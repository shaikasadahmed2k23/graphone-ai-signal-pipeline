import httpx
import pytest

from src.crawlers.base_crawler import BaseCrawler, BlockedError, DomainLimiter, FetchError
from src.metrics import metrics


@pytest.fixture(autouse=True)
def _reset():
    metrics.reset()


def crawler(handler, **kw):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    c = BaseCrawler(transport=httpx.MockTransport(handler), sleep=fake_sleep, per_domain_rps=0, **kw)
    return c, sleeps


async def test_retries_429_honoring_retry_after():
    n = {"i": 0}

    def h(req):
        n["i"] += 1
        return httpx.Response(429, headers={"Retry-After": "3"}) if n["i"] < 3 else httpx.Response(200, text="ok")

    c, sleeps = crawler(h)
    r = await c.fetch("https://x.test/a", escalate=False)
    assert r.ok and sleeps == [3.0, 3.0] and metrics.get("fetch.429") == 2
    await c.aclose()


async def test_gives_up_after_max_retries():
    c, _ = crawler(lambda req: httpx.Response(503), max_retries=2)
    with pytest.raises(FetchError):
        await c.fetch("https://x.test/a", escalate=False)
    await c.aclose()


async def test_403_raises_blocked_when_not_escalating():
    c, _ = crawler(lambda req: httpx.Response(403))
    with pytest.raises(BlockedError):
        await c.fetch("https://x.test/a", escalate=False)
    await c.aclose()


async def test_cloudflare_challenge_detected_as_blocked():
    c, _ = crawler(lambda req: httpx.Response(200, text="<title>Just a moment...</title>"))
    with pytest.raises(BlockedError):
        await c.fetch("https://x.test/a", escalate=False)
    await c.aclose()


async def test_404_returned_not_retried_or_escalated():
    c, _ = crawler(lambda req: httpx.Response(404, text="nope"))
    r = await c.fetch("https://x.test/a")
    assert r.status == 404 and metrics.get("fetch.requests") == 1
    await c.aclose()


async def test_domain_limiter_spaces_requests():
    t, sleeps = [0.0], []

    async def sl(s):
        sleeps.append(round(s, 3))

    lim = DomainLimiter(rps=2, clock=lambda: t[0], sleep=sl)
    for _ in range(3):
        await lim.wait("a.com")
    await lim.wait("b.com")
    assert sleeps == [0.5, 1.0]  # a.com: 0, 0.5, 1.0 ; b.com independent


async def test_playwright_and_curl_errors_become_fetch_errors_not_crashes(monkeypatch):
    import src.crawlers.base_crawler as bc

    async def boom(self, *a, **k):
        raise bc.httpx.TransportError("playwright: net::ERR_CONNECTION_TIMED_OUT")

    c, _ = crawler(lambda req: httpx.Response(200), max_retries=1)
    monkeypatch.setattr(bc.BaseCrawler, "_do_playwright", boom)
    with pytest.raises(FetchError):  # retried, then a clean FetchError (caller can skip the item)
        await c.fetch("https://x.test/a", render=True)
    assert metrics.get("fetch.retries") == 1
    await c.aclose()
