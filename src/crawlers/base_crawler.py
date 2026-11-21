"""Async base crawler: semaphore, per-domain rate limit, retries, UA rotation,
and tier escalation httpx -> curl_cffi -> playwright.

Escalation only happens on *blocked* responses (403/401/challenge pages), never on
404 or transient 5xx/429 (those are retried within the same tier).
"""
from __future__ import annotations

import asyncio
import json as _json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

import httpx

from src.logging_setup import get_logger
from src.metrics import metrics
from src.retry import full_jitter, parse_retry_after

log = get_logger("crawler")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
]
TIERS = ("httpx", "curl_cffi", "playwright")
BLOCK_STATUS = {401, 403, 451}
RETRY_STATUS = {429, 500, 502, 503, 504}
CHALLENGE_MARKERS = ("just a moment...", "cf-chl", "attention required! | cloudflare", "captcha-delivery")


class FetchError(Exception):
    pass


class BlockedError(FetchError):
    pass


@dataclass
class FetchResult:
    url: str
    status: int
    text: str
    headers: dict[str, str] = field(default_factory=dict)
    tier: str = "httpx"

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        return _json.loads(self.text)


class DomainLimiter:
    """Spaces requests to the same host by >= 1/rps seconds. Slots are reserved under a
    lock, so it stays fair under concurrency."""

    def __init__(
        self,
        rps: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        host_rps: dict[str, float] | None = None,
    ) -> None:
        self.interval = 1.0 / rps if rps > 0 else 0.0
        self.host_interval = {h: (1.0 / r if r > 0 else 0.0) for h, r in (host_rps or {}).items()}
        self._next: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._clock, self._sleep = clock, sleep

    async def wait(self, host: str) -> None:
        interval = self.host_interval.get(host, self.interval)
        if not interval:
            return
        async with self._lock:
            now = self._clock()
            slot = max(now, self._next.get(host, 0.0))
            self._next[host] = slot + interval
        delay = slot - now
        if delay > 0:
            await self._sleep(delay)


class BaseCrawler:
    def __init__(
        self,
        concurrency: int = 16,
        per_domain_rps: float = 4,
        host_rps: dict[str, float] | None = None,  # per-host override, e.g. {"export.arxiv.org": 0.33}
        max_retries: int = 4,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,  # injectable for tests
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.sem = asyncio.Semaphore(concurrency)
        self.limiter = DomainLimiter(per_domain_rps, host_rps=host_rps)
        self.max_retries = max_retries
        self.timeout = timeout
        self._sleep = sleep
        self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True, transport=transport)
        self._cffi = None
        self._pw = None
        self._pw_browser = None
        self._pw_lock = asyncio.Lock()

    # ---- lifecycle -------------------------------------------------------
    async def aclose(self) -> None:
        await self._client.aclose()
        if self._cffi is not None:
            await self._cffi.close()
        if self._pw_browser is not None:
            await self._pw_browser.close()
            await self._pw.stop()

    async def __aenter__(self) -> "BaseCrawler":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    # ---- public API ------------------------------------------------------
    async def fetch(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        json: Any = None,
        render: bool = False,
        escalate: bool = True,
        max_tier: str | None = None,
    ) -> FetchResult:
        """Fetch with retries. render=True starts at playwright; escalate=False pins to httpx
        (use for JSON APIs where a 403 means 'bad key', not 'bot wall')."""
        tiers = TIERS[2:] if render else (TIERS if escalate else TIERS[:1])
        if max_tier:  # e.g. bulk article fetches stop at curl_cffi: a headless browser per article is too slow
            tiers = tiers[: tiers.index(max_tier) + 1] if max_tier in tiers else tiers
        last: Exception | None = None
        for i, tier in enumerate(tiers):
            try:
                res = await self._fetch_tier(tier, url, method, headers, params, json)
                metrics.inc(f"fetch.ok.{tier}")
                return res
            except BlockedError as e:
                last = e
                metrics.inc("fetch.blocked")
                if i + 1 < len(tiers):
                    metrics.inc("fetch.escalations")
                    log.warning("escalating", url=url, from_tier=tier, to_tier=tiers[i + 1])
        raise last or FetchError(f"no tier available for {url}")

    # ---- tier dispatch with retry loop ------------------------------------
    async def _fetch_tier(self, tier, url, method, headers, params, json) -> FetchResult:
        host = urlparse(url).netloc
        hdrs = {"User-Agent": random.choice(USER_AGENTS), "Accept": "*/*", **(headers or {})}
        for attempt in range(self.max_retries + 1):
            async with self.sem:
                await self.limiter.wait(host)
                metrics.inc("fetch.requests")
                try:
                    res = await self._do(tier, url, method, hdrs, params, json)
                except (httpx.TransportError, asyncio.TimeoutError, OSError) as e:
                    if attempt >= self.max_retries:
                        raise FetchError(f"{url}: transport error {e!r}") from e
                    metrics.inc("fetch.retries")
                    await self._sleep(full_jitter(attempt, base=0.5, cap=20))
                    continue
            if res.status in BLOCK_STATUS or self._is_challenge(res):
                raise BlockedError(f"{url}: blocked ({res.status}) on {tier}")
            if res.status in RETRY_STATUS:
                if res.status == 429:
                    metrics.inc("fetch.429")
                if attempt >= self.max_retries:
                    raise FetchError(f"{url}: gave up after {attempt + 1} tries, status {res.status}")
                metrics.inc("fetch.retries")
                wait = parse_retry_after(res.headers)
                delay = min(wait, 120.0) if wait is not None else full_jitter(attempt, base=1.0, cap=30)
                log.info("retrying", url=url, status=res.status, delay=round(delay, 2))
                await self._sleep(delay)
                continue
            return res
        raise FetchError(f"{url}: exhausted retries")

    @staticmethod
    def _is_challenge(res: FetchResult) -> bool:
        if res.status not in (200, 503):
            return False
        head = res.text[:2000].lower()
        return any(m in head for m in CHALLENGE_MARKERS)

    async def _do(self, tier, url, method, headers, params, json) -> FetchResult:
        if tier == "httpx":
            r = await self._client.request(method, url, headers=headers, params=params, json=json)
            return FetchResult(str(r.url), r.status_code, r.text, dict(r.headers), "httpx")
        if tier == "curl_cffi":
            return await self._do_curl(url, method, headers, params, json)
        return await self._do_playwright(url, headers)

    async def _do_curl(self, url, method, headers, params, json) -> FetchResult:
        if self._cffi is None:
            from curl_cffi.requests import AsyncSession  # lazy

            self._cffi = AsyncSession(impersonate="chrome")
        try:
            r = await self._cffi.request(
                method, url, headers=headers, params=params, json=json,
                timeout=self.timeout, allow_redirects=True,
            )
        except Exception as e:  # curl_cffi raises its own error types; map to the retryable transport error
            raise httpx.TransportError(f"curl_cffi: {e}") from e
        return FetchResult(str(r.url), r.status_code, r.text, dict(r.headers), "curl_cffi")

    async def _do_playwright(self, url, headers) -> FetchResult:
        async with self._pw_lock:
            if self._pw_browser is None:
                from playwright.async_api import async_playwright  # lazy

                self._pw = await async_playwright().start()
                self._pw_browser = await self._pw.chromium.launch(headless=True)
        ctx = await self._pw_browser.new_context(user_agent=headers.get("User-Agent"))
        try:
            page = await ctx.new_page()
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=int(self.timeout * 1000))
            html = await page.content()
            return FetchResult(page.url, resp.status if resp else 200, html, {}, "playwright")
        except Exception as e:  # playwright.Error (net::ERR_*, timeouts, target closed) -> retryable transport error
            raise httpx.TransportError(f"playwright: {str(e).splitlines()[0][:200]}") from e
        finally:
            await ctx.close()
