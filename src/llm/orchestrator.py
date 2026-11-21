"""LLM orchestrator: provider fallback chain with per-provider circuit breakers.

  Gemini Flash -> Groq Llama -> DeepSeek (OpenRouter)

* 429  -> wait (Retry-After if sent, else exponential backoff with full jitter), retry;
          if retries are exhausted or Retry-After is absurdly long, fall back to next provider.
* 413 / context-length -> avoided up front by token estimation + semantic chunking
          (chunking.prepare_payload); if a provider still rejects, halve the budget and redo.
* Circuit breaker per provider: N consecutive failures -> open for a cooldown, then one
          half-open probe.
* Results cached by content hash (any object with async get/set; in-memory by default).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

import httpx

from src.llm.chunking import estimate_tokens, prepare_payload
from src.logging_setup import get_logger
from src.metrics import metrics
from src.retry import full_jitter, parse_retry_after

log = get_logger("llm")


# ---------------------------------------------------------------- errors
class ProviderError(Exception):
    pass


class RateLimitError(ProviderError):
    def __init__(self, msg: str = "429", retry_after: float | None = None) -> None:
        super().__init__(msg)
        self.retry_after = retry_after


class ProviderExhausted(ProviderError):
    """Billing/quota exhausted (402, or 429 'exceeded your quota' with no Retry-After): retrying in
    seconds is pointless, so the circuit stays open for a long time."""


class ContextTooLargeError(ProviderError):
    pass


class ProviderUnavailable(ProviderError):
    pass


class AllProvidersFailed(Exception):
    pass


# ---------------------------------------------------------------- providers
class Provider(ABC):
    name: str
    max_input_tokens: int  # hard cap we enforce per call (<= real context, also respects TPM limits)

    @abstractmethod
    async def generate(self, system: str, prompt: str, json_mode: bool = True) -> str: ...


def _raise_for(status: int, headers: Any, body: str) -> None:
    low = body.lower()
    if status == 402:
        raise ProviderExhausted(body[:200])
    if status == 429:
        ra = parse_retry_after(dict(headers))
        if ra is None and ("quota" in low or "billing" in low):
            raise ProviderExhausted(body[:200])
        raise RateLimitError(body[:200], ra)
    if status == 413 or (status == 400 and ("context" in low or "too large" in low or "token" in low and "limit" in low)):
        raise ContextTooLargeError(body[:200])
    if status >= 500 or status in (408, 409):
        raise ProviderUnavailable(f"{status}: {body[:200]}")
    if status >= 400:
        raise ProviderError(f"{status}: {body[:200]}")


class OpenAICompatProvider(Provider):
    """Groq and OpenRouter both speak the OpenAI chat-completions dialect."""

    def __init__(self, name: str, base_url: str, api_key: str, model: str, max_input_tokens: int,
                 client: httpx.AsyncClient | None = None, extra_headers: dict | None = None,
                 max_output_tokens: int = 2048) -> None:
        self.max_output_tokens = max_output_tokens
        self.name, self.base_url, self.api_key, self.model = name, base_url, api_key, model
        self.max_input_tokens = max_input_tokens
        self._client = client or httpx.AsyncClient(timeout=30)
        self._extra = extra_headers or {}

    async def generate(self, system: str, prompt: str, json_mode: bool = True) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_output_tokens,  # unset = provider default (16k+), which can 402 on low credit
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            r = await self._client.post(
                f"{self.base_url}/chat/completions", json=body,
                headers={"Authorization": f"Bearer {self.api_key}", **self._extra},
            )
        except httpx.TransportError as e:
            raise ProviderUnavailable(repr(e)) from e
        _raise_for(r.status_code, r.headers, r.text)
        try:
            return r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as e:
            raise ProviderUnavailable(f"malformed response: {r.text[:200]}") from e


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", max_input_tokens: int = 100_000,
                 client: httpx.AsyncClient | None = None) -> None:
        self.api_key, self.model, self.max_input_tokens = api_key, model, max_input_tokens
        self._client = client or httpx.AsyncClient(timeout=30)

    async def generate(self, system: str, prompt: str, json_mode: bool = True) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 2048},
        }
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        try:
            r = await self._client.post(url, json=body, headers={"x-goog-api-key": self.api_key})
        except httpx.TransportError as e:
            raise ProviderUnavailable(repr(e)) from e
        _raise_for(r.status_code, r.headers, r.text)
        try:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, ValueError) as e:
            raise ProviderUnavailable(f"malformed/blocked response: {r.text[:200]}") from e


def default_providers(settings) -> list[Provider]:
    """Chain order = free tiers first, paid last: Gemini -> Groq -> OpenRouter (saves credit).
    Model ids are env-overridable (GEMINI_MODEL / GROQ_MODEL / OPENROUTER_MODEL) because providers retire them."""
    import os

    chain: list[Provider] = []
    if settings.gemini_key:  # NB: gemini-flash-latest alias hung in testing
        chain.append(GeminiProvider(settings.gemini_key, model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash")))
    if settings.groq_key:  # free-tier TPM is small -> a low per-call input cap is what actually avoids 413s
        chain.append(OpenAICompatProvider("groq", "https://api.groq.com/openai/v1", settings.groq_key,
                                          os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), max_input_tokens=3_000))
    if settings.groq_key and os.getenv("GROQ_SMALL_MODEL", "openai/gpt-oss-20b"):
        # separate per-model free quota on Groq -> extra headroom when the 120b daily token budget is spent
        chain.append(OpenAICompatProvider("groq-small", "https://api.groq.com/openai/v1", settings.groq_key,
                                          os.getenv("GROQ_SMALL_MODEL", "openai/gpt-oss-20b"), max_input_tokens=3_000))
    if settings.github_token and os.getenv("GITHUB_MODELS", "0") == "1":
        # GitHub Models (free, rate-limited per day) via the same PAT; needs models:read. OPT-IN: from the dev
        # network every models.github.ai endpoint answered a plain-text "OK" (not an API response), so it is off
        # by default; a malformed reply is a provider failure anyway and falls through to the next tier.
        chain.append(OpenAICompatProvider("github-models", "https://models.github.ai/inference", settings.github_token,
                                          os.getenv("GITHUB_MODEL", "openai/gpt-4.1-mini"), max_input_tokens=6_000,
                                          extra_headers={"Accept": "application/vnd.github+json",
                                                         "X-GitHub-Api-Version": "2022-11-28"}))
    if settings.openrouter_key:
        chain.append(OpenAICompatProvider("openrouter", "https://openrouter.ai/api/v1", settings.openrouter_key,
                                          os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat"), max_input_tokens=30_000,
                                          max_output_tokens=1500))  # low credit balance: 2048 was rejected
    return chain


# ---------------------------------------------------------------- circuit breaker
class CircuitBreaker:
    def __init__(self, threshold: int = 3, cooldown: float = 30.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.threshold, self.cooldown, self._clock = threshold, cooldown, clock
        self.failures = 0
        self.opened_at: float | None = None
        self._probing = False

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half_open" if self._clock() - self.opened_at >= self.cooldown else "open"

    def allow(self) -> bool:
        s = self.state
        if s == "closed":
            return True
        if s == "half_open" and not self._probing:
            self._probing = True  # exactly one probe request
            return True
        return False

    def success(self) -> None:
        self.failures, self.opened_at, self._probing = 0, None, False

    def force_open(self, duration: float) -> None:
        """Open for `duration` seconds regardless of the failure count (billing / quota exhaustion)."""
        self.failures = self.threshold
        self.opened_at = self._clock() + max(0.0, duration - self.cooldown)  # half-open after `duration`
        self._probing = False

    def failure(self) -> None:
        self.failures += 1
        self._probing = False
        if self.failures >= self.threshold or self.opened_at is not None:
            self.opened_at = self._clock()


# ---------------------------------------------------------------- cache
class Cache(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str) -> None: ...


class InMemoryCache:
    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._d.get(key)

    async def set(self, key: str, value: str) -> None:
        self._d[key] = value


# ---------------------------------------------------------------- merge
def merge_json_results(parts: list[str]) -> str:
    """Merge per-chunk outputs. Dicts: first non-empty scalar wins, lists concatenate (deduped).
    Non-JSON: joined text."""
    objs = []
    for p in parts:
        try:
            objs.append(json.loads(p))
        except (ValueError, TypeError):
            return "\n".join(parts)
    if not all(isinstance(o, dict) for o in objs):
        return json.dumps(objs)
    merged: dict[str, Any] = {}
    for o in objs:
        for k, v in o.items():
            if isinstance(v, list):
                cur = merged.setdefault(k, [])
                cur.extend(x for x in v if x not in cur)
            elif merged.get(k) in (None, "", "unknown") and v not in (None, ""):
                merged[k] = v
            else:
                merged.setdefault(k, v)
    return json.dumps(merged)


# ---------------------------------------------------------------- orchestrator
@dataclass
class LLMResult:
    text: str
    provider: str
    cached: bool = False
    chunks: int = 1

    def json(self) -> Any:
        return json.loads(self.text)


class LLMOrchestrator:
    def __init__(
        self,
        providers: list[Provider],
        cache: Cache | None = None,
        max_429_retries: int = 4,
        max_retry_wait: float = 60.0,
        breaker_threshold: int = 3,
        breaker_cooldown: float = 30.0,
        exhausted_cooldown: float = 3600.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.providers = providers
        self.cache = cache or InMemoryCache()
        self.max_429_retries, self.max_retry_wait = max_429_retries, max_retry_wait
        self.exhausted_cooldown = exhausted_cooldown
        self._sleep = sleep
        self.breakers = {p.name: CircuitBreaker(breaker_threshold, breaker_cooldown, clock) for p in providers}

    @staticmethod
    def cache_key(task: str, system: str, text: str) -> str:
        return hashlib.sha256(f"{task}\x00{system}\x00{text}".encode()).hexdigest()

    async def complete(
        self,
        task: str,
        system: str,
        text: str,
        instruction: str = "",
        json_mode: bool = True,
        merge: Callable[[list[str]], str] = merge_json_results,
    ) -> LLMResult:
        key = self.cache_key(task, system + instruction, text)
        if (hit := await self.cache.get(key)) is not None:
            metrics.inc("llm.cache_hits")
            return LLMResult(hit, "cache", cached=True)

        errors: list[str] = []
        for i, prov in enumerate(self.providers):
            br = self.breakers[prov.name]
            if not br.allow():
                metrics.inc(f"llm.circuit_skip.{prov.name}")
                errors.append(f"{prov.name}: circuit open")
                continue
            try:
                out, n_chunks = await self._run_provider(prov, system, instruction, text, json_mode, merge)
            except ProviderError as e:
                if isinstance(e, ProviderExhausted):
                    br.force_open(self.exhausted_cooldown)
                    metrics.inc(f"llm.exhausted.{prov.name}")
                elif isinstance(e, RateLimitError) and e.retry_after and e.retry_after > self.max_retry_wait:
                    br.force_open(e.retry_after)  # e.g. Groq daily-token limit: "try again in 9m33s"
                    metrics.inc(f"llm.long_wait.{prov.name}")
                else:
                    br.failure()
                errors.append(f"{prov.name}: {e!r}")
                if i + 1 < len(self.providers):
                    metrics.inc("llm.fallbacks")
                    log.warning("llm_fallback", frm=prov.name, error=str(e)[:150], breaker=br.state)
                continue
            br.success()
            await self.cache.set(key, out)
            return LLMResult(out, prov.name, chunks=n_chunks)
        raise AllProvidersFailed("; ".join(errors))

    async def _run_provider(self, prov, system, instruction, text, json_mode, merge) -> tuple[str, int]:
        overhead = estimate_tokens(system) + estimate_tokens(instruction) + 200
        budget = max(500, prov.max_input_tokens - overhead)
        for shrink in range(4):  # 413 despite estimate -> halve and redo
            pieces = prepare_payload(text, budget)
            if len(pieces) > 1 or pieces[0] != text:
                metrics.inc("llm.413_avoided")
                metrics.inc("llm.chunks_created", len(pieces))
            try:
                outs = []
                for piece in pieces:
                    prompt = f"{instruction}\n\n{piece}" if instruction else piece
                    outs.append(await self._call_with_retry(prov, system, prompt, json_mode))
                return (outs[0] if len(outs) == 1 else merge(outs)), len(pieces)
            except ContextTooLargeError:
                metrics.inc(f"llm.413.{prov.name}")
                log.warning("llm_413_despite_estimate", provider=prov.name, budget=budget)
                budget //= 2
        raise ContextTooLargeError(f"{prov.name}: still too large after shrinking")

    async def _call_with_retry(self, prov: Provider, system: str, prompt: str, json_mode: bool) -> str:
        for attempt in range(self.max_429_retries + 1):
            metrics.inc(f"llm.calls.{prov.name}")
            try:
                out = await prov.generate(system, prompt, json_mode)
                if json_mode:
                    try:
                        json.loads(out)
                    except ValueError as e:  # malformed JSON is a provider failure -> retry once, then fall back
                        metrics.inc(f"llm.bad_json.{prov.name}")
                        raise ProviderUnavailable(f"invalid JSON from {prov.name}: {out[:80]!r}") from e
                return out
            except RateLimitError as e:
                metrics.inc(f"llm.429.{prov.name}")
                if attempt >= self.max_429_retries:
                    raise
                if e.retry_after is not None:
                    if e.retry_after > self.max_retry_wait:
                        raise  # server wants a long wait -> cheaper to fall back
                    delay = e.retry_after + full_jitter(0, base=0.5, cap=1.0)  # respect it, tiny jitter
                else:
                    delay = full_jitter(attempt, base=1.0, cap=self.max_retry_wait)
                log.info("llm_429_backoff", provider=prov.name, delay=round(delay, 2), attempt=attempt)
                await self._sleep(delay)
            except ProviderUnavailable:
                metrics.inc(f"llm.5xx.{prov.name}")
                if attempt >= 1:  # one quick retry for transient 5xx, then fall back
                    raise
                await self._sleep(full_jitter(attempt, base=0.5, cap=5))
        raise RateLimitError("exhausted")
