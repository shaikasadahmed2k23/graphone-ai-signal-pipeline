"""Live smoke test of the LLM chain.  python scripts/llm_smoke.py

1. one real call per configured provider (provider + latency printed)
2. forced fallback: Gemini with a bad key must fall through to the next live provider
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings  # noqa: E402
from src.llm.orchestrator import GeminiProvider, LLMOrchestrator, default_providers  # noqa: E402
from src.metrics import metrics  # noqa: E402

SYSTEM = "You classify software pricing. Reply ONLY with JSON."
TEXT = "Acme AI offers a free tier with 100 requests/month; paid plans start at $20/month."
INSTR = 'Return {"pricingModel": "FREE|FREEMIUM|PAID|ENTERPRISE"} for the text below.'


async def main() -> int:
    s = get_settings()
    providers = default_providers(s)
    names = [p.name for p in providers]
    print(f"configured providers: {names or 'NONE'}")
    for need, have in (("gemini", s.gemini_key), ("groq", s.groq_key), ("openrouter", s.openrouter_key)):
        if not have:
            print(f"  ! {need}: no key in .env (GEMINI_API_KEY / GROQ_API_KEY / OPENROUTER_API_KEY) -> skipped")
    ok = True

    print("\n[1] one real call per provider")
    for p in providers:
        orch = LLMOrchestrator([p], max_429_retries=1)
        t = time.perf_counter()
        try:
            r = await orch.complete("smoke", SYSTEM, TEXT, INSTR)
            print(f"  {p.name:<11} OK   {time.perf_counter() - t:5.2f}s  -> {r.text.strip()[:80]}")
        except Exception as e:
            ok = False
            print(f"  {p.name:<11} FAIL {time.perf_counter() - t:5.2f}s  -> {e!r}"[:250])

    print("\n[2] forced fallback: Gemini with a bad key")
    rest = [p for p in providers if p.name != "gemini"]
    if not rest:
        print("  skipped (need at least one non-Gemini key)")
    else:
        orch = LLMOrchestrator([GeminiProvider("BAD_KEY_FOR_TEST"), *rest], max_429_retries=1)
        metrics.reset()
        t = time.perf_counter()
        try:
            r = await orch.complete("smoke-fb", SYSTEM, TEXT, INSTR)
            landed = r.provider
            print(f"  answered by: {landed}  ({time.perf_counter() - t:.2f}s)  fallbacks={metrics.get('llm.fallbacks')}")
            ok = ok and landed == rest[0].name and metrics.get("llm.fallbacks") >= 1
        except Exception as e:
            ok = False
            print(f"  FAIL {e!r}"[:250])
    print("\n[3] batched call: 15 items in ONE request through the full chain")
    import json

    from src.llm.schemas import PricingBatch

    items = [{"id": str(i), "name": f"Tool{i}", "text": TEXT if i % 2 else "Tool is open source and free to use."}
             for i in range(15)]
    t = time.perf_counter()
    try:
        r = await LLMOrchestrator(providers, max_429_retries=1).complete(
            "smoke-batch", SYSTEM, json.dumps(items),
            'Input is a JSON list. Return {"results":[{"id":"<id>","pricingModel":"FREE|FREEMIUM|PAID|ENTERPRISE"}]} for EVERY item.')
        got = PricingBatch.model_validate(r.json()).results
        print(f"  answered by: {r.provider} ({time.perf_counter() - t:.2f}s)  items returned={len(got)}/15")
        ok = ok and len(got) == 15
    except Exception as e:
        ok = False
        print(f"  FAIL {e!r}"[:250])
    print("\nmetrics:\n" + metrics.report())
    return 0 if ok and providers else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
