"""Demo of the escalation ladder on Cloudflare-protected pages.  python scripts/playwright_demo.py [url ...]

Expected: httpx -> 403 (blocked) -> curl_cffi (TLS impersonation) -> [if still blocked] playwright.
Which tier finally succeeds is logged and counted in metrics (fetch.ok.<tier>, fetch.escalations).
"""
from __future__ import annotations

import asyncio
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.crawlers.base_crawler import BaseCrawler, FetchError  # noqa: E402
from src.logging_setup import setup_logging  # noqa: E402
from src.metrics import metrics  # noqa: E402

DEFAULT = ["https://www.producthunt.com/", "https://theresanaiforthat.com/"]


async def main(urls: list[str]) -> int:
    setup_logging("WARNING")
    bad = 0
    async with BaseCrawler(per_domain_rps=2) as c:
        for u in urls:
            t = time.perf_counter()
            try:
                r = await c.fetch(u)
                title = re.search(r"<title[^>]*>(.*?)</title>", r.text, re.S | re.I)
                print(f"OK   {u}\n     tier={r.tier} status={r.status} bytes={len(r.text)} "
                      f"{time.perf_counter() - t:.1f}s title={title.group(1).strip()[:70] if title else None!r}")
            except FetchError as e:
                bad += 1
                print(f"FAIL {u}\n     {e}")
        # force the playwright tier explicitly on a JS-rendered page (content absent from raw HTML)
        u = "https://www.producthunt.com/"
        t = time.perf_counter()
        r = await c.fetch(u, render=True)
        print(f"FORCED playwright {u}: status={r.status} bytes={len(r.text)} {time.perf_counter() - t:.1f}s")
    print("\nmetrics:\n" + metrics.report())
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:] or DEFAULT)))
