# AI-startup ingestion pipeline

Async, fault-tolerant pipeline that collects **startups, products, research papers, jobs and news**, normalises them
to schema `1.0` JSON in a database (SQLite now, Postgres via `DB_URL`), and exports six tabs to a public Google Sheet:
**Startups, Products, Research Papers, Jobs, News, Entity Mapping Log**.

Architecture, scaling to 500k, 413/429 handling, distributed dedup and storage: **[architecture.pdf](architecture.pdf)**
(source: [docs/architecture.md](docs/architecture.md)).

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Linux/macOS: .venv/bin/pip
.venv/Scripts/playwright install chromium
cp .env.example .env                                  # then fill in keys
```

`.env` (never committed; `.env` and the service-account JSON are in `.gitignore`):

| Variable | Used for |
|---|---|
| `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY` | LLM chain (any subset works; order Gemini -> Groq -> Groq small -> OpenRouter) |
| `GITHUB_TOKEN` | GraphQL star lookups for papers (without it: Hugging Face star fallback) |
| `PRODUCT_HUNT_TOKEN` | Product Hunt GraphQL |
| `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_SHEET_ID` | Sheet export (share the sheet with the service-account e-mail as Editor) |
| `DB_URL`, `CONCURRENCY`, `PER_DOMAIN_RPS` | storage and politeness (default `sqlite:///data/pipeline.db`) |

## Run

```bash
python scripts/llm_smoke.py                    # live check: 1 call/provider, forced fallback, 15-item batch
python -m src.pipeline --vertical <name> [--limit N] [--reset] [--dry-run] [--log-level INFO]
pytest                                         # 149 tests, no network needed
python scripts/build_docs.py                   # docs/architecture.md -> architecture.pdf
```

| `--vertical` | What it does |
|---|---|
| `startups` | YC directory via Algolia, partitioned by batch; keeps everything, flags `is_ai` |
| `papers` | arXiv in 7-day windows until >= 1,000 have a GitHub repo, then stars (GraphQL, 100 repos/query; HF fallback); handles renamed/deleted repos |
| `products` | Product Hunt AI topic; rule + batched LLM pricing |
| `product_directory` | theresanaiforthat.com: **explicit pricing labels** mapped deterministically (`pricingSource=directory_label`), deduped/upgraded against stored products |
| `product_sites` | quality pass: real website/domain + `/pricing` page -> site-grounded pricing, re-resolves with domain |
| `jobs` | Greenhouse, Ashby, Lever, RemoteOK, Remotive, WWR; strict 24 h; LLM only for role family / remote gaps |
| `news` | TechCrunch AI, VentureBeat, The Verge AI, MIT TR, Hacker News; strict 24 h; full text, summary, entities |
| `export` | writes the 6 tabs to the Google Sheet (and `data/export/*.csv`); `--dry-run` skips the Sheet |
| `all` | startups, products, papers, news, jobs |

Every run checkpoints (resume after a crash), is idempotent (content-hash upserts) and prints metrics at the end.
Re-run `jobs`, `news`, then `export` right before reading the sheet so the 24 h window is fresh.

## Design overview

* **Structured fields first, LLM only for gaps** (pricing, role family, remote flag, news structuring); results cached by content hash.
* **Fetch tiers** httpx -> curl_cffi -> Playwright, escalating only on blocks; 429/5xx retried with `Retry-After` / full jitter; per-host rate limits.
* **LLM gateway:** token pre-estimation, compression -> split/merge, budget halving on 413, 15 items per call, per-provider circuit breakers, quota hold.
* **Entity resolver:** domain -> exact -> compact -> core -> fuzzy -> new. Generic single-word names (Sierra, Modal, Scale...) are *ambiguous* and need a domain or AI-context signal, otherwise a new canonical is created and logged `ambiguous_rejected`. Every decision (raw, canonical, method, score) feeds the **Entity Mapping Log** tab.
* **Freshness and dedup:** UTC-normalised dates (feed -> HTML meta/JSON-LD -> first-seen), canonical-URL hash + banded simhash.

## Results from real runs (2026-11-21)

| Vertical | Result | Notes |
|---|---|---|
| Startups | 6,406 in ~25 s; **2,897 AI** exported | YC Algolia, 1 request per batch facet page |
| Papers | **1,303** with GitHub URL in ~40 s; **1,264 with stars** | 871 via 10 GraphQL batches, 393 via HF; 39 `repo_missing` (stars `null`), 5 `repo_renamed` |
| Products | 1,885 stored; **1,301 exported** (grounded pricing) | `llm` 607, `directory_label` 518, `site_pricing_page` 84, `rule` 67, `site_homepage` 25; 584 `llm_guess` kept in DB, flagged, not exported |
| Jobs | **114** in the last 24 h | Ashby, Greenhouse, WWR contribute |
| News | **152** in the last 24 h, all summarised | HN supplies most; a re-run dropped 145 duplicate URLs + 1 syndicated copy of 223 fetched |
| Entity log | 2,446 decisions | exact / domain / fuzzy / new / `ambiguous_rejected` |
| LLM chain | Gemini 1.7-2 s, Groq 0.7 s, OpenRouter 1.2-1.7 s | forced bad Gemini key -> Groq in ~0.7 s; 15-item batch returns 15/15 |

## Known limitations

**Product Hunt redirects.** Product Hunt's API exposes `website` / `productLinks` only as `producthunt.com/r/<id>`
redirects behind Cloudflare: 403 to httpx and curl_cffi, only fresh-context Playwright gets through (~60% at first,
and almost nothing after sustained use). So only ~200 of the 1,206 Product Hunt products have a real `websiteUrl` /
`domain`. Mitigation: the TAAFT directory supplies explicit pricing labels and real website URLs (curl_cffi tier), and
the Products tab contains only rows with grounded pricing. The remaining Product Hunt rows with `pricingSource =
llm_guess` (a model guess from the tagline) stay in the DB for transparency. Re-running `product_sites` from another
network resolves more domains.

**Other**
* TAAFT labels are the directory's own claims (e.g. "Free + from $20/mo" -> `FREEMIUM`); "No pricing" is unmappable and goes to the LLM as a flagged `llm_guess`.
* Jobs: RemoteOK and Remotive delay data by 24 h, so they add ~0 rows to a strict 24 h window; Lever's one board had nothing fresh.
* News: only ~12-15 fresh items/day come from the 4 RSS sources; Hacker News supplies the bulk.
* GitHub: deleted/private repos get `github_stars = null` (logged `repo_missing`); renamed repos are followed and the URL updated.
* LLM free tiers are quota-limited (Gemini and Groq daily budgets were hit during development); the chain degrades to the next tier and backfill passes fill anything left empty.
* GitHub Models is wired but opt-in (`GITHUB_MODELS=1`): from the dev network every `models.github.ai` endpoint returned a plain-text `OK`.
