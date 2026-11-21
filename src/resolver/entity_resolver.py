"""Entity resolution: raw company string (+ optional domain) -> canonical name.

Pipeline (first hit wins; every decision is recorded with method + score):
  0. domain      raw looks like a URL/domain ("openai.com")           score 100
  1. exact       normalized string equals a known alias               score 100
  2. compact     same with spaces removed ("Open AI" -> "openai")      score 99
  3. core        legal/brand suffixes stripped ("Mistral AI" ~ "Mistral") score 95
  4. fuzzy       rapidfuzz token_set_ratio >= threshold (+ length guard)  score = ratio
  5. domain_hint caller-supplied website domain matches a known entity   score 100
  *  ambiguous   a generic single-word seed name ("Sierra", "Lambda") matched by name alone is NOT
                 accepted: needs a matching domain or an AI context signal, else -> "ambiguous_rejected"
                 (a separate canonical "<name> (unverified)" is created)
  6. new         register a new canonical (so later aliases resolve to it)  score 0

`decisions` feeds the 'Entity Mapping Log' sheet tab.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from rapidfuzz import fuzz, process

from src.logging_setup import get_logger
from src.metrics import metrics

log = get_logger("resolver")

SEED_PATH = Path(__file__).with_name("seed_entities.json")
LEGAL_SUFFIXES = {"inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "co", "company",
                  "gmbh", "pbc", "plc", "sas", "sa", "bv", "ag", "se", "opco", "lp", "llp"}
BRAND_SUFFIXES = {"labs", "lab", "ai", "technologies", "technology", "systems", "software", "group", "holdings"}
MIN_CORE_LEN = 5
AI_CONTEXT = re.compile(
    r"(?<![A-Za-z0-9])(AI|A\.I\.|LLMs?|GPT|machine[- ]learning|deep[- ]learning|generative|neural|computer vision|NLP|"
    r"foundation models?|artificial intelligence|agents?|copilot|inference|fine-?tun\w*)(?![A-Za-z0-9])", re.I)
_DOMAIN_RE =re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)+)(?::\d+)?(?:[/?#].*)?$", re.I)
_TWO_PART_TLDS = {"co.uk", "com.au", "co.jp", "com.cn", "co.in", "com.br"}


@dataclass
class Decision:
    raw: str
    canonical: str
    method: str
    score: float
    timestamp: str


def normalize(name: str) -> str:
    """lowercase, strip accents, '&'->'and', drop punctuation, collapse spaces, drop trailing legal suffixes."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)  # punctuation (incl. dots: 'Character.AI' -> 'character ai')
    toks = s.split()
    while len(toks) > 1 and toks[-1] in LEGAL_SUFFIXES:
        toks.pop()
    return " ".join(toks)


def compact(norm: str) -> str:
    return norm.replace(" ", "")


def core(norm: str) -> str:
    """Compact form with brand suffixes (ai/labs/...) stripped, keeping at least one token."""
    toks = norm.split()
    while len(toks) > 1 and toks[-1] in BRAND_SUFFIXES:
        toks.pop()
    stripped = "".join(toks)
    # Too-short remainders are generic words ("Open AI" -> "open") and would merge unrelated names.
    return stripped if len(stripped) >= MIN_CORE_LEN else "".join(norm.split())


def extract_domain(raw: str) -> str | None:
    """Registered domain of a URL/domain-looking string, else None."""
    s = raw.strip()
    m = _DOMAIN_RE.match(s)
    if not m or " " in s:
        return None
    host = m.group(1).lower()
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in _TWO_PART_TLDS:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


class EntityResolver:
    def __init__(self, seed_path: Path | None = SEED_PATH, fuzzy_threshold: float = 88.0) -> None:
        self.threshold = fuzzy_threshold
        self.canonicals: dict[str, dict] = {}  # canonical -> {"aliases": set, "domain": str|None}
        self._exact: dict[str, str] = {}
        self._compact: dict[str, str] = {}
        self._core: dict[str, str] = {}
        self._domain: dict[str, str] = {}
        self.decisions: list[Decision] = []
        self._seen: set[tuple[str, str]] = set()
        self._ambiguous: set[str] = set()  # canonicals whose bare single-word name is generic
        self._rejected: dict[str, str] = {}  # normalized raw -> "<name> (unverified)" canonical
        if seed_path:
            for e in json.loads(Path(seed_path).read_text(encoding="utf8")):
                self.add_entity(e["canonical"], e.get("aliases", []), e.get("domain"), e.get("ambiguous", False))

    # ---- index ----------------------------------------------------------------
    def add_entity(self, canonical: str, aliases: list[str] | None = None, domain: str | None = None,
                   ambiguous: bool = False) -> None:
        if ambiguous:
            self._ambiguous.add(canonical)
        rec = self.canonicals.setdefault(canonical, {"aliases": set(), "domain": None})
        for a in [canonical, *(aliases or [])]:
            rec["aliases"].add(a)
            n = normalize(a)
            if not n:
                continue
            self._exact.setdefault(n, canonical)
            self._compact.setdefault(compact(n), canonical)
            self._core.setdefault(core(n).replace(" ", ""), canonical)
            d = extract_domain(a)
            if d:
                self._domain.setdefault(d, canonical)
        if domain:
            d = extract_domain(domain) or domain.lower()
            rec["domain"] = d
            self._domain.setdefault(d, canonical)

    # ---- resolve ---------------------------------------------------------------
    def resolve(self, raw: str, domain: str | None = None, context: str | None = None) -> str:
        """Return canonical name; always records a Decision.
        `domain`: the company's website, if known.  `context`: tags/description text, used only to
        accept matches on ambiguous (generic) names."""
        raw = (raw or "").strip()
        canonical, method, score = self._decide(raw, domain, context)
        self._record(raw, canonical, method, score)
        return canonical

    def _gate(self, canonical: str, method: str, score: float, norm: str, hint_domain: str | None,
              context: str | None, raw: str) -> tuple[str, str, float]:
        """Name-only matches on generic single-word seed names need corroboration."""
        if canonical not in self._ambiguous or " " in norm:
            return canonical, method, score
        hd = extract_domain(hint_domain) if hint_domain else None
        if hd and hd == self.canonicals[canonical]["domain"]:
            return canonical, f"{method}+domain", 100.0
        if context and AI_CONTEXT.search(context):
            return canonical, f"{method}+context", score
        alt = self._rejected.get(norm)
        if alt is None:
            alt = f"{self._clean_display(raw)} (unverified)"
            self._rejected[norm] = alt
        return alt, "ambiguous_rejected", 0.0

    def _decide(self, raw: str, hint_domain: str | None, context: str | None = None) -> tuple[str, str, float]:
        if not raw:
            return "", "empty", 0.0
        d = extract_domain(raw)
        if d and d in self._domain:
            return self._domain[d], "domain", 100.0
        norm = normalize(d.split(".")[0] if d else raw)
        if norm in self._exact:
            return self._gate(self._exact[norm], "exact", 100.0, norm, hint_domain, context, raw)
        cmp_ = compact(norm)
        if cmp_ in self._compact:
            return self._gate(self._compact[cmp_], "compact", 99.0, norm, hint_domain, context, raw)
        cr = core(norm).replace(" ", "")
        if cr in self._core:
            return self._gate(self._core[cr], "core", 95.0, norm, hint_domain, context, raw)
        best = self._fuzzy(norm)
        if best:
            return self._gate(*best, norm, hint_domain, context, raw)
        hd = extract_domain(hint_domain) if hint_domain else None
        if hd and hd in self._domain:
            return self._domain[hd], "domain_hint", 100.0
        canonical = self._clean_display(raw if not d else d.split(".")[0])
        self.add_entity(canonical, [raw], hd)
        return canonical, "new", 0.0

    def _fuzzy(self, norm: str) -> tuple[str, str, float] | None:
        """Typo-level matching only: plain ratio on compacted names (NOT token_set, which merges any
        shared-token pair), same first letter, and a stricter cutoff for short names."""
        cmp_ = compact(norm)
        if len(cmp_) < 7:
            return None
        cutoff = self.threshold if len(cmp_) >= 9 else max(self.threshold, 94.0)
        hit = process.extractOne(cmp_, self._compact.keys(), scorer=fuzz.ratio, score_cutoff=cutoff)
        if not hit:
            return None
        key, score, _ = hit
        if key[0] != cmp_[0] or min(len(key), len(cmp_)) / max(len(key), len(cmp_)) < 0.75:
            return None
        return self._compact[key], "fuzzy", float(score)

    @staticmethod
    def _clean_display(raw: str) -> str:
        toks = re.sub(r"\s+", " ", raw.strip(" ,.")).split(" ")
        while len(toks) > 1 and re.sub(r"[^a-z]", "", toks[-1].lower()) in LEGAL_SUFFIXES:
            toks.pop()
        return " ".join(toks).rstrip(",")

    # ---- log -------------------------------------------------------------------
    def _record(self, raw: str, canonical: str, method: str, score: float) -> None:
        metrics.inc(f"resolver.{method}")
        key = (raw, canonical)
        if key in self._seen:
            return
        self._seen.add(key)
        dec = Decision(raw, canonical, method, round(score, 1), datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        self.decisions.append(dec)
        log.info("entity_resolved", raw=raw, canonical=canonical, method=method, score=dec.score)

    def log_rows(self) -> list[list]:
        """Rows for the 'Entity Mapping Log' tab."""
        return [["raw_name", "canonical_name", "method", "score", "timestamp"],
                *[[d.raw, d.canonical, d.method, d.score, d.timestamp] for d in self.decisions]]

    def log_dicts(self) -> list[dict]:
        return [asdict(d) for d in self.decisions]
