"""Token estimation and semantic chunking so we never send an oversized payload (413).

Order of operations for a payload that exceeds a provider's budget:
  1. strip boilerplate (trafilatura for HTML, line heuristics for text)
  2. semantic compression: keep title, headings, first paragraphs, then the
     highest-scoring 'key sentences' (numbers, entities, money, launch/funding verbs)
  3. if still too big: split into <=budget chunks on paragraph/sentence boundaries
     (caller runs each chunk and merges the results)
"""
from __future__ import annotations

import re

CHARS_PER_TOKEN = 3.5  # deliberately conservative (real ~4 for English)

_BOILERPLATE = re.compile(
    r"^(cookie|accept all|subscribe|sign up|sign in|log in|share this|follow us|advertisement|"
    r"read more|related (articles|posts)|privacy policy|terms of (use|service)|©|all rights reserved)",
    re.I,
)
_KEY_WORDS = re.compile(
    r"\b(raised?|funding|series [a-z]|seed|valuation|launch(ed|es)?|announce[ds]?|acquir\w+|"
    r"hiring|pricing|free|per month|enterprise|customers?|revenue|founded|ceo|model|benchmark)\b",
    re.I,
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def strip_boilerplate(text: str) -> str:
    if re.search(r"<(html|body|div|p|article)\b", text[:5000], re.I):
        try:
            import trafilatura

            extracted = trafilatura.extract(text, include_comments=False, include_tables=False)
            if extracted:
                text = extracted
        except Exception:  # trafilatura is best-effort
            pass
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or _BOILERPLATE.match(s):
            continue
        lines.append(s)
    return "\n\n".join(lines)


def _is_heading(p: str) -> bool:
    return p.startswith("#") or (len(p) < 90 and not p.endswith((".", "!", "?", ",", ";")))


def compress_text(text: str, budget_tokens: int, head_paragraphs: int = 3) -> str:
    """Keep title + headings + first N paragraphs + best sentences, in original order."""
    text = strip_boilerplate(text)
    if estimate_tokens(text) <= budget_tokens:
        return text
    paras = [p for p in text.split("\n\n") if p.strip()]
    if not paras:
        return text[: int(budget_tokens * CHARS_PER_TOKEN)]

    keep: dict[tuple[int, int], str] = {}  # (para_idx, sent_idx) -> sentence, ordered later
    used = 0

    def add(pi: int, si: int, s: str) -> bool:
        nonlocal used
        cost = estimate_tokens(s) + 1
        if (pi, si) in keep:
            return True
        if used + cost > budget_tokens:
            return False
        keep[(pi, si)] = s
        used += cost
        return True

    sents = [[s for s in _SENT_SPLIT.split(p) if s.strip()] for p in paras]
    add(0, 0, paras[0]) if len(paras[0]) < 300 else None  # title-ish first line
    for pi, p in enumerate(paras):
        if pi == 0 or _is_heading(p):
            add(pi, 0, p) if _is_heading(p) else None
    for pi in range(min(head_paragraphs, len(paras))):
        for si, s in enumerate(sents[pi]):
            if not add(pi, si, s):
                break

    scored = []
    for pi, ss in enumerate(sents):
        for si, s in enumerate(ss):
            if (pi, si) in keep:
                continue
            score = (
                2 * len(_KEY_WORDS.findall(s))
                + min(3, len(re.findall(r"\d", s)) // 2)
                + min(3, len(re.findall(r"\b[A-Z][a-z]+", s)) // 3)
                + (1 if "$" in s else 0)
            )
            scored.append((score, -pi, -si, pi, si, s))
    for _, _, _, pi, si, s in sorted(scored, reverse=True):
        add(pi, si, s)

    out, last_pi = [], None
    for (pi, _si), s in sorted(keep.items()):
        if last_pi is not None and pi != last_pi:
            out.append("\n\n")
        elif out:
            out.append(" ")
        out.append(s)
        last_pi = pi
    return "".join(out).strip()


def split_text(text: str, budget_tokens: int) -> list[str]:
    """Split into chunks each <= budget_tokens (hard guarantee)."""
    max_chars = max(200, int((budget_tokens - 1) * CHARS_PER_TOKEN) - 1)  # estimate_tokens adds +1
    pieces: list[str] = []
    for p in text.split("\n\n"):
        p = p.strip()
        if not p:
            continue
        if len(p) <= max_chars:
            pieces.append(p)
            continue
        for s in _SENT_SPLIT.split(p):  # oversize paragraph: sentences, then hard slice
            while len(s) > max_chars:
                pieces.append(s[:max_chars])
                s = s[max_chars:]
            if s.strip():
                pieces.append(s.strip())
    chunks, cur, cur_len = [], [], 0
    for piece in pieces:
        if cur and cur_len + len(piece) + 2 > max_chars:
            chunks.append("\n\n".join(cur))
            cur, cur_len = [], 0
        cur.append(piece)
        cur_len += len(piece) + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks or [""]


def prepare_payload(text: str, budget_tokens: int) -> list[str]:
    """Return 1+ pieces, each within budget. 1 piece if it already fits."""
    if estimate_tokens(text) <= budget_tokens:
        return [text]
    compressed = compress_text(text, budget_tokens)
    if estimate_tokens(compressed) <= budget_tokens:
        return [compressed]
    return split_text(strip_boilerplate(text), budget_tokens)
