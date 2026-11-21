import pytest

from src.resolver.entity_resolver import EntityResolver, extract_domain, normalize


@pytest.fixture
def r():
    return EntityResolver()


@pytest.mark.parametrize("raw,canonical", [
    ("OpenAI", "OpenAI"),
    ("OpenAI, Inc.", "OpenAI"),
    ("Open AI", "OpenAI"),
    ("openai.com", "OpenAI"),
    ("https://www.openai.com/careers", "OpenAI"),
    ("Anthropic PBC", "Anthropic"),
    ("Anthropic", "Anthropic"),
    ("Hugging Face", "Hugging Face"),
    ("HuggingFace", "Hugging Face"),
    ("huggingface.co", "Hugging Face"),
    ("Mistral AI", "Mistral AI"),
    ("Mistral", "Mistral AI"),
    ("MISTRAL AI SAS", "Mistral AI"),
    ("Character AI", "Character.AI"),
    ("Weights and Biases", "Weights & Biases"),
])
def test_known_entities_resolve(r, raw, canonical):
    assert r.resolve(raw) == canonical


def test_every_decision_is_logged_with_method_and_score(r):
    r.resolve("OpenAI, Inc.")
    r.resolve("openai.com")
    r.resolve("Open AI")
    r.resolve("HuggingFace")
    by_raw = {d.raw: d for d in r.decisions}
    assert by_raw["OpenAI, Inc."].method == "exact" and by_raw["OpenAI, Inc."].score == 100
    assert by_raw["openai.com"].method == "domain"
    assert by_raw["HuggingFace"].method in ("exact", "compact") and by_raw["HuggingFace"].canonical == "Hugging Face"
    rows = r.log_rows()
    assert rows[0] == ["raw_name", "canonical_name", "method", "score", "timestamp"] and len(rows) == 5


def test_unknown_becomes_new_canonical_and_later_aliases_resolve_to_it(r):
    assert r.resolve("Zorblax Robotics, Inc.") == "Zorblax Robotics"
    assert r.decisions[-1].method == "new"
    assert r.resolve("zorblax robotics") == "Zorblax Robotics"
    assert r.decisions[-1].method == "exact"


def test_domain_hint_used_when_names_differ(r):
    assert r.resolve("Totally Unrelated Brand Name", domain="https://cursor.com") == "Anysphere"
    assert r.decisions[-1].method == "domain_hint"


def test_fuzzy_typo_matches(r):
    assert r.resolve("Anthropik") == "Anthropic"
    assert r.decisions[-1].method == "fuzzy" and r.decisions[-1].score >= 88


def test_no_false_merge_on_subset_or_short_names(r):
    assert r.resolve("Open") != "OpenAI"            # subset token must not merge
    assert r.resolve("Scale Foundation Fund Of America") != "Scale AI"
    assert r.resolve("Ai") not in ("OpenAI", "Mistral AI")
    assert r.resolve("") == ""


def test_normalize_and_domain():
    assert normalize("  Acme, Inc.  ") == "acme"
    assert normalize("Müller & Söhne GmbH") == "muller and sohne"
    assert extract_domain("https://www.bbc.co.uk/news") == "bbc.co.uk"
    assert extract_domain("Open AI") is None
    assert extract_domain("OpenAI, Inc.") is None


# ---- ambiguous generic names -------------------------------------------------
@pytest.mark.parametrize("raw", ["Sierra", "Lambda", "Modal", "Scale", "Together", "Perplexity", "Magic", "Cursor"])
def test_bare_generic_name_alone_is_rejected(r, raw):
    out = r.resolve(raw)
    assert out == f"{raw} (unverified)"
    d = r.decisions[-1]
    assert d.method == "ambiguous_rejected" and d.score == 0 and d.raw == raw


def test_ambiguous_accepted_with_domain_or_ai_context(r):
    assert r.resolve("Sierra", domain="https://sierra.ai") == "Sierra"
    assert r.decisions[-1].method == "exact+domain"
    assert r.resolve("Lambda", context="GPU cloud for AI training and inference") == "Lambda"
    assert r.decisions[-1].method == "exact+context"
    assert r.resolve("Modal", context="Pottery studio and ceramics classes") == "Modal (unverified)"


def test_ambiguous_wrong_domain_does_not_help(r):
    assert r.resolve("Sierra", domain="sierrawireless.com") == "Sierra (unverified)"


def test_explicit_full_names_are_not_ambiguous(r):
    assert r.resolve("Sierra AI") == "Sierra"
    assert r.resolve("Together AI") == "Together AI"
    assert r.resolve("Scale AI") == "Scale AI"
    assert r.resolve("Perplexity AI") == "Perplexity"
    assert r.decisions[-1].method in ("exact", "core")


def test_rejected_is_consistent_and_distinct(r):
    a, b = r.resolve("Sierra"), r.resolve("sierra")
    assert a == b == "Sierra (unverified)" and a != r.resolve("Sierra AI")


def test_fuzzy_does_not_merge_lookalike_short_names():
    r = EntityResolver()
    r.add_entity("Trata Systems", [], None)
    r.add_entity("Punchcard", [], None)
    r.add_entity("Keiki Labs", [], None)
    for raw, bad in [("Strata", "Trata Systems"), ("AgentLive", "Punchcard"), ("Zetro", "Keiki Labs")]:
        assert r.resolve(raw, context="AI tool") != bad
