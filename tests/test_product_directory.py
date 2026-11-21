import pytest

from src.crawlers.product_directory import DirectoryCrawler, Snapshot, compact, map_label, parse_cards
from src.resolver.entity_resolver import EntityResolver
from src.storage.db import Database
from src.storage.export_sheets import build_tabs


@pytest.mark.parametrize("label,expected", [
    ("100% Free", "FREE"), ("Free", "FREE"),
    ("Free + from $19.00/mo", "FREEMIUM"), ("Free + from $9.99/mo", "FREEMIUM"),
    ("From $5.90", "PAID"), ("From $149.00/yr", "PAID"), ("$20/mo", "PAID"), ("Free trial", "PAID"),
    ("Contact for pricing", "ENTERPRISE"),
    ("No pricing", None), ("", None), (None, None), ("???", None),
])
def test_label_map_is_deterministic(label, expected):
    assert map_label(label) == expected


CARD = """<body><ul>
<li class="li m verified" data-id="42" data-name="Nico &amp; Co" data-task="Task automation"  data-url="https://nico.app/?ref=taaft&amp;utm_source=taaft" >
<a class="ai_link new_tab" href="https://theresanaiforthat.com/ai/nico/"><span>Nico</span></a>
<div class="short_desc">A voice agent.</div>
<a class="ai_launch_date"  href="https://theresanaiforthat.com/ai/nico/">Free + from $20/mo</a></li>
<li class="li m" data-id="43" data-name="Plain" data-url="https://plain.io/x?utm_medium=y" >
<div class="short_desc">Plain tool</div><a class="ai_launch_date" href="x">No pricing</a></li></ul></body>"""


def test_parse_cards_extracts_label_site_and_strips_tracking():
    a, b = parse_cards(CARD)
    assert (a["id"], a["name"], a["website"], a["label"], a["slug"]) == ("42", "Nico & Co", "https://nico.app", "Free + from $20/mo", "nico")
    assert a["desc"] == "A voice agent." and a["task"] == "Task automation"
    assert b["website"] == "https://plain.io/x" and map_label(b["label"]) is None


def prod(name, domain=None, src="llm_guess", pm="PAID"):
    return {"schemaVersion": "1.0", "recordType": "PRODUCT", "collectedAt": "2026-11-21T00:00:00Z",
            "source": {"name": "Product Hunt", "url": "u"},
            "content": {"startupName": name, "pricingModel": pm, "pricingSource": src, "productName": name, "domain": domain}}


def test_snapshot_dedup_rules():
    s = Snapshot([("a", prod("Nico", "nico.app")), ("b", prod("Raft")), ("c", prod("Sierra", "sierra.ai"))])
    assert s.match("NICO", "nico.app") == "a"          # same domain
    assert s.match("Nico", None) == "a"                # same normalised name, no domain on the new side
    assert s.match("Raft", "raft.build") == "b"        # existing has no domain -> name decides
    assert s.match("Sierra", "sierra-pizza.com") is None  # same name, different domains -> different product
    assert compact("AI Flow-Chat") == "aiflowchat" and s.match("Unknown", None) is None


async def test_ingest_upgrades_existing_inserts_new_and_never_duplicates():
    db = Database("sqlite:///:memory:")
    await db.upsert_many([("PRODUCT", "ph:1", prod("Nico & Co", None, "llm_guess", "PAID"), "Product Hunt"),
                          ("PRODUCT", "ph:2", prod("Plain", "plain.io", "rule", "FREE"), "Product Hunt")])
    c = DirectoryCrawler(db, EntityResolver(), None, per_domain_rps=0)
    stats = {"new": 0, "upgraded": 0, "skipped_existing": 0, "llm_guess": 0}
    cards = parse_cards(CARD) + [{"id": "7", "name": "Brand New", "website": "https://brandnew.ai", "label": "100% Free",
                                  "desc": "x", "slug": "brand-new", "task": None}]
    await c._ingest(cards, stats)
    await c._ingest(cards, stats)  # second pass = idempotent
    rows = dict(await db.get_keyed("PRODUCT"))
    assert set(rows) == {"ph:1", "ph:2", "taaft:7"}  # no twins
    assert rows["ph:1"]["content"]["pricingSource"] == "directory_label" and rows["ph:1"]["content"]["pricingModel"] == "FREEMIUM"
    assert rows["ph:1"]["content"]["domain"] == "nico.app"  # domain learned from the directory
    assert rows["ph:2"]["content"]["pricingSource"] == "rule"  # grounded source never overwritten; "No pricing" ignored
    assert rows["taaft:7"]["content"]["pricingSource"] == "directory_label" and rows["taaft:7"]["content"]["pricingModel"] == "FREE"


async def test_export_products_tab_excludes_llm_guess():
    db = Database("sqlite:///:memory:")
    await db.upsert_many([
        ("PRODUCT", "1", prod("A", src="directory_label", pm="FREE"), "x"),
        ("PRODUCT", "2", prod("B", src="llm_guess", pm="PAID"), "x"),
        ("PRODUCT", "3", prod("C", src="site_pricing_page", pm="PAID"), "x"),
        ("PRODUCT", "4", prod("D", src=None, pm=None), "x"),
    ])
    rows = (await build_tabs(db))["Products"]
    i = rows[0].index("pricingSource") if "pricingSource" in rows[0] else rows[0].index("content.pricingSource")
    assert sorted(r[i] for r in rows[1:]) == ["directory_label", "site_pricing_page"]
    assert len(await db.get_records("PRODUCT")) == 4  # guesses stay in the DB
