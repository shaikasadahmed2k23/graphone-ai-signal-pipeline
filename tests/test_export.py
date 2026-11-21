import csv
import json
from datetime import datetime, timedelta, timezone

from src.storage.db import Database
from src.storage.export_sheets import MIN_STARTUPS, build_tabs, write_csvs, write_sheet

NOW = datetime(2026, 11, 21, 12, 0, tzinfo=timezone.utc)


def iso(h):
    return (NOW - timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")


def startup(i, is_ai):
    return {"schemaVersion": "1.0", "recordType": "STARTUP", "collectedAt": iso(0), "source": {"name": "YC Directory", "url": f"u{i}"},
            "content": {"entityName": f"S{i}", "data": {"employeeCount": i, "is_ai": is_ai}}}


async def seeded(n_ai):
    db = Database("sqlite:///:memory:")
    await db.upsert_many([("STARTUP", f"s{i}", startup(i, i < n_ai), "yc") for i in range(1200)])
    base = {"schemaVersion": "1.0", "collectedAt": iso(0), "source": {"name": "x", "url": "u"}}
    await db.upsert_many([
        ("JOB", "j1", {**base, "recordType": "JOB", "content": {"company": "A", "date": iso(3), "is_remote": True, "role_family": "Data"}}, "x"),
        ("JOB", "j2", {**base, "recordType": "JOB", "content": {"company": "B", "date": iso(30), "is_remote": True, "role_family": "Data"}}, "x"),
        ("NEWS", "n1", {**base, "recordType": "NEWS", "content": {"title": "t", "url": "u", "source": "s", "published_date": iso(2),
                                                                  "full_text": "x" * 60000, "summary": "s", "entities": ["A"]}}, "x"),
        ("NEWS", "n2", {**base, "recordType": "NEWS", "content": {"title": "old", "url": "u2", "source": "s", "published_date": iso(50),
                                                                  "full_text": "x", "summary": "s", "entities": []}}, "x"),
        ("ENTITY_LOG", "e1", {"raw_name": "Sierra", "canonical_name": "Sierra (unverified)", "method": "ambiguous_rejected",
                              "score": 0, "timestamp": iso(1)}, "r"),
    ])
    return db


async def test_all_ai_startups_when_enough():
    db = await seeded(1100)
    tabs = await build_tabs(db, NOW)
    rows = tabs["Startups"]
    assert len(rows) - 1 == 1100
    i = rows[0].index("is_ai")
    assert all(r[i] is True for r in rows[1:])


async def test_fills_with_non_ai_and_marks_when_short():
    db = await seeded(300)
    tabs = await build_tabs(db, NOW)
    rows = tabs["Startups"]
    i = rows[0].index("is_ai")
    assert len(rows) - 1 == MIN_STARTUPS
    assert sum(1 for r in rows[1:] if r[i] is True) == 300 and sum(1 for r in rows[1:] if r[i] is False) == 700


async def test_strict_24h_at_export_and_cell_limits(tmp_path):
    db = await seeded(1100)
    tabs = await build_tabs(db, NOW)
    assert len(tabs["Jobs"]) == 2 and tabs["Jobs"][1][tabs["Jobs"][0].index("company")] == "A"
    assert len(tabs["News"]) == 2
    for row in tabs["News"][1:]:
        assert all(len(str(c)) <= 50_000 for c in row)
        rec = json.loads(row[-1])
        assert rec["recordType"] == "NEWS" and rec["schemaVersion"] == "1.0"
    assert tabs["Entity Mapping Log"][0] == ["raw_name", "canonical_name", "method", "score", "timestamp"]
    paths = write_csvs(tabs, str(tmp_path))
    assert len(paths) == 6
    with open(tmp_path / "Jobs.csv", encoding="utf-8") as f:
        assert len(list(csv.reader(f))) == 2


def test_write_sheet_uses_one_tab_per_name_and_shares_publicly():
    class WS:
        def __init__(self, t): self.title, self.calls = t, []
        def clear(self): pass
        def resize(self, rows, cols): pass
        def update(self, values, range_name, value_input_option): self.calls.append((range_name, len(values)))
        def freeze(self, rows): pass
        def get_all_values(self): return [[]]

    class SH:
        url = "https://sheet"
        def __init__(self): self.ws, self.shared = {}, None
        def worksheets(self): return list(self.ws.values())
        def add_worksheet(self, title, rows, cols): self.ws[title] = WS(title); return self.ws[title]
        def del_worksheet(self, w): self.ws.pop(w.title)
        def share(self, v, perm_type, role): self.shared = (perm_type, role)

    sh = SH()
    class GC:
        def open_by_key(self, k): return sh
    tabs = {"A": [["h"]] + [[i] for i in range(4500)], "B": [["h"], [1]]}
    url = write_sheet(tabs, "id", "creds", gspread_client=GC())
    assert url == "https://sheet" and sh.shared == ("anyone", "reader")
    assert [c[1] for c in sh.ws["A"].calls] == [2000, 2000, 501]  # chunked writes
