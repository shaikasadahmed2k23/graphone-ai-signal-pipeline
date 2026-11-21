"""SQLite state store: records (idempotent upsert), checkpoints (resume), llm_cache.

Swappable to Postgres: all SQL here is portable (ON CONFLICT ... DO UPDATE, TEXT columns);
only the connection factory and the '?' placeholder style (-> %s) change. Everything goes through
the `Database` class, so a `PostgresDatabase` with the same async API can replace it.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    record_type   TEXT NOT NULL,
    dedup_key     TEXT NOT NULL,
    payload       TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    source_name   TEXT,
    first_seen_at TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (record_type, dedup_key)
);
CREATE TABLE IF NOT EXISTS checkpoints (
    job        TEXT PRIMARY KEY,
    state      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_cache (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _content_hash(payload: dict[str, Any]) -> str:
    stable = {k: v for k, v in payload.items() if k != "collectedAt"}  # volatile field
    return hashlib.sha256(json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()


class Database:
    def __init__(self, url: str = "sqlite:///data/pipeline.db") -> None:
        if not url.startswith("sqlite:///"):
            raise ValueError("only sqlite:/// implemented; add PostgresDatabase with the same API")
        path = url[len("sqlite:///"):]
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # ---- sync primitives ---------------------------------------------------
    def _upsert(self, record_type: str, dedup_key: str, payload: dict, source_name: str | None) -> str:
        h, now = _content_hash(payload), _now()
        with self._lock:
            row = self._conn.execute(
                "SELECT content_hash FROM records WHERE record_type=? AND dedup_key=?", (record_type, dedup_key)
            ).fetchone()
            self._conn.execute(
                """INSERT INTO records(record_type, dedup_key, payload, content_hash, source_name, first_seen_at, updated_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(record_type, dedup_key) DO UPDATE SET
                     payload=excluded.payload, content_hash=excluded.content_hash,
                     updated_at=excluded.updated_at
                   WHERE records.content_hash <> excluded.content_hash""",
                (record_type, dedup_key, json.dumps(payload, default=str), h, source_name, now, now),
            )
            self._conn.commit()
        if row is None:
            return "inserted"
        return "unchanged" if row["content_hash"] == h else "updated"

    def _upsert_many(self, items: list[tuple[str, str, dict, str | None]]) -> dict[str, int]:
        out = {"inserted": 0, "updated": 0, "unchanged": 0}
        for it in items:
            out[self._upsert(*it)] += 1
        return out

    def _q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def _exec(self, sql: str, args: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, args)
            self._conn.commit()

    # ---- async API (what crawlers use) --------------------------------------
    async def upsert_record(self, record_type: str, dedup_key: str, payload: dict, source_name: str | None = None) -> str:
        return await asyncio.to_thread(self._upsert, record_type, dedup_key, payload, source_name)

    async def upsert_many(self, items: list[tuple[str, str, dict, str | None]]) -> dict[str, int]:
        return await asyncio.to_thread(self._upsert_many, items)

    async def get_records(self, record_type: str) -> list[dict]:
        rows = await asyncio.to_thread(
            self._q, "SELECT payload FROM records WHERE record_type=? ORDER BY first_seen_at, dedup_key", (record_type,)
        )
        return [json.loads(r["payload"]) for r in rows]

    async def get_keyed(self, record_type: str) -> list[tuple[str, dict]]:
        rows = await asyncio.to_thread(
            self._q, "SELECT dedup_key, payload FROM records WHERE record_type=? ORDER BY first_seen_at, dedup_key", (record_type,)
        )
        return [(r["dedup_key"], json.loads(r["payload"])) for r in rows]

    async def count(self, record_type: str) -> int:
        rows = await asyncio.to_thread(self._q, "SELECT COUNT(*) c FROM records WHERE record_type=?", (record_type,))
        return rows[0]["c"]

    async def get_checkpoint(self, job: str) -> dict:
        rows = await asyncio.to_thread(self._q, "SELECT state FROM checkpoints WHERE job=?", (job,))
        return json.loads(rows[0]["state"]) if rows else {}

    async def set_checkpoint(self, job: str, state: dict) -> None:
        await asyncio.to_thread(
            self._exec,
            """INSERT INTO checkpoints(job, state, updated_at) VALUES (?,?,?)
               ON CONFLICT(job) DO UPDATE SET state=excluded.state, updated_at=excluded.updated_at""",
            (job, json.dumps(state), _now()),
        )

    async def clear_checkpoint(self, job: str) -> None:
        await asyncio.to_thread(self._exec, "DELETE FROM checkpoints WHERE job=?", (job,))

    # ---- llm cache (satisfies orchestrator.Cache) ----------------------------
    async def get(self, key: str) -> str | None:
        rows = await asyncio.to_thread(self._q, "SELECT value FROM llm_cache WHERE key=?", (key,))
        return rows[0]["value"] if rows else None

    async def set(self, key: str, value: str) -> None:
        await asyncio.to_thread(
            self._exec,
            "INSERT INTO llm_cache(key,value,created_at) VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value, _now()),
        )
