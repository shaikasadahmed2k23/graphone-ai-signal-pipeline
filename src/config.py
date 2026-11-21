"""Central settings, loaded from environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    gemini_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    groq_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    openrouter_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    product_hunt_token: str = field(default_factory=lambda: os.getenv("PRODUCTHUNT_TOKEN") or os.getenv("PRODUCT_HUNT_TOKEN", ""))
    github_token: str = field(default_factory=lambda: os.getenv("GITHUB_TOKEN", ""))
    google_credentials: str = field(default_factory=lambda: os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", ""))
    google_sheet_id: str = field(default_factory=lambda: os.getenv("GOOGLE_SHEET_ID", ""))
    db_url: str = field(default_factory=lambda: os.getenv("DB_URL", "sqlite:///data/pipeline.db"))
    concurrency: int = field(default_factory=lambda: _int("CONCURRENCY", 16))
    per_domain_rps: int = field(default_factory=lambda: _int("PER_DOMAIN_RPS", 4))


def get_settings() -> Settings:
    return Settings()
