"""Pydantic output schemas (schemaVersion 1.0). Field names are the contract."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Source(BaseModel):
    name: str
    url: str


class _Record(BaseModel):
    schemaVersion: str = SCHEMA_VERSION
    collectedAt: str = Field(default_factory=utcnow_iso)


# ---- STARTUP -------------------------------------------------------------
class StartupData(BaseModel):
    model_config = ConfigDict(extra="allow")
    employeeCount: int | None = None


class StartupContent(BaseModel):
    entityName: str
    data: StartupData


class StartupRecord(_Record):
    recordType: Literal["STARTUP"] = "STARTUP"
    source: Source
    content: StartupContent


# ---- PRODUCT -------------------------------------------------------------
class ProductContent(BaseModel):
    model_config = ConfigDict(extra="allow")
    startupName: str
    pricingModel: Literal["FREE", "FREEMIUM", "PAID", "ENTERPRISE"] | None = None


class ProductRecord(_Record):
    recordType: Literal["PRODUCT"] = "PRODUCT"
    source: Source
    content: ProductContent


# ---- RESEARCH_PAPER ------------------------------------------------------
class PaperContent(BaseModel):
    title: str
    authors: list[str]
    paper_url: str
    github_url: str | None = None
    github_stars: int | None = None
    published_date: str


class PaperRecord(_Record):
    recordType: Literal["RESEARCH_PAPER"] = "RESEARCH_PAPER"
    source: Source
    content: PaperContent


# ---- JOB -----------------------------------------------------------------
class JobContent(BaseModel):
    model_config = ConfigDict(extra="allow")
    company: str  # canonical, via resolver
    date: str
    is_remote: bool | None = None
    role_family: str | None = None


class JobRecord(_Record):
    recordType: Literal["JOB"] = "JOB"
    source: Source
    content: JobContent


# ---- NEWS ----------------------------------------------------------------
class NewsContent(BaseModel):
    title: str
    url: str
    source: str
    published_date: str
    full_text: str
    summary: str | None = None
    entities: list[str] = Field(default_factory=list)


class NewsRecord(_Record):
    recordType: Literal["NEWS"] = "NEWS"
    source: Source
    content: NewsContent


def to_payload(rec: BaseModel) -> dict[str, Any]:
    return rec.model_dump(mode="json")
