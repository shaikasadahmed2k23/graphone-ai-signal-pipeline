"""Validation models for LLM outputs (never trust model JSON blindly)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

Pricing = Literal["FREE", "FREEMIUM", "PAID", "ENTERPRISE"]


class PricingItem(BaseModel):
    id: str
    pricingModel: Pricing | None = None

    @field_validator("id", mode="before")
    @classmethod
    def _id_str(cls, v):
        return str(v)

    @field_validator("pricingModel", mode="before")
    @classmethod
    def _norm(cls, v):
        if v is None:
            return None
        v = str(v).strip().upper()
        return None if v in ("", "NULL", "NONE", "UNKNOWN") else v


class PricingBatch(BaseModel):
    results: list[PricingItem]
