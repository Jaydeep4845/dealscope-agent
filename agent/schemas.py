"""Output contract for the research agent.

Every factual claim carries at least one source so a reviewer can verify it.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl, field_validator


class SourceType(StrEnum):
    SEC_FILING = "sec_filing"
    NEWS = "news"
    COMPANY_SITE = "company_site"
    DATA_PROVIDER = "data_provider"


class Source(BaseModel):
    type: SourceType
    url: HttpUrl
    title: str
    retrieved_on: date


class Claim(BaseModel):
    text: str = Field(min_length=1)
    sources: list[Source] = Field(min_length=1)


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Risk(BaseModel):
    description: str
    level: RiskLevel
    sources: list[Source] = Field(min_length=1)


class Financials(BaseModel):
    fiscal_year: int | None = None
    revenue_usd: float | None = Field(default=None, ge=0)
    net_income_usd: float | None = None
    revenue_growth_pct: float | None = None


class CompanyBrief(BaseModel):
    company_name: str
    ticker: str | None = None
    cik: str | None = Field(
        default=None, description="SEC Central Index Key, zero-padded to 10 digits"
    )
    summary: Claim
    business_model: list[Claim] = Field(default_factory=list)
    financials: Financials = Field(default_factory=Financials)
    recent_developments: list[Claim] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @field_validator("cik")
    @classmethod
    def normalise_cik(cls, v: str | None) -> str | None:
        if v is None:
            return v
        digits = v.strip().lstrip("CIK").strip()
        if not digits.isdigit() or len(digits) > 10:
            raise ValueError("CIK must be up to 10 digits")
        return digits.zfill(10)
