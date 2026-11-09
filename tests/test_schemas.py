from datetime import date

import pytest
from pydantic import ValidationError

from agent.schemas import Claim, CompanyBrief, Source, SourceType

SRC = Source(
    type=SourceType.SEC_FILING,
    url="https://www.sec.gov/Archives/edgar/data/320193/",
    title="10-K",
    retrieved_on=date(2026, 11, 9),
)


def test_brief_accepts_cited_summary_and_pads_cik():
    brief = CompanyBrief(
        company_name="Apple Inc.",
        cik="320193",
        summary=Claim(text="Designs consumer electronics.", sources=[SRC]),
        confidence=0.8,
    )
    assert brief.cik == "0000320193"


def test_claim_without_source_is_rejected():
    with pytest.raises(ValidationError):
        Claim(text="Unsupported statement", sources=[])


def test_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValidationError):
        CompanyBrief(
            company_name="X",
            summary=Claim(text="y", sources=[SRC]),
            confidence=1.5,
        )
