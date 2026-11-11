"""Build a question-to-passage retrieval dataset from SEC 10-K filings.

Steps:
  1. Resolve tickers to CIKs and find each company's latest 10-K on EDGAR.
  2. Extract text, keep the sections most useful for deal research
     (Item 1 Business, 1A Risk Factors, 7 MD&A, 7A Market Risk) and chunk them.
  3. Ask an LLM (Groq free tier, OpenAI-compatible API) to write one realistic analyst
     question per passage.
  4. Split train/test BY COMPANY so test questions never come from companies seen in training.

Output: data/train.jsonl, data/test.jsonl
Optional: --push uploads the dataset to the Hugging Face Hub.

Usage:
  uv run --extra embeddings python embeddings/build_dataset.py --limit-companies 3   # smoke test
  uv run --extra embeddings python embeddings/build_dataset.py --push
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

DATA = Path(__file__).parent / "data"
SEC_HEADERS = {"User-Agent": os.environ.get("SEC_USER_AGENT", "")}
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Free-tier limits are per model (about 8K tokens/minute and 200K tokens/day each),
# so rotate across models and drop any model that hits its daily limit.
GROQ_MODELS = os.environ.get(
    "GROQ_MODELS", "openai/gpt-oss-20b,openai/gpt-oss-120b,qwen/qwen3.8-27b"
).split(",")
_rotation = itertools.count()


class DailyLimitReached(Exception):
    """Every model has used its free daily token allowance."""


def next_model() -> str:
    if not GROQ_MODELS:
        raise DailyLimitReached
    return GROQ_MODELS[next(_rotation) % len(GROQ_MODELS)]


# Diverse sectors so the model learns filing language, not one industry.
TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "CRM",
    "ADBE",
    "ORCL",
    "INTC",
    "JPM",
    "GS",
    "V",
    "MA",
    "PYPL",
    "BLK",
    "JNJ",
    "PFE",
    "UNH",
    "MRK",
    "XOM",
    "CVX",
    "NEE",
    "DUK",
    "WMT",
    "COST",
    "HD",
    "NKE",
    "MCD",
    "SBUX",
    "BA",
    "CAT",
    "GE",
    "LMT",
    "UPS",
    "TSLA",
    "F",
    "DIS",
    "NFLX",
    "T",
]
KEEP_SECTIONS = {"1", "1A", "7", "7A"}
ITEM_RE = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)\s*[\.:\-—]", re.IGNORECASE)
WORDS_MIN, WORDS_MAX = 120, 220
PASSAGES_PER_COMPANY = 60
# ~2K tokens per request, three models at ~8K tokens/minute each: about 12 requests/minute.
PACE_SECONDS = 5
SEC_DELAY = 0.15  # stay well under SEC's 10 requests/second limit


def with_retries(fn, what: str, tries: int = 6):
    """Retry on network errors (dropped Wi-Fi, DNS hiccups) with growing waits."""
    for attempt in range(tries):
        try:
            return fn()
        except httpx.TransportError as e:
            if attempt == tries - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"  network error on {what} ({e.__class__.__name__}); retrying in {wait}s")
            time.sleep(wait)


def sec_get(client: httpx.Client, url: str) -> httpx.Response:
    time.sleep(SEC_DELAY)
    r = with_retries(lambda: client.get(url, headers=SEC_HEADERS, timeout=60), url)
    r.raise_for_status()
    return r


def ticker_to_cik(client: httpx.Client) -> dict[str, str]:
    rows = sec_get(client, "https://www.sec.gov/files/company_tickers.json").json().values()
    return {row["ticker"]: str(row["cik_str"]).zfill(10) for row in rows}


def latest_10k_url(client: httpx.Client, cik: str) -> tuple[str, str] | None:
    recent = sec_get(client, f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    recent = recent["filings"]["recent"]
    for form, acc, doc, date in zip(
        recent["form"],
        recent["accessionNumber"],
        recent["primaryDocument"],
        recent["filingDate"],
        strict=False,
    ):
        if form == "10-K":
            url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{doc}"
            return url, date
    return None


def filing_lines(html: bytes | str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "table"]):  # tables chunk badly as prose
        tag.decompose()
    text = soup.get_text("\n")
    lines = (re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines())
    return [ln for ln in lines if ln]


def chunk_sections(lines: list[str]) -> list[tuple[str, str]]:
    """Return (section, passage) pairs from the sections we keep."""
    chunks: list[tuple[str, str]] = []
    section, buf = None, []

    def flush() -> None:
        if section in KEEP_SECTIONS and len(" ".join(buf).split()) >= WORDS_MIN:
            chunks.append((section, " ".join(buf)))

    for line in lines:
        m = ITEM_RE.match(line)
        if m:
            flush()
            section, buf = m.group(1).upper(), []
            continue
        if len(line.split()) < 6:  # page numbers, headers, short headings
            continue
        buf.append(line)
        if len(" ".join(buf).split()) >= WORDS_MAX:
            flush()
            buf = []
    flush()
    return chunks


QUESTION_PROMPT = """You write search queries that a private-equity analyst would type when \
researching a company in its 10-K. For EACH numbered passage, write ONE specific question that \
the passage answers. Do not copy long phrases from the passage. Mention the company name.

Company: {company}
{passages}

Reply with JSON only: {{"questions": ["...", "..."]}} with exactly {n} questions in order."""


def generate_questions(client: httpx.Client, company: str, passages: list[str]) -> list[str] | None:
    numbered = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    body = {
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": QUESTION_PROMPT.format(
                    company=company, passages=numbered, n=len(passages)
                ),
            }
        ],
    }
    headers = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
    reason = "unknown"
    for _attempt in range(12):
        model = next_model()
        extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
        payload = {**body, **extra, "model": model}
        r = with_retries(
            lambda p=payload: client.post(GROQ_URL, json=p, headers=headers, timeout=120), "Groq"
        )
        if r.status_code == 429:
            reason = "rate limited"
            if "tokens per day" in r.text:  # out for today: stop using this model
                GROQ_MODELS.remove(model)
                print(f"  {model} reached its daily limit; {len(GROQ_MODELS)} models left")
            else:  # per-minute limit: try the next model, pausing only briefly
                time.sleep(min(float(r.headers.get("retry-after", 5)), 10))
            continue
        if r.status_code >= 400:
            reason = f"HTTP {r.status_code}: {r.text[:200]}"
            time.sleep(5)
            continue
        try:
            questions = json.loads(r.json()["choices"][0]["message"]["content"])["questions"]
        except (json.JSONDecodeError, KeyError, TypeError):
            reason = "invalid JSON"
            continue
        if len(questions) == len(passages) and all(isinstance(q, str) for q in questions):
            time.sleep(PACE_SECONDS)
            return [q.strip() for q in questions]
        reason = f"expected {len(passages)} questions, got {len(questions)}"
    print(f"  {company}: skipped a batch of {len(passages)} ({reason})", flush=True)
    return None


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-companies", type=int, default=len(TICKERS))
    ap.add_argument("--test-companies", type=int, default=8)
    ap.add_argument("--batch", type=int, default=5, help="passages per LLM request")
    ap.add_argument("--push", action="store_true", help="upload to the Hugging Face Hub")
    ap.add_argument(
        "--repo", default=os.environ.get("HF_DATASET_REPO", "jc4845/sec-10k-retrieval-qa")
    )
    args = ap.parse_args()

    if not SEC_HEADERS["User-Agent"]:
        raise SystemExit(
            "Set SEC_USER_AGENT in .env, e.g. 'dealscope-agent your-email@example.com'"
        )
    if "GROQ_API_KEY" not in os.environ:
        raise SystemExit("Set GROQ_API_KEY in .env (free key from console.groq.com)")

    DATA.mkdir(exist_ok=True)
    random.seed(42)
    rows: list[dict] = []
    cache = DATA / "cache"  # one file per company, so an interrupted run can resume
    cache.mkdir(exist_ok=True)
    with httpx.Client() as client:
        ciks = ticker_to_cik(client)
        for ticker in TICKERS[: args.limit_companies]:
            cached = cache / f"{ticker}.jsonl"
            if cached.exists():
                rows += [json.loads(line) for line in cached.open(encoding="utf-8")]
                print(f"{ticker}: loaded from cache")
                continue
            found = latest_10k_url(client, ciks[ticker])
            if not found:
                print(f"{ticker}: no 10-K found, skipped")
                continue
            url, date = found
            chunks = chunk_sections(filing_lines(sec_get(client, url).content))
            if len(chunks) > PASSAGES_PER_COMPANY:
                chunks = random.Random(ticker).sample(chunks, PASSAGES_PER_COMPANY)
            print(f"{ticker}: {len(chunks)} passages from 10-K filed {date}", flush=True)
            company_rows: list[dict] = []
            for i in range(0, len(chunks), args.batch):
                batch = chunks[i : i + args.batch]
                try:
                    questions = generate_questions(client, ticker, [p for _, p in batch])
                except DailyLimitReached:
                    break
                if questions is None:
                    continue
                print(f"  {ticker}: {i + len(batch)}/{len(chunks)}", flush=True)
                for (section, passage), q in zip(batch, questions, strict=True):
                    company_rows.append(
                        {
                            "id": f"{ticker}-{len(company_rows)}",
                            "ticker": ticker,
                            "section": f"Item {section}",
                            "filing_date": date,
                            "source_url": url,
                            "question": q,
                            "passage": passage,
                        }
                    )
            if not GROQ_MODELS:  # partial company is not cached, so a later run redoes it
                print(f"Daily LLM limit reached during {ticker}; stopping here.")
                break
            write_jsonl(cached, company_rows)
            rows += company_rows

    tickers = sorted({r["ticker"] for r in rows})
    random.shuffle(tickers)
    test_set = set(tickers[: args.test_companies]) if len(tickers) > 1 else set()
    train = [r for r in rows if r["ticker"] not in test_set]
    test = [r for r in rows if r["ticker"] in test_set]
    write_jsonl(DATA / "train.jsonl", train)
    write_jsonl(DATA / "test.jsonl", test)
    print(
        f"{len(rows)} pairs: {len(train)} train, {len(test)} test",
        f"(test companies: {sorted(test_set)})",
    )

    if args.push:
        from datasets import load_dataset

        ds = load_dataset(
            "json",
            data_files={"train": str(DATA / "train.jsonl"), "test": str(DATA / "test.jsonl")},
        )
        ds.push_to_hub(args.repo, token=os.environ.get("HF_TOKEN"))
        print(f"pushed to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
