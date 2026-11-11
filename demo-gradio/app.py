"""SEC filing search demo: fine-tuned vs base embedding model, side by side."""

from __future__ import annotations

import html
import json
from pathlib import Path

import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer

DATA = Path(__file__).parent / "data"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MODELS = {"tuned": "jc4845/bge-small-sec-filings", "base": "BAAI/bge-small-en-v1.5"}
TOP_K = 5

NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "GOOGL": "Alphabet (Google)",
    "META": "Meta",
    "CRM": "Salesforce",
    "ADBE": "Adobe",
    "ORCL": "Oracle",
    "JPM": "JPMorgan Chase",
    "GS": "Goldman Sachs",
    "V": "Visa",
    "MA": "Mastercard",
    "PYPL": "PayPal",
    "BLK": "BlackRock",
    "JNJ": "Johnson & Johnson",
    "PFE": "Pfizer",
    "UNH": "UnitedHealth",
    "MRK": "Merck",
    "CVX": "Chevron",
    "NEE": "NextEra Energy",
    "DUK": "Duke Energy",
    "WMT": "Walmart",
    "COST": "Costco",
    "HD": "Home Depot",
}
SECTIONS = {
    "Item 1": "Business",
    "Item 1A": "Risk Factors",
    "Item 7": "MD&A",
    "Item 7A": "Market Risk",
}

# ---------------------------------------------------------------- load once at startup
PASSAGES = [json.loads(line) for line in (DATA / "passages.jsonl").open(encoding="utf-8")]
HELD_OUT = set(json.loads((DATA / "meta.json").read_text())["held_out"])
TICKERS = np.array([p["ticker"] for p in PASSAGES])
EMB = {name: np.load(DATA / f"emb_{name}.npy").astype(np.float32) for name in MODELS}
ENCODERS = {name: SentenceTransformer(model_id) for name, model_id in MODELS.items()}


def label(ticker: str) -> str:
    tag = "  (unseen in training)" if ticker in HELD_OUT else ""
    return f"{NAMES.get(ticker, ticker)} ({ticker}){tag}"


CHOICES = ["All companies"] + [
    label(t) for t in sorted(set(TICKERS), key=lambda t: NAMES.get(t, t))
]
LABEL_TO_TICKER = {label(t): t for t in set(TICKERS)}


def search(name: str, question: str, company: str) -> list[tuple[int, float]]:
    q = ENCODERS[name].encode(QUERY_PREFIX + question, normalize_embeddings=True)
    scores = EMB[name] @ q  # cosine similarity (vectors are normalised)
    if company != "All companies":
        scores = np.where(TICKERS == LABEL_TO_TICKER[company], scores, -np.inf)
    top = np.argsort(-scores)[:TOP_K]
    return [(int(i), float(scores[i])) for i in top]


def render(hits: list[tuple[int, float]], other_top: int, title: str, accent: str) -> str:
    cards = []
    for rank, (i, score) in enumerate(hits, 1):
        p = PASSAGES[i]
        sec = f"{p['section']} · {SECTIONS.get(p['section'], '')}"
        text = html.escape(p["passage"][:420]) + ("…" if len(p["passage"]) > 420 else "")
        badge = ""
        if rank == 1 and i != other_top:
            badge = "<span class='diff'>different #1 from the other model</span>"
        cards.append(f"""
<div class='card'>
  <div class='meta'><b>#{rank}</b> · {html.escape(NAMES.get(p["ticker"], p["ticker"]))} · {sec}
    <span class='score'>similarity {score:.3f}</span></div>
  <div class='text'>{text}</div>
  <div class='foot'>{badge}<a href='{p["source_url"]}' target='_blank'>Open the 10-K on SEC EDGAR ↗</a></div>
</div>""")
    return f"<div class='col'><h3 style='color:{accent}'>{title}</h3>{''.join(cards)}</div>"


def run(question: str, company: str):
    question = (question or "").strip()
    if not question:
        return "<p>Type a question to search.</p>", ""
    tuned, base = search("tuned", question, company), search("base", question, company)
    return (
        render(tuned, base[0][0], "Fine-tuned model (jc4845/bge-small-sec-filings)", "#15803d"),
        render(base, tuned[0][0], "Base model (BAAI/bge-small-en-v1.5)", "#6b7280"),
    )


EXAMPLES = [
    # Held-out benchmark questions where the fine-tuned model ranks the correct passage first
    # and the base model does not (35 such cases vs 10 the other way, searching within each company).
    [
        "How does Microsoft assess the potential impact of security vulnerabilities on its reputation "
        "and operating margins?",
        label("MSFT"),
    ],
    [
        "How might META's ability to attract and retain technical talent affect its user engagement and "
        "advertising revenue?",
        label("META"),
    ],
    [
        "Which macro-economic and geopolitical factors does Oracle identify as risks that could reduce "
        "customer IT spending?",
        label("ORCL"),
    ],
    [
        "What intellectual property risks does JNJ face, particularly regarding patent challenges and "
        "potential generic or biosimilar competition?",
        label("JNJ"),
    ],
    # Free-form questions
    ["How do changes in oil and gas prices affect earnings?", label("CVX")],
    ["How does the company manage interest rate risk?", "All companies"],
]

CSS = """
.col h3 {margin: 4px 0 10px; font-size: 15px}
.card {border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 10px 12px; margin-bottom: 10px;
       background: var(--background-fill-secondary)}
.meta {font-size: 13px; margin-bottom: 6px}
.score {float: right; opacity: .7; font-family: monospace}
.text {font-size: 13.5px; line-height: 1.45}
.foot {margin-top: 6px; font-size: 12.5px; display: flex; justify-content: space-between; gap: 8px}
.diff {background: #fef3c7; color: #92400e; border-radius: 6px; padding: 0 6px}
"""

INTRO = """
# SEC Filing Search: fine-tuned vs base embeddings

Ask a question about a US public company. Both models search the same **1,422 passages from 10-K
filings** of 24 companies, and you see their top results side by side. The fine-tuned model was
trained on question-passage pairs from 18 companies; companies marked **(unseen in training)** were
held out for evaluation, so they are the fairest test. The first four examples are real benchmark
questions where the two models disagree. Similarity scores are on different scales for the two models,
so compare the passages, not the numbers.
"""

RESULTS = """
### Benchmark on held-out companies

355 questions from 6 companies never seen in training (Chevron, Johnson & Johnson, Meta, Microsoft,
Oracle, Walmart). Each question is ranked against all 355 passages from those companies.

| Metric | Base bge-small | Fine-tuned | Change |
|---|---|---|---|
| Recall@1 (correct passage ranked first) | 0.730 | **0.797** | +0.068 |
| MRR@10 | 0.816 | **0.868** | +0.052 |
| NDCG@10 | 0.854 | **0.898** | +0.043 |
| Recall@10 | 0.972 | **0.989** | +0.017 |

**Links:** [model](https://huggingface.co/jc4845/bge-small-sec-filings) ·
[dataset](https://huggingface.co/datasets/jc4845/sec-10k-retrieval-qa) ·
[code](https://github.com/Jaydeep4845/dealscope-agent)

**Limitations:** training questions are LLM-generated; only narrative text from 10-K Items 1, 1A, 7
and 7A; free-form questions here have no ground truth, so judge the results by reading them.
"""

with gr.Blocks(title="SEC Filing Search") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        company = gr.Dropdown(CHOICES, value=label("MSFT"), label="Company", scale=2)
        question = gr.Textbox(
            label="Question",
            placeholder="e.g. What supply-chain risks does the company face?",
            scale=3,
        )
    go = gr.Button("Search", variant="primary")
    gr.Examples(EXAMPLES, inputs=[question, company], label="Try an example")
    with gr.Row(equal_height=False):
        out_tuned = gr.HTML()
        out_base = gr.HTML()
    with gr.Accordion("How well does it work? (benchmark and links)", open=False):
        gr.Markdown(RESULTS)
    go.click(run, [question, company], [out_tuned, out_base])
    question.submit(run, [question, company], [out_tuned, out_base])

if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())  # Gradio 6: styling is passed to launch()
