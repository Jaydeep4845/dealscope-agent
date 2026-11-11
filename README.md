# DealScope Agent

**A LangGraph research agent that turns a company name into a validated, source-cited deal-sourcing brief.**

![CI](https://github.com/Jaydeep4845/dealscope-agent/actions/workflows/ci.yml/badge.svg)
![Status](https://img.shields.io/badge/status-active%20development-orange)

**Live demo:** [SEC Filing Search](https://huggingface.co/spaces/jc4845/sec-filing-search), the agent's
filings-retrieval model running in your browser ·
[model](https://huggingface.co/jc4845/bge-small-sec-filings) ·
[dataset](https://huggingface.co/datasets/jc4845/sec-10k-retrieval-qa)

## The problem

Before a deal team spends time on a company, an analyst assembles a first-pass profile: what the company does,
its size and financial trend, recent news, and red flags. Gathering that from filings, news and data providers
takes hours per company and the output is inconsistent from one analyst to the next.

DealScope automates the first pass. It researches a company through tools, cites every claim, returns a
schema-validated brief that downstream systems can consume, and asks a human to approve before anything is saved.

## Architecture

```mermaid
flowchart LR
    U[Company name] --> P[Planner node]
    P --> T{Tool calls}
    T --> W[Web search]
    T --> M[SEC filings MCP server]
    T --> F[Financial data API]
    M --> E[(EDGAR 10-K / 10-Q<br/>hybrid search: BM25 + pgvector)]
    W & M & F --> S[Synthesizer node]
    S --> V{Schema valid?}
    V -- no --> S
    V -- yes --> H[Human approval]
    H --> O[CompanyBrief JSON]
    P & S -.traces.-> L[Langfuse]
```

## Domain-tuned retrieval for SEC filings (done)

Retrieval quality decides whether the agent cites the right passage, and general-purpose embedding models
are trained on web text, not 10-K language. So the filings search uses an embedding model fine-tuned on
SEC filings, built and measured in [`embeddings/`](embeddings/):

1. **Dataset:** latest 10-K of 24 companies from SEC EDGAR, Items 1, 1A, 7 and 7A chunked into 1,422
   passages; an LLM wrote one analyst-style question per passage; split **by company** to avoid leakage.
2. **Training:** `BAAI/bge-small-en-v1.5` fine-tuned with MultipleNegativesRankingLoss (in-batch negatives)
   on a free Kaggle T4.
3. **Evaluation** on 355 questions from 6 companies never seen in training:

| Metric | Base bge-small | Fine-tuned | Change |
|---|---|---|---|
| Recall@1 (correct passage ranked first) | 0.730 | **0.797** | +0.068 |
| MRR@10 | 0.816 | **0.868** | +0.052 |
| NDCG@10 | 0.854 | **0.898** | +0.043 |
| Recall@10 | 0.972 | **0.989** | +0.017 |

4. **Demo:** the model is exported to int8 ONNX and runs client-side with Transformers.js
   ([`demo-web/`](demo-web/)); Recall@1 stays at 0.789 vs 0.730 for the int8 base model.

## Features

| Feature | Status |
|---|---|
| `CompanyBrief` output schema with per-claim source citations | done |
| Fine-tuned retrieval model for 10-K filings, with dataset, benchmark and live demo | done |
| LangGraph state graph: plan, research, synthesize, validate, approve | in progress |
| SEC EDGAR filings MCP server (fetch, section-aware chunking, hybrid search) | planned |
| Web search and financial data tools | planned |
| Langfuse tracing: cost, latency and tool calls per run | planned |
| Eval suite: task success, tool-call accuracy, citation faithfulness | planned |
| Prompt-injection test cases and PII redaction | planned |
| Response caching and cheap-model-first routing | planned |
| Docker image and Terraform for AWS | planned |

## Tech stack

Python 3.11, LangGraph, Model Context Protocol (MCP), Pydantic, PostgreSQL + pgvector, Langfuse, DeepEval,
Docker, Terraform, GitHub Actions. LLM-agnostic: runs on Gemini or Groq free tiers, or locally with Ollama.

## Project layout

```
agent/         LangGraph graph, nodes, tools and output schemas
mcp_server/    SEC EDGAR filings MCP server
evals/         golden company set and eval runner
embeddings/    10-K retrieval dataset builder, fine-tuning script, Kaggle notebook, results
demo-web/      in-browser search demo (Transformers.js) and its ONNX export / index scripts
demo-gradio/   the same demo as a Gradio app, for running locally
infra/         Dockerfile and Terraform
tests/         unit tests
```

## Getting started

```bash
uv sync --extra dev
cp .env.example .env    # add one LLM key and your SEC User-Agent
uv run pytest
```

## Evaluation

Results will be published here once the eval suite runs: task success rate, tool-call accuracy,
citation faithfulness, median latency and cost per brief.

## License

MIT
