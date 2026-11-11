---
title: SEC Filing Search
emoji: 📑
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
short_description: Search 10-K filings with a fine-tuned embedding model
models:
  - jc4845/bge-small-sec-filings
  - BAAI/bge-small-en-v1.5
datasets:
  - jc4845/sec-10k-retrieval-qa
---

# SEC Filing Search

Ask a question about a US public company and compare the top 10-K passages returned by
[jc4845/bge-small-sec-filings](https://huggingface.co/jc4845/bge-small-sec-filings) (fine-tuned) and
`BAAI/bge-small-en-v1.5` (base), side by side.

On 355 held-out questions from 6 companies never seen in training, the fine-tuned model ranks the correct
passage first 79.7% of the time vs 73.0% for the base model (NDCG@10 0.898 vs 0.854).

Hosted demo: the in-browser version at
[huggingface.co/spaces/jc4845/sec-filing-search](https://huggingface.co/spaces/jc4845/sec-filing-search)
(Gradio Spaces need a paid plan). Run this version locally with `python app.py` after `build_index.py`.

Part of [dealscope-agent](https://github.com/Jaydeep4845/dealscope-agent). Passage vectors are precomputed
by `build_index.py`; the app encodes only the query at search time.
