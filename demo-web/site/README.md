---
title: SEC Filing Search
emoji: 📑
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Fine-tuned vs base embeddings on 10-K filings, in-browser
models:
  - jc4845/bge-small-sec-filings
  - Xenova/bge-small-en-v1.5
datasets:
  - jc4845/sec-10k-retrieval-qa
---

# SEC Filing Search

Ask a question about a US public company and compare the top 10-K passages returned by
[jc4845/bge-small-sec-filings](https://huggingface.co/jc4845/bge-small-sec-filings) (fine-tuned) and
bge-small-en-v1.5 (base), side by side.

Both models run **in the browser** with Transformers.js (int8 ONNX). Passage vectors for 1,422 passages are
precomputed with the same int8 models; only the question is encoded live.

On 355 held-out questions from 6 companies never seen in training, the fine-tuned model ranks the correct
passage first 79.7% of the time vs 73.0% for the base model (78.9% vs 73.0% with the int8 models used here).

Part of [dealscope-agent](https://github.com/Jaydeep4845/dealscope-agent).
