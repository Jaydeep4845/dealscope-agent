"""Fine-tune a small embedding model for 10-K retrieval and measure the gain.

Runs on a free Kaggle/Colab T4 GPU in roughly 10-20 minutes (CPU works but is slow).

  1. Evaluate the base model on the held-out test companies.
  2. Fine-tune with MultipleNegativesRankingLoss (in-batch negatives) on train pairs.
  3. Evaluate again on the same test set and write results.json.
  4. Optionally push the model and a generated model card to the Hugging Face Hub.

Usage:
  python embeddings/finetune.py                       # train + evaluate
  python embeddings/finetune.py --push                # also upload to the Hub (needs HF_TOKEN)
  python embeddings/finetune.py --dataset jc4845/sec-10k-retrieval-qa   # read data from the Hub
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

HERE = Path(__file__).parent
BASE_MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this instruction on queries (not on passages)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
METRICS = ["ndcg@10", "mrr@10", "recall@1", "recall@10"]


def load(dataset: str | None):
    if dataset:
        return load_dataset(dataset)
    files = {"train": str(HERE / "data/train.jsonl"), "test": str(HERE / "data/test.jsonl")}
    return load_dataset("json", data_files=files)


def build_evaluator(test) -> InformationRetrievalEvaluator:
    # Corpus = every test passage, so each query competes against all other passages
    # from the same held-out companies (hard, same-domain distractors).
    corpus = {r["id"]: r["passage"] for r in test}
    queries = {r["id"]: QUERY_PREFIX + r["question"] for r in test}
    relevant = {r["id"]: {r["id"]} for r in test}
    return InformationRetrievalEvaluator(
        queries,
        corpus,
        relevant,
        name="sec10k",
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 10],
        precision_recall_at_k=[1, 10],
        show_progress_bar=False,
    )


def pick(scores: dict) -> dict:
    out = {}
    for m in METRICS:
        key = next(k for k in scores if k.endswith(f"_{m}") and "cosine" in k)
        out[m] = round(scores[key], 4)
    return out


def model_card(
    repo: str, dataset: str, before: dict, after: dict, n_train: int, n_test: int
) -> str:
    rows = "\n".join(
        f"| {m} | {before[m]:.3f} | {after[m]:.3f} | {after[m] - before[m]:+.3f} |" for m in METRICS
    )
    return f"""---
license: mit
base_model: {BASE_MODEL}
datasets: [{dataset}]
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags: [sentence-transformers, retrieval, finance, sec-filings, embeddings]
---

# {repo.split("/")[-1]}

`{BASE_MODEL}` fine-tuned to retrieve passages from SEC 10-K filings (Business, Risk Factors,
MD&A, Market Risk) for analyst-style questions. Built as the retrieval model for
[dealscope-agent](https://github.com/Jaydeep4845/dealscope-agent).

## Results

Held-out evaluation on **{n_test} questions from companies never seen in training**. Each query
is ranked against every passage from those companies.

| Metric | Base model | Fine-tuned | Change |
|---|---|---|---|
{rows}

## Training

- Data: [{dataset}](https://huggingface.co/datasets/{dataset}), {n_train} question-passage pairs.
  Questions are LLM-generated from the passage, then split by company to avoid leakage.
- Loss: MultipleNegativesRankingLoss (in-batch negatives), no-duplicates batch sampler.
- Queries use the bge instruction prefix: `{QUERY_PREFIX}`

## Usage

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("{repo}")
q = model.encode("{QUERY_PREFIX}What supply chain risks does Apple disclose?")
```

## Limitations

- Questions are synthetic (LLM-generated), so they are cleaner than real analyst queries.
- Only US 10-K prose; tables, 10-Qs and non-US filings are not covered.
- Small evaluation set; treat differences under about 0.02 as noise.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="Hub dataset id; defaults to local data/*.jsonl")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-seq-length", type=int, default=384)
    ap.add_argument("--push", action="store_true")
    ap.add_argument(
        "--repo", default=os.environ.get("HF_MODEL_REPO", "jc4845/bge-small-sec-filings")
    )
    args = ap.parse_args()

    ds = load(args.dataset)
    train, test = ds["train"], ds["test"]
    evaluator = build_evaluator(test)

    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = args.max_seq_length  # passages are ~220 words, about 300 tokens
    before = pick(evaluator(model))
    print("base:", before)

    pairs = train.map(
        lambda r: {"anchor": QUERY_PREFIX + r["question"], "positive": r["passage"]},
        remove_columns=train.column_names,
    )
    out_dir = HERE / "output" / "bge-small-sec-filings"
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True if __import__("torch").cuda.is_available() else False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # duplicate passages would be false negatives
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        seed=42,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=pairs,
        loss=MultipleNegativesRankingLoss(model),
        evaluator=evaluator,
    )
    trainer.train()

    after = pick(evaluator(model))
    print("fine-tuned:", after)
    results = {
        "base_model": BASE_MODEL,
        "n_train": len(train),
        "n_test": len(test),
        "base": before,
        "fine_tuned": after,
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    model.save(str(out_dir))

    if args.push:
        dataset_id = args.dataset or os.environ.get(
            "HF_DATASET_REPO", "jc4845/sec-10k-retrieval-qa"
        )
        card = model_card(args.repo, dataset_id, before, after, len(train), len(test))
        (out_dir / "README.md").write_text(card, encoding="utf-8")
        model.push_to_hub(args.repo, token=os.environ.get("HF_TOKEN"), exist_ok=True)
        from huggingface_hub import upload_file

        upload_file(
            path_or_fileobj=str(out_dir / "README.md"),
            path_in_repo="README.md",
            repo_id=args.repo,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"pushed to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
