"""Precompute passage embeddings for the demo Space with both models.

Encoding 1.4K passages twice takes a few minutes on CPU, so we do it once here and ship the
vectors with the app. Startup then only loads the two models (to encode queries).

  uv run --extra embeddings python demo-gradio/build_index.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
DATASET = "jc4845/sec-10k-retrieval-qa"
MODELS = {"base": "BAAI/bge-small-en-v1.5", "tuned": "jc4845/bge-small-sec-filings"}


def main() -> None:
    ds = load_dataset(DATASET)
    test_tickers = sorted(set(ds["test"]["ticker"]))
    rows = []
    for split in ("train", "test"):
        for r in ds[split]:
            rows.append({k: r[k] for k in ("id", "ticker", "section", "source_url", "passage")})
    rows.sort(key=lambda r: (r["ticker"], r["id"]))

    data = HERE / "data"
    data.mkdir(exist_ok=True)
    with (data / "passages.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (data / "meta.json").write_text(json.dumps({"held_out": test_tickers}, indent=2))

    texts = [r["passage"] for r in rows]
    for name, model_id in MODELS.items():
        model = SentenceTransformer(model_id)
        model.max_seq_length = 384
        vecs = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
        np.save(data / f"emb_{name}.npy", vecs.astype(np.float16))
        print(name, vecs.shape)


if __name__ == "__main__":
    main()
