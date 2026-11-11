"""Precompute passage vectors for the in-browser demo.

Uses the SAME int8 ONNX models the browser runs, and re-runs the held-out benchmark with them.
Outputs in demo-web/site/data/: passages.json, emb_tuned.bin and emb_base.bin
(float32, row-major), meta.json.

  python demo-web/build_index.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

HERE = Path(__file__).parent
SITE_DATA = HERE / "site" / "data"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
LOCAL_TUNED = HERE / "onnx_export" / "onnx"
TOK_TUNED = HERE.parent / "embeddings/output/bge-small-sec-filings"


class OnnxEmbedder:
    def __init__(self, onnx_path: str, tokenizer_src: str | Path):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(tokenizer_src)
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def encode(self, texts: list[str], batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok(
                texts[i : i + batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            feed = {k: v.astype(np.int64) for k, v in enc.items() if k in self.inputs}
            hidden = self.sess.run(None, feed)[0]
            cls = hidden[:, 0]  # bge uses CLS pooling
            out.append(cls / np.linalg.norm(cls, axis=1, keepdims=True))
        return np.concatenate(out).astype(np.float32)


def benchmark(model: OnnxEmbedder, test) -> dict:
    p = model.encode([r["passage"] for r in test])
    q = model.encode([QUERY_PREFIX + r["question"] for r in test])
    ranks = np.argsort(-(q @ p.T), axis=1)
    pos = np.array([int(np.where(ranks[i] == i)[0][0]) + 1 for i in range(len(test))])
    r10 = pos <= 10
    return {
        "recall@1": round(float(np.mean(pos == 1)), 4),
        "recall@10": round(float(np.mean(r10)), 4),
        "mrr@10": round(float(np.mean(np.where(r10, 1 / pos, 0))), 4),
        "ndcg@10": round(float(np.mean(np.where(r10, 1 / np.log2(pos + 1), 0))), 4),
    }


def main() -> None:
    ds = load_dataset("jc4845/sec-10k-retrieval-qa")
    models = {
        "tuned": OnnxEmbedder(str(LOCAL_TUNED / "model_quantized.onnx"), TOK_TUNED),
        "base": OnnxEmbedder(
            hf_hub_download("Xenova/bge-small-en-v1.5", "onnx/model_quantized.onnx"),
            "Xenova/bge-small-en-v1.5",
        ),
    }

    rows = [r for split in ("train", "test") for r in ds[split]]
    rows.sort(key=lambda r: (r["ticker"], r["id"]))
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    passages = [{k: r[k] for k in ("ticker", "section", "source_url", "passage")} for r in rows]
    (SITE_DATA / "passages.json").write_text(
        json.dumps(passages, ensure_ascii=False), encoding="utf-8"
    )

    bench = {}
    for name, m in models.items():
        m.encode([r["passage"] for r in rows]).tofile(SITE_DATA / f"emb_{name}.bin")
        bench[name] = benchmark(m, ds["test"])
        print(name, "int8 benchmark:", bench[name], flush=True)

    meta = {
        "n": len(rows),
        "dim": 384,
        "held_out": sorted(set(ds["test"]["ticker"])),
        "int8_benchmark": bench,
    }
    (SITE_DATA / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
