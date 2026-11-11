"""Export the fine-tuned embedding model to ONNX (fp32 + int8) for Transformers.js.

Transformers.js loads `onnx/model_quantized.onnx` from a model repo when asked for dtype "q8",
so we write that layout and later upload the onnx/ folder to jc4845/bge-small-sec-filings.

  python demo-web/export_onnx.py
"""

from __future__ import annotations

from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModel, AutoTokenizer

SRC = Path(__file__).parents[1] / "embeddings/output/bge-small-sec-filings"
OUT = Path(__file__).parent / "onnx_export" / "onnx"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(SRC)
    model = AutoModel.from_pretrained(SRC).eval()
    model.config.return_dict = True

    enc = tok(["a sample sentence for tracing"], return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    dyn = {n: {0: "batch_size", 1: "sequence_length"} for n in names}
    dyn["last_hidden_state"] = {0: "batch_size", 1: "sequence_length"}

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            ).last_hidden_state

    fp32 = OUT / "model.onnx"
    torch.onnx.export(
        Wrapper(model),
        tuple(enc[n] for n in names),
        str(fp32),
        input_names=names,
        output_names=["last_hidden_state"],
        dynamic_axes=dyn,
        opset_version=14,
        dynamo=False,
    )
    quantize_dynamic(str(fp32), str(OUT / "model_quantized.onnx"), weight_type=QuantType.QUInt8)
    for f in OUT.iterdir():
        print(f.name, round(f.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
