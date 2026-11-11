# In-browser demo

Static page that compares the fine-tuned and base embedding models side by side, running both in the browser
with [Transformers.js](https://huggingface.co/docs/transformers.js). Live at
[huggingface.co/spaces/jc4845/sec-filing-search](https://huggingface.co/spaces/jc4845/sec-filing-search).

| File | Purpose |
|---|---|
| `export_onnx.py` | Export the fine-tuned model to ONNX and int8 (`onnx/model_quantized.onnx`, 34 MB) |
| `build_index.py` | Encode all 1,422 passages with the same int8 models the browser uses, and re-run the benchmark |
| `site/index.html` | The page: loads both models, encodes only the question, ranks precomputed passage vectors |
| `site/README.md` | Hugging Face Space configuration (`sdk: static`) |

Rebuild and preview:

```bash
python demo-web/export_onnx.py       # then upload onnx/model_quantized.onnx to the model repo
python demo-web/build_index.py       # writes site/data/ (git-ignored)
cd demo-web/site && python -m http.server 8765
```
