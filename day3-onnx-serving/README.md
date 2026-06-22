# Day 3 — ONNX Export + FastAPI Serving + Latency Benchmark

Take a PyTorch model to production-style inference: export to ONNX, serve it over HTTP, and
measure latency properly.

## What it does

- Exports ResNet-18 to **ONNX** with `torch.onnx.export` — dynamic batch axis, named I/O, and
  `onnx.checker` validation.
- Runs it under **ONNX Runtime** and benchmarks against PyTorch with proper methodology
  (warm-up runs, `perf_counter`, mean + p95, latency *and* throughput).
- Serves predictions through a **FastAPI** endpoint (`server.py`) that accepts image uploads.
- Investigates **dynamic INT8 quantization** and explains the realistic finding: it barely helps
  CNNs on CPU because ORT's CPU provider only quantizes `MatMul`/`Gemm`, not `Conv`.

## How to run

```bash
pip install torch torchvision onnx onnxruntime fastapi uvicorn python-multipart pillow numpy matplotlib

# notebook (export + benchmark)
jupyter notebook day3_onnx_serving.ipynb

# serve (after the notebook has produced the .onnx file)
uvicorn server:app --reload
# then POST an image to the endpoint, or use the auto-generated docs at /docs
```

> The `.onnx` files are not committed (regenerate by running the notebook).

## Output

ONNX Runtime is **≈2× faster than PyTorch on CPU** (mean latency 97ms → 45ms for ResNet-18) —
because ORT is inference-only and fuses ops (Conv+BN+ReLU) at load time. Benchmark:

![Benchmark](benchmark.png)
