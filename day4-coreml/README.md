# Day 4 — CoreML Conversion + Parity Validation

Convert a PyTorch model to Apple's on-device inference format and prove the conversion is correct.

## What it does

- Traces ResNet-18 with `torch.jit.trace` (wrapped in `no_grad`, in `eval()` mode) to produce a
  TorchScript graph.
- Converts it to **CoreML** (`.mlpackage`) with `coremltools`, baking ImageNet normalization into
  the model via an `ImageType` input (`scale`/`bias`).
- **Validates parity**: runs the same input through both models and compares logits with *both*
  max-diff and `np.allclose` (atol + rtol) — each answers a different question.
- Explains the production realities: `trace` vs `script`, why CoreML stores weights in float16 by
  default (≈half the ONNX size), and that `coremltools` can *convert* on Linux but only *runs*
  inference on macOS (so ONNX is used as a parity proxy on non-Apple hardware).

## How to run

```bash
pip install torch torchvision coremltools onnx numpy
jupyter notebook day4_coreml.ipynb
```

> The `.mlpackage` / `.onnx` artifacts are not committed (regenerate by running the notebook).

## Output

A `resnet18.mlpackage` whose logits match the source PyTorch model within numerical tolerance
(parity check passes), at roughly half the on-disk size thanks to float16 weights. Conversion
details and the parity report are inline in the notebook.
