# Day 4: CoreML conversion and parity validation

Converts a PyTorch model to Apple's on-device inference format and checks that the conversion is
correct.

## What it does

- Traces ResNet-18 with `torch.jit.trace` (in `eval()` mode, wrapped in `no_grad`) to produce a
  TorchScript graph.
- Converts it to CoreML (`.mlpackage`) with `coremltools`, baking ImageNet normalization into the
  model through an `ImageType` input (`scale` and `bias`).
- Validates parity by running the same input through both models and comparing logits with both
  max-diff and `np.allclose` (absolute and relative tolerance), which answer different questions.
- Includes notes on the production realities: `trace` versus `script`, why CoreML stores weights in
  float16 by default (about half the ONNX size), and that `coremltools` can convert on Linux but only
  runs inference on macOS, so ONNX is used as a parity proxy on non-Apple hardware.

## Requirements

`coremltools` runs on macOS and Linux, not on native Windows. On Windows, run this project under
WSL (a Linux environment). Conversion works on Linux; running the converted `.mlpackage` for
inference requires macOS, which is why ONNX is used as a parity proxy on non-Apple hardware.

## How to run

```bash
pip install torch torchvision coremltools onnx numpy
jupyter notebook day4_coreml.ipynb
```

The `.mlpackage` and `.onnx` artifacts are not committed; running the notebook regenerates them.

## Output

A `resnet18.mlpackage` whose logits match the source PyTorch model within numerical tolerance (the
parity check passes), at roughly half the on-disk size thanks to float16 weights. The parity report
is printed inline in the notebook.
