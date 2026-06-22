# Day 2 — ResNet-18 Inference + Feature-Map Visualization

Inference-only exploration of a pretrained ResNet-18: classify an image, then look *inside* the
network to see what its convolutional layers actually detect.

## What it does

- Loads ResNet-18 with ImageNet-pretrained weights and the correct preprocessing contract
  (`Resize(256) → CenterCrop(224)` + ImageNet normalization).
- Runs classification on a sample image and reads off top-5 predictions.
- Attaches a **forward hook** to an early conv layer to capture its feature maps, then visualizes
  the individual filter activations — edges, color gradients, textures.
- Explains the concepts: the input contract and why normalization mismatch is a silent bug,
  `model.eval()` vs `torch.no_grad()`, BatchNorm train/inference behavior, and why hooks beat
  surgically editing the model to grab intermediate outputs.

## How to run

```bash
pip install torch torchvision numpy matplotlib pillow
jupyter notebook day2_resnet_features.ipynb
```

A sample photo is included at `data/Amy.jpg`, so the notebook runs with no extra setup. The
pretrained weights download automatically on first use.

## Output

Top-5 ImageNet predictions for the input image, plus a grid of early-layer feature maps showing
which visual patterns each filter responds to. Visualizations are embedded inline in the notebook.
