# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A hands-on ML/DL engineering portfolio: six self-contained projects spanning training, inference optimization, edge deployment, embeddings, and fine-tuning. All projects use **Python + PyTorch**. Work is done in **Jupyter notebooks** for step-by-step, explainable exploration.

## Project layout

```
day1-mlp-mnist/        # MLP from scratch, MNIST, training loop
day2-resnet-features/  # ResNet-18 inference + feature map viz
day3-onnx-serving/     # ONNX export + FastAPI + latency benchmark
day4-coreml/           # CoreML conversion + parity validation
day5-photo-clustering/ # CLIP embeddings + k-means album generation
day6-finetune/         # MobileNet fine-tuning, augmentation, F1
```

## Environment

- Python with PyTorch, torchvision, matplotlib, numpy
- Jupyter notebooks (`.ipynb`) are the primary format — one notebook per day
- Run notebooks with: `jupyter notebook` or open in VS Code with the Jupyter extension

## Key teaching conventions

- Every notebook cell has a purpose — explain WHAT and WHY before code
- Print shapes at every tensor operation: `print(tensor.shape)`
- Prefer explicit over concise: no one-liners that hide what's happening
- After each major block, add a markdown cell summarizing what just happened
