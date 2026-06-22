# Applied Deep Learning

A collection of six self-contained PyTorch projects that work through the applied deep-learning
lifecycle from end to end: training a model from scratch, interpreting a pretrained one, optimizing
it for inference, deploying it to the edge, clustering images with embeddings, and fine-tuning.

The aim is to cover the whole path from a trained model to a deployed one, giving the production
side (export, serving, latency, on-device conversion, evaluation) the same weight as the modeling.
Each project takes one concrete task all the way through and makes the engineering decisions
explicit: preprocessing and input contracts, serving and latency, conversion parity, and how
results are measured, including cases where a technique did not help. Every project is a single
Jupyter notebook that runs on CPU and records the reasoning behind each step.

Longer conceptual notes for all six projects are collected in [LEARN.md](LEARN.md).

## Featured project: Photo clustering with CLIP and k-means

[photo-clustering](photo-clustering/)

Groups an unlabeled set of photos into semantically coherent albums such as beaches, food, people,
and pets. Each image is encoded with CLIP (ViT-B/32) into a 512-dimensional embedding. The
embeddings are L2-normalized so k-means clusters by cosine similarity, the cluster count k is
chosen with the elbow method (using a PCA projection to 50 dimensions to get a usable signal in
high-dimensional space), and the result is rendered as a per-cluster image grid and a 2D t-SNE map.
The notebook also covers running this at scale: per-user clustering, FAISS, MiniBatchKMeans, and a
distributed task queue.

## Projects

| # | Project | Description |
|---|---------|-------------|
| 1 | [mlp-mnist](mlp-mnist/) | An MLP built from scratch on MNIST: the full training loop, dropout, optimizers, and error analysis. Reaches 98.1% test accuracy. |
| 2 | [resnet-features](resnet-features/) | ResNet-18 inference plus feature-map visualization through forward hooks, showing what the early convolutional layers respond to. |
| 3 | [onnx-serving](onnx-serving/) | Export to ONNX, serve over FastAPI, and benchmark latency. ONNX Runtime runs about 2x faster than PyTorch on CPU (PyTorch 97 ms, ONNX Runtime 45 ms). Includes why dynamic INT8 quantization gives little benefit for CNNs on CPU. |
| 4 | [coreml](coreml/) | Convert a PyTorch model to CoreML (`.mlpackage`) through TorchScript, with parity validation (max-diff and `np.allclose`) confirming the converted model matches the original. |
| 5 | [photo-clustering](photo-clustering/) | CLIP embeddings plus k-means to group photos into albums with no labels. See the featured section above. |
| 6 | [finetune](finetune/) | Transfer learning on MobileNetV2: feature extraction versus progressive unfreezing with differential learning rates. On this small, near-ImageNet task, feature extraction scored higher than fine-tuning (test macro-F1 0.9917 vs 0.9833). |

## What's covered

A guide to the topics across the six projects:

- Training: building and training a network from scratch (mlp-mnist)
- Model interpretation: inspecting the intermediate activations of a pretrained model (resnet-features)
- Inference optimization: export, serving, and latency benchmarking (onnx-serving)
- Edge deployment: converting to a device-native format with parity checks (coreml)
- Embeddings and clustering: representing images as vectors and grouping them, with notes on scaling (photo-clustering)
- Fine-tuning: adapting a pretrained model and comparing it against feature extraction (finetune)

## Tech stack

`PyTorch` · `torchvision` · `ONNX` / `ONNX Runtime` · `coremltools` · `OpenAI CLIP` ·
`scikit-learn` · `FastAPI` · `matplotlib` · `numpy`

## Repo structure

```
.
├── mlp-mnist/          # MLP from scratch, MNIST, training loop
├── resnet-features/    # ResNet-18 inference + feature-map viz
├── onnx-serving/       # ONNX export + FastAPI + latency benchmark
├── coreml/             # CoreML conversion + parity validation
├── photo-clustering/   # CLIP embeddings + k-means album generation
├── finetune/           # MobileNet fine-tuning, augmentation, F1
├── LEARN.md            # Concept notes for each project
└── README.md
```

## Getting started

Each project is independent and has its own `requirements.txt`.

```bash
# 1. create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install one project's dependencies and open it
cd photo-clustering
pip install -r requirements.txt
jupyter notebook
```

Each project folder has its own README with run instructions and output. Note that the CoreML project
needs a Unix-based environment; see its README.

## Data and models

To keep the repo small, datasets and trained model artifacts are not committed. They regenerate
when you run the notebooks.

- Datasets (MNIST, Oxford-IIIT Pet, CLIP demo images) download automatically on first run. The
  exception is `resnet-features/data/Amy.jpg`, a sample photo committed so the ResNet project
  runs without setup.
- Model artifacts (`.pt`, `.pth`, `.onnx`, `.mlpackage`) are produced by running the notebooks.
  Each notebook keeps its result cells (metrics, plots, visualizations) inline.
