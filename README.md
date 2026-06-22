# ML / Deep Learning Engineering Portfolio

Six self-contained PyTorch projects spanning the full applied-ML lifecycle: **training a model
from scratch, interpreting what it learned, optimizing it for production inference, deploying it
to the edge, using it for embeddings/clustering, and fine-tuning it on a new task.** Each project
is a single, heavily-annotated Jupyter notebook — built to be read, not just run.

> Every notebook explains the *why* before the code and prints tensor shapes at each step.
> Concept-level write-ups for all six projects live in **[LEARN.md](LEARN.md)** — the engineering
> deep-dives behind these projects.

---

## ⭐ Featured — Photo Clustering Pipeline (CLIP + k-means)

[**→ day5-photo-clustering/**](day5-photo-clustering/)

The automatic album-generation problem: given a pile of photos, group them into semantically
meaningful albums (beaches, food, people, pets) **with no labels**. The pipeline runs each image
through **CLIP** to get 512-dim embeddings, L2-normalizes them, clusters with **k-means**, and
visualizes the result as per-cluster image grids and a **t-SNE** projection.

The interesting part is what it taught about *evaluation*: in 512-D space the elbow method is
blinded by the curse of dimensionality (PCA-to-50D is needed to even see an elbow), and **dataset
diversity matters as much as the algorithm** — a single-domain dataset (all food) produces no
cluster gaps because the gaps don't exist in the data. The notebook also covers how this scales
to millions of users (per-user clustering, FAISS, MiniBatchKMeans, distributed task queues).

---

## Projects

| # | Project | What it demonstrates |
|---|---------|----------------------|
| 1 | [day1-mlp-mnist](day1-mlp-mnist/) | An MLP built from scratch on MNIST — the full training loop (`zero_grad → forward → loss → backward → step`), Dropout, optimizers, error analysis. **98.1% test accuracy.** |
| 2 | [day2-resnet-features](day2-resnet-features/) | ResNet-18 inference and **feature-map visualization** via forward hooks — seeing what early conv layers actually detect (edges, textures, color). |
| 3 | [day3-onnx-serving](day3-onnx-serving/) | Export to **ONNX**, serve via **FastAPI**, and benchmark latency. ONNX Runtime ≈ **2× faster** than PyTorch on CPU (97ms → 45ms). Includes a realistic look at why dynamic INT8 quantization barely helps CNNs on CPU. |
| 4 | [day4-coreml](day4-coreml/) | Convert PyTorch → **CoreML** (`.mlpackage`) via TorchScript, with **parity validation** (max-diff + `np.allclose`) to prove the converted model matches the original. |
| 5 | [day5-photo-clustering](day5-photo-clustering/) | ⭐ The featured CLIP + k-means album-generation pipeline (above). |
| 6 | [day6-finetune](day6-finetune/) | Transfer learning on MobileNetV2 — feature extraction vs. progressive unfreezing with **differential learning rates**. Key finding: **fine-tuning *regressed* vs. head-only** on this small near-ImageNet task — and knowing *why* is the lesson. |

---

## Roadmap — the arc these projects cover

The projects are ordered as a deliberate progression through applied ML engineering:

1. **Train** — build and train a network from scratch, understand the loop end-to-end *(day1)*
2. **Interpret** — load a pretrained model, inspect intermediate activations *(day2)*
3. **Optimize for inference** — export, serve, and benchmark for production *(day3)*
4. **Deploy to the edge** — convert to a device-native format and validate parity *(day4)*
5. **Embeddings & unsupervised ML** — represent images as vectors and cluster them at scale *(day5)*
6. **Adapt** — fine-tune a pretrained model on a new task, and measure whether it actually helped *(day6)*

---

## Tech stack

`PyTorch` · `torchvision` · `ONNX` / `ONNX Runtime` · `coremltools` · `OpenAI CLIP` ·
`scikit-learn` · `FastAPI` · `matplotlib` · `numpy`

---

## Repo structure

```
.
├── day1-mlp-mnist/         # MLP from scratch, MNIST, training loop
├── day2-resnet-features/   # ResNet-18 inference + feature-map viz
├── day3-onnx-serving/      # ONNX export + FastAPI + latency benchmark
├── day4-coreml/            # CoreML conversion + parity validation
├── day5-photo-clustering/  # CLIP embeddings + k-means album generation  ⭐
├── day6-finetune/          # MobileNet fine-tuning, augmentation, F1
├── LEARN.md                # Engineering notes & concept deep-dives
└── README.md
```

---

## Getting started

```bash
# 1. create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install the core stack
pip install torch torchvision numpy matplotlib scikit-learn

# project-specific extras:
#   day3:  pip install onnx onnxruntime fastapi uvicorn python-multipart pillow
#   day4:  pip install coremltools onnx
#   day5:  pip install git+https://github.com/openai/CLIP.git ftfy regex

# 3. open any project notebook
jupyter notebook
```

Each project's own README has run instructions and expected output.

---

## A note on data & models

To keep the repo lean, **datasets and trained model artifacts are not committed** — they
regenerate when you run the notebooks:

- **Datasets** (MNIST, Oxford-IIIT Pet, CLIP demo images) **auto-download** on first run.
  The one exception is `day2-resnet-features/data/Amy.jpg`, a sample photo included so day 2 runs
  out of the box.
- **Model artifacts** (`.pt`, `.pth`, `.onnx`, `.mlpackage`) are produced by running the notebooks.
  Each notebook keeps its result cells (metrics, plots, visualizations) inline as proof of output.
