# Day 5 — CLIP Embeddings + k-means Photo Clustering ⭐

The automatic album-generation problem: group a pile of photos into semantically meaningful
albums — beaches, food, people, pets — **with no labels**.

## What it does

- Embeds each image with **CLIP** (`ViT-B/32`) into a 512-dim semantic vector.
- L2-normalizes the embeddings and clusters them with **k-means** (cosine-equivalent after
  normalization).
- Chooses `k` with the **elbow method** — and shows why it fails directly in 512-D (curse of
  dimensionality), requiring a **PCA-to-50D** step just to make an elbow visible.
- Visualizes results as **per-cluster image grids** and a **t-SNE** 2-D projection (and explains
  why t-SNE separates clusters where PCA-to-2D can't).
- Discusses production scale-up: per-user clustering, **FAISS**, `MiniBatchKMeans`, vector DBs,
  multi-signal clustering (CLIP + timestamp + GPS + faces), and distributed task queues.

## How to run

```bash
pip install torch torchvision numpy matplotlib scikit-learn ftfy regex
pip install git+https://github.com/openai/CLIP.git
jupyter notebook day5-clip-clustering.ipynb
```

Demo images download automatically on first run.

## Output

Clean, semantically-coherent clusters when the photo set spans multiple domains — and the key
engineering lesson: **dataset diversity matters as much as the algorithm.** A single-domain set
(all food) yields no cluster gaps because the gaps don't exist in the data. Cluster grids and the
t-SNE plot are embedded inline in the notebook.
