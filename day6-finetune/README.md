# Day 6 — Transfer Learning & Fine-Tuning MobileNetV2

Adapt an ImageNet-pretrained MobileNetV2 to a binary cat-vs-dog task (Oxford-IIIT Pet), comparing
two transfer-learning strategies and measuring whether the more aggressive one actually helps.

## What it does

- Swaps MobileNetV2's 1000-class head for a fresh `Linear(1280 → 2)`.
- **Phase 1 — feature extraction:** freezes the backbone (both `requires_grad` *and* BatchNorm
  running-stat locks) and trains only the head.
- **Phase 2 — fine-tuning:** progressively unfreezes the last two blocks and trains them with a
  **differential learning rate** (tiny LR on the backbone, larger on the head), rebuilding the
  optimizer because its param list is a construction-time snapshot.
- Uses train/val/test correctly, checkpoints on **best val macro-F1**, and compares the two phases
  on the held-out test set with a full training-curve plot.
- Covers the supporting concepts: data augmentation (train-only), loss vs. F1, precision/recall,
  and Adam's cold-start instability at the phase boundary.

## How to run

```bash
pip install torch torchvision numpy matplotlib scikit-learn
jupyter notebook day6-finetune.ipynb
```

Oxford-IIIT Pet auto-downloads on first run.

> Checkpoints (`.pt`) are not committed (regenerate by running the notebook).

## Output — and the actual lesson

| | Phase 1 (head only) | Phase 2 (fine-tuned) | Δ |
|---|---|---|---|
| Test macro-F1 | **0.9917** | 0.9833 | **−0.0083** |

**Fine-tuning made it *worse*.** On a small dataset for a task that's squarely inside ImageNet's
distribution, feature extraction had already nearly solved it — and unfreezing ~40% of the params
on only 800 images dragged the pretrained weights *away* from their good state. The takeaway isn't
the number; it's the judgment: *more training capacity is not more performance*, and you should
measure the delta rather than assume fine-tuning helps. The training curves (inline in the
notebook) also show Adam's characteristic loss spike at the Phase 1→2 boundary.
