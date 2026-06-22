# Day 1 — MLP from Scratch on MNIST

A multi-layer perceptron trained on MNIST handwritten digits, built to exercise the full PyTorch
training loop end-to-end — no high-level training wrappers.

## What it does

- Loads MNIST via `torchvision`, normalizes with the dataset's mean/std, and batches with a `DataLoader`.
- Defines an MLP (`Flatten → Linear → ReLU → Dropout → Linear`) as an `nn.Module`.
- Trains with the explicit 5-step loop: `zero_grad → forward → loss → backward → step`.
- Runs error analysis on misclassified digits and saves the trained weights (`state_dict`).
- Covers the *why* behind each piece: logits vs. softmax, `model.train()` vs `model.eval()`,
  why `zero_grad` comes first, and how the common optimizers (SGD, momentum, Adam, AdamW) differ.

## How to run

```bash
pip install torch torchvision numpy matplotlib
jupyter notebook day1_mlp_mnist.ipynb
```

MNIST auto-downloads to `data/` on first run.

## Output

**98.1% test accuracy** after 5 epochs on CPU (~5 minutes), loss dropping every epoch with no
overfitting. Training curves:

![Training curves](training_curves.png)
