# Engineering Notes & Concept Deep-Dives

Concept-level notes accumulated while building the six projects in this portfolio — the *why* behind the code, not the code itself. Organized per project. Intended both as a learning record and as full context for an LLM working in this repo.

---

## Day 1 — MLP on MNIST

### Environment & Tooling
- **PyTorch** replaces NumPy for ML work — same n-dimensional array math, but tensors can live on GPU and track gradients (needed for backprop). NumPy is still imported for occasional matplotlib compatibility.
- **torch.nn** — where layer types live (Linear, ReLU, Dropout, etc.)
- **torch.optim** — where optimizers live (Adam, SGD). Gradient descent is not a separate library — it's inside the optimizer.
- **torchvision** — companion library for image datasets. MNIST is built in.
- **pandas** is for tabular/structured data (CSVs, spreadsheets). Not used in vision ML.
- **sklearn** is for classical ML and evaluation metrics. Not used for training neural nets.

### DataLoader
- Wraps a dataset and feeds data in **batches** during training (e.g. 64 images at a time).
- Prevents OOM — you can't fit the full dataset in memory at once, especially on GPU.
- **Shuffles** data every epoch — randomizes order so the model doesn't learn the sequence. This is NOT the same as k-fold cross validation.
- K-fold = splitting data into k chunks and training k separate models for more reliable accuracy estimates. Not used here — MNIST has a fixed pre-made train/test split.
- Mini-batch training (SGD) actually generalizes *better* than full-dataset gradient — the noise helps escape local minima.

### Hyperparameters (Cell 2)
- **Hyperparameters** = settings you choose before training. The model never learns them.
- **BATCH_SIZE** — images per weight update. 60k images / 64 batch = 937 updates per epoch.
- **LEARNING_RATE** — most sensitive knob. Too high → loss explodes/oscillates. Too low → barely moves.
- **EPOCHS** — full passes over the dataset. 1 epoch ≠ 1 weight update. 1 epoch = N/batch_size updates.
- **HIDDEN_SIZE** — width of hidden layers (neurons per layer). Width ≠ depth. They're independent.
- Depth (number of layers) is hardcoded in the architecture, not a standalone hyperparameter here. Could be made one.
- `device` — CPU or GPU. Set once, used everywhere to move tensors to the right hardware.

### Data Loading (Cell 3)
- **Transform pipeline** — PIL images (int 0–255) → float tensors (0.0–1.0) via `ToTensor()`, then normalized to mean≈0, std≈1 via `Normalize()`.
- Normalization values (0.1307, 0.3081) are MNIST's pre-computed dataset-wide mean and std. Using them keeps gradients stable — large raw pixel values cause unstable gradient magnitudes.
- For custom datasets: compute mean/std over training set only — never look at val/test during preprocessing (data leakage).
- After `ToTensor()`, image shape = `(1, 28, 28)` — the leading `1` is the channel dimension (grayscale). RGB would be `3`.
- PyTorch tensor convention is NCHW: `(batch, channels, height, width)`.
- `squeeze()` removes any dimension of size exactly 1. `permute(1,2,0)` reorders axes — used to go from `(3,28,28)` → `(28,28,3)` for matplotlib with RGB.

### Model Definition (Cell 5)
- Every PyTorch model subclasses `nn.Module`. Define layers in `__init__`, define data flow in `forward(x)`.
- `super().__init__()` — must call parent constructor. Sets up PyTorch's internal bookkeeping (parameter tracking, `.to(device)`, `.state_dict()`). Without it everything silently breaks.
- `nn.Sequential` — chains layers in order. Use when data flows straight through. Breaks down when you need skip connections or branching.
- `nn.Flatten()` — reshapes `(batch, 1, 28, 28)` → `(batch, 784)`. Not a learned layer — just a reshape. No weights.
- `nn.Linear(in, out)` — fully connected layer. Computes `y = Wx + b`. W and b are the learned parameters.
- `nn.ReLU()` — activation function: `max(0, x)`. Without activations, stacking Linear layers collapses to one Linear layer. ReLU adds non-linearity.
- `nn.Dropout(0.2)` — zeros 20% of neuron outputs randomly during training. Prevents overfitting. Automatically disabled during `model.eval()`.
- Output = **logits** (raw scores, not probabilities). `CrossEntropyLoss` applies softmax internally — never add softmax to the model or you double-apply it.
- Use `argmax` on logits for prediction — softmax doesn't change the order, so `argmax(logits) == argmax(softmax(logits))`.
- `model.to(device)` — moves all weights to CPU or GPU. Data tensors also need `.to(device)` inside the loop.
- `p.numel()` — number of elements in a tensor. Used to count total model parameters.

### Training Loop (Cell 7)
- The 5-step cycle per batch: `zero_grad → forward → loss → backward → step`
- `zero_grad` must come first — PyTorch accumulates gradients by default. Old gradients are stale (weights already moved) and corrupt the current update.
- `model.train()` is a misleading name — it does NOT train the model. It's just a mode switch that enables Dropout.
- `loss.item()` — pulls the scalar value out of the tensor. Without it you'd accumulate tensor objects, not floats.
- `outputs.argmax(dim=1)` — index of highest logit per sample = predicted class. `dim=1` = across the 10 class scores, not across the batch.
- Model weights update after every single batch — 937 updates per epoch. Each batch sees a slightly improved model.
- **Optimizers:**
  - SGD = fixed LR gradient step, zigzags toward minima
  - SGD+momentum = adds velocity term, smoother path, resists sudden direction changes
  - Adam = per-parameter adaptive LR + momentum. Good default, works out of the box
  - AdamW = Adam with correct weight decay separation. Preferred when regularization matters, standard for transformers
  - SGD+momentum sometimes beats Adam on vision tasks trained from scratch. AdamW wins on transformers.

### Error Analysis (Cell 10)
- `outputs.argmax(dim=1)` — shape `(64,)`. One predicted class per image.
- `mask = (predicted != labels)` — boolean tensor, True where model was wrong.
- `images[mask]` — boolean indexing, pulls only wrongly-predicted images from the batch.
- `.cpu()` on tensors before matplotlib — matplotlib can't read GPU memory. No-op on CPU but essential for GPU compatibility.
- `torch.no_grad()` — context manager, sets a flag: don't build computation graph. Saves memory and time during inference.
- `zero_grad()` vs `no_grad()`: zero_grad = erase gradients already computed. no_grad = don't compute gradients at all.
- Skipping `no_grad()` doesn't break correctness — just wastes memory. Always use it for inference.

### Model Saving & Portability (Cell 11)
- `state_dict()` — dictionary of all learned weights/biases. Just numbers, no architecture. PyTorch-only format.
- To reload: recreate the architecture (empty skeleton) first, then `load_state_dict()` slots weights in. Wrong architecture → RuntimeError with size mismatch. Fails loudly, never silently.
- `map_location=device` — handles loading a GPU-saved model onto CPU (or vice versa).
- **ONNX** — exports computation graph + weights together. Runtime-agnostic (C++, Java, mobile, browser). One-way — can't load back into PyTorch as a trainable model.
- `state_dict` = checkpointing during research. ONNX = deploying to production across runtimes.
- TensorFlow cannot load a PyTorch state_dict. ONNX is the bridge between frameworks.

### Training Results
- 5 epochs, CPU only, ~5 minutes on a laptop.
- Final test accuracy: 98.08%. Loss dropped every epoch, no overfitting.
- Test accuracy > train accuracy in early epochs — Dropout handicaps the model during training. Eval mode disables Dropout → cleaner predictions.

---

## Day 2 — ResNet-18 Inference + Feature Maps

### Environment & Setup
- Day 2 is inference only — no training loop, no dataset. Just load pretrained weights and run forward pass.
- `torchvision.models` — model hub for classic CV architectures (ResNet, MobileNet, VGG). Same concept as HuggingFace but scoped to vision.

### Preprocessing & Input Contract
- Every pretrained model has an input contract: the exact format it expects, derived from how it was trained. Mismatch → garbage predictions.
- ResNet-18's contract: RGB image, 224×224 center crop, normalized with ImageNet mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]`.
- Standard ImageNet eval pipeline: `Resize(256) → CenterCrop(224)` — not a direct resize. Resize(256) with a single int scales the shorter side, preserving aspect ratio. CenterCrop then cuts a clean square.
- `transforms.Resize(n)` (single int) = preserve aspect ratio, scale shorter side to n. `Resize((h, w))` (tuple) = force exact size, may distort.
- Normalization values are the ImageNet dataset statistics — computed once over 1.2M images, baked into every ImageNet-pretrained model forever.

### Batching
- PyTorch models always expect shape `(batch, channels, height, width)`. No single-image mode.
- `tensor.unsqueeze(0)` adds a size-1 batch dimension: `(3, 224, 224)` → `(1, 3, 224, 224)`.
- In training, DataLoader handles batching automatically. Without a DataLoader (inference on one image), you unsqueeze manually.
- `unsqueeze` happens after `Compose`, not inside it — Compose is a preprocessing pipeline, not a batching tool.

### Multimodal Models (conceptual)
- Multimodal models (GPT-4, Claude): one encoder per modality (image, text, audio), all output same-shaped vectors, one shared transformer processes everything.
- Each encoder has its own preprocessing contract. Preprocessing = fulfill that contract, convert raw input to vectors the shared model understands.
- CLIP (Day 5): explicitly trains image + text embeddings to land in the same region if semantically matched — not automatic, that's the design.

### Loading Pretrained Models
- `torchvision.models.resnet18(weights=...)` — architecture and weights are decoupled by design. Architecture is the blueprint (layer structure). Weights are what the model knows (learned numbers).
- Weights are NOT bundled in the package — downloaded from PyTorch servers on first use, cached in `~/.cache/torch`. Keeps the package small.
- `models.ResNet18_Weights.DEFAULT` — alias for the best available pretrained weights. Multiple versions exist (IMAGENET1K_V1, V2...) from different training runs. DEFAULT always points to the best.
- Loading without weights = random initialization = training from scratch. Valid when you want to train on your own dataset entirely.
- Same architecture + different weights = completely different model behavior. Weights encode all learned knowledge.

### model.eval() and inference mode
- `model.eval()` is a behavioral switch only — does NOT freeze or modify weights.
  - Dropout: disabled (no random neuron zeroing)
  - BatchNorm: switches from batch statistics to stored running_mean/running_var
- `torch.no_grad()` — stops building the computation graph. Saves memory and time. Independent from eval().
- Weights don't update at inference because there's no optimizer and no `backward()` call — not because of eval().
- For correct inference always use both: `model.eval()` + `with torch.no_grad()`.

### BatchNorm: training vs inference
- BatchNorm normalizes layer activations to keep values in a stable range throughout the network.
- γ (gamma) and β (beta) — learnable parameters, updated via backprop like any weight. Frozen at inference like all weights.
- running_mean and running_var — NOT learned via backprop. Updated as a side effect of every training forward pass via exponential moving average.
- At inference (eval mode): normalize using running_mean/running_var instead of current batch stats. Reason: a batch of 1 image has no meaningful distribution to compute stats from.

### Inference & Logits
- ResNet-18 final layer: `Linear(512 → 1000)` — 1000 raw scores (logits), one per ImageNet class.
- Softmax is intentionally NOT part of the model. Reason: CrossEntropyLoss applies softmax internally during training — baking it into the model would double-apply it → wrong gradients.
- At inference: apply softmax yourself when you want probabilities. For just the predicted class, `logits.argmax(dim=1)` is enough (softmax preserves order).
- CrossEntropyLoss requires a ground truth label to measure error. At inference there's no ground truth — use softmax alone for probabilities.
- `torch.topk(probs, k=5)` returns two tensors: the top-k values and their indices. Indices map to class names via `weights.meta["categories"]`.
- `CrossEntropyLoss = log_softmax + NLLLoss` — mathematically identical, CE just does it in one fused numerically stable operation. You pass raw logits to CEL directly; it applies softmax internally as its first step.
- Gradient of CE w.r.t. logits has a clean closed form: `softmax(logits) - one_hot(true_class)`. PyTorch computes this directly — no numerical instability, no step-by-step backprop through softmax.
- Normalization mismatch is a silent bug — the model won't crash, it'll just give confident wrong predictions. Always match the exact mean/std the model was trained with.
- `model.eval()` is critical for **correctness** (wrong Dropout/BatchNorm behavior without it). `torch.no_grad()` is critical for **performance** (wastes memory without it, but predictions still correct). Use both every time.

### Forward Hooks
- A hook is a callback attached to a layer that fires automatically every time that layer completes its forward pass. Same concept as event listeners or LangFuse callbacks on a graph.
- `layer.register_forward_hook(fn)` — fn receives (module, input, output). Grab output to save the feature map.
- The main forward pass continues unaffected — hook only intercepts, doesn't modify.
- Always call `hook.remove()` after use — don't leave listeners dangling.
- Use hooks when you want intermediate layer outputs without cutting the network short. Alternative: `model.fc = nn.Identity()` strips the head to get final features, but can't reach intermediate layers.

### ResNet-18 Architecture
- Named layer groups: `conv1` → `bn1` → `maxpool` → `layer1` → `layer2` → `layer3` → `layer4` → `avgpool` → `fc`.
- "18" = total count of weighted layers inside all groups, not number of named attributes.
- Spatial reduction before layer1: `conv1` stride=2 (224→112) + `maxpool` stride=2 (112→56). Layer1 output: `(1, 64, 56, 56)`.
- 64 filters in layer1 = 64 different patterns the layer learned to detect.

### Feature Maps & What Layers Learn
- Feature map = the output of a conv layer for a given input. Shape: `(batch, num_filters, H, W)`.
- Layer 1 (early): detects low-level patterns — edges, color gradients, contrast, luminance. Some filters visually resemble the original image (broad color response), some detect sharp outlines, some detect patterns not obvious to humans.
- Deeper layers: textures → semantic parts. Harder to visualize (spatial resolution too low).
- "Nonsense-looking" filters still contribute to the final prediction — they detect real patterns, just not ones humans can easily name.

### Determinism
- Inference is fully deterministic on CPU: same input + same weights = identical output every time.
- LLM outputs appear non-deterministic because of **sampling** at decoding time — the model outputs a probability distribution, then randomly samples from it. Temperature=0 → argmax → deterministic. Temperature>0 → sampled → varies each run. The architecture itself is deterministic.

---

## Day 3 — ONNX + Edge ML: Production Inference

### ONNX Format
- ONNX = Open Neural Network Exchange. Serializes computation graph + weights into a single portable file.
- Encoded in Protobuf (binary), not JSON or plain text.
- Two uses: (1) run via ONNX Runtime (lightweight inference), (2) convert to platform-specific formats (CoreML, TensorRT, OpenVINO).
- Export is one-way — you can't reload an ONNX file into PyTorch as a trainable model.
- `torch.onnx.export` traces the forward pass through a dummy input to record the graph. Dummy values don't matter — only shapes do.
- `dynamic_axes` — marks which dimensions can vary at runtime. Without it, batch size is frozen to whatever the dummy input used. Mark both input and output batch dimensions.
- `input_names` / `output_names` — always set them. Auto-generated names like `onnx::Conv_0` are unreadable and can change across re-exports.
- `onnx.checker.check_model` — validates the graph is well-formed. Always run after export.
- Model must be in `eval()` mode before export — otherwise Dropout randomness and BatchNorm batch stats get baked into the graph.

### ONNX Runtime (ORT)
- `ort.InferenceSession("model.onnx")` — loads and optimizes the graph at startup.
- Graph optimizations at load: op fusion (Conv+BN+ReLU → single kernel), constant folding, memory reuse, node elimination. All automatic, all-on by default.
- ORT is ~2× faster than plain PyTorch on CPU for ResNet-18 inference (97ms → 45ms mean latency) — because PyTorch carries full training machinery; ORT is inference-only.
- `session.run(None, {"input": np_array})` — None means return all output nodes. Key must match `input_names` from export.
- `session.get_inputs()[0].name` — how to find the input name if you didn't set it at export.
- Input must be numpy array (not PyTorch tensor). ORT doesn't accept torch tensors.

### Warm-up Runs
- Always do 5–10 warm-up runs before benchmarking. First calls trigger graph optimization, memory allocation, and (on GPU) CUDA kernel compilation — all of which skew timing.
- Use `time.perf_counter()` for wall-clock measurement, not `time.time()` (lower resolution).
- Report mean + p95, not just mean. Outliers matter in production.
- Batch size 1 = latency benchmark. Batch size 32 = throughput benchmark. Measure both.

### Quantization
- **FP32** (default): 4 bytes/weight. **FP16**: 2 bytes. **INT8**: 1 byte.
- **PTQ (Post-Training Quantization)**: quantize an already-trained model. No retraining needed.
- **QAT (Quantization-Aware Training)**: simulate quantization during training. Better accuracy, more work. NOT what `quantize_dynamic` does.
- **Dynamic PTQ**: weights pre-quantized to INT8, activations quantized on-the-fly per input. No calibration data needed. Simple but suboptimal.
- **Static PTQ**: run calibration data through first to pre-compute activation scale factors. Faster at inference, better accuracy, needs data.
- `quantize_dynamic` only supports `MatMul` and `Gemm` op types on CPU — `ConvInteger` is not implemented in ORT's CPUExecutionProvider. For CNNs (which are 99% Conv), dynamic INT8 gives almost no compression or speedup on CPU.
- INT8 quantization has real impact on Transformer/LSTM models (dominated by MatMul) and on GPU/edge hardware that natively supports ConvInteger.
- `model.half()` converts PyTorch model to FP16 before export. FP16 on CPU is often slower than FP32 — x86 CPUs have no native FP16 compute and convert back to FP32 anyway.

### FastAPI Serving
- FastAPI is async — built on Starlette + uvicorn event loop. `async def` + `await` lets the server handle other requests during I/O waits.
- `UploadFile` — FastAPI's type for file uploads. `File(...)` declares the field as form data; `...` (Python's Ellipsis) means required.
- `await file.read()` — async I/O read; yields control to event loop while waiting for bytes from network.
- `io.BytesIO` — wraps raw bytes in a file-like object. `PIL.Image.open()` needs a file handle, not raw bytes.
- `.convert("RGB")` — normalizes any input format (grayscale, RGBA, CMYK, palette) to 3-channel RGB. Without it, non-RGB uploads crash the preprocessing pipeline.
- Initialize session, labels, and preprocess pipeline at module level (startup), not inside the handler — avoid reconstructing on every request.
- `python-multipart` is a required dependency for FastAPI form/file uploads — not installed by default.
- FastAPI auto-generates Swagger UI at `/docs` — use it for manual endpoint testing.
- `argsort()` returns indices that would sort the array (ascending). `[-5:]` gives top-5 indices. `[::-1]` reverses to descending order (best first). `[:5]` alone would give the 5 *worst* predictions.

---

## Day 4 — CoreML Conversion + Parity Validation

### CoreML & Why It Exists
- CoreML is Apple's on-device inference format — optimized for their hardware stack: CPU, GPU, and the Apple Neural Engine (ANE).
- ANE is a dedicated ML chip on every modern iPhone/iPad/Mac. Extremely fast and power-efficient for supported ops.
- CoreML's compute hierarchy: ANE → GPU → CPU. CoreML picks automatically based on op support.
- CoreML is NOT cross-platform. ONNX Runtime doesn't know ANE exists — CoreML is the format that maps cleanly onto Apple hardware.
- `.mlmodel` = legacy single-file format. `.mlpackage` = modern directory bundle (separate files for weights, spec, metadata). Always use `.mlpackage` for new projects.

### TorchScript
- TorchScript is PyTorch's own portable IR (Intermediate Representation) — a frozen, serializable version of a model that can run without Python.
- IR = Intermediate Representation. Sits between source code (high level) and machine code (low level). A structured graph of ops that's portable and serializable.
- Two ways to produce TorchScript:
  - `torch.jit.trace` — runs a dummy input through the model, records every op that fires. Fast, simple, misses data-dependent control flow.
  - `torch.jit.script` — statically analyzes Python source, compiles to TorchScript. Handles all control flow but requires TorchScript-compatible code (restricted Python subset).
- Both produce TorchScript. The Python classes are `TopLevelTracedModule` (trace) and `RecursiveScriptModule` (script) — both inherit from `torch.jit.ScriptModule`.
- TorchScript is the format/concept, not a Python class.

### jit.trace vs jit.script — Practical Reality
- `trace` works on almost anything. `script` requires the code was written to be TorchScript-compatible — rare in third-party models.
- HuggingFace models almost universally fail to script — they use complex Python patterns (`**kwargs`, mixed-type dicts, optional outputs).
- "Dynamic branching" in this context = Python-level control flow that depends on tensor *values* at runtime. Different from ONNX dynamic axes (shape flexibility).
- ResNet has no data-dependent branching — trace is fine.
- Standard pipeline: trace + parity validation. If parity fails, something got missed.

### Why wrap jit.trace in torch.no_grad()
- Tracing runs a forward pass. Without `no_grad`, PyTorch builds both the TorchScript graph (intentional) AND the autograd computation graph (wasted). `no_grad` prevents the autograd graph from being built — no gradients needed, no backward pass ever called.

### CoreML Conversion Pipeline
- Pipeline: `PyTorch model → eval() → jit.trace → ct.convert() → .mlpackage`
- Must call `model.eval()` before tracing — otherwise Dropout randomness and BatchNorm batch stats get baked into the frozen graph permanently.
- `ct.convert()` takes the TorchScript graph and maps every op to CoreML's operator set.
- `minimum_deployment_target=ct.target.iOS15` sets the CoreML spec version (spec 6). iOS 15+ supports `.mlpackage`.

### ct.ImageType vs ct.TensorType
- `ct.TensorType` — raw multi-dimensional array input. Caller handles all preprocessing.
- `ct.ImageType` — image-specific input. Lets you bake normalization into the model so iOS caller can pass a raw `UIImage` directly.
- `ImageType` uses `scale` and `bias` (not `mean`/`std` — that API was removed in coremltools 7+).
  - Formula: `output = pixel * scale + bias`
  - To replicate PyTorch `Normalize(mean, std)`: `scale = 1/(255*avg_std)`, `bias = [-m/s for m, s in zip(mean, std)]`
- Pull mean/std from weights metadata — don't hardcode: `weights.transforms().mean`, `weights.transforms().std`
- `ImageType` is better practice for production image models — normalization is documented inside the model, less surface area for mismatch bugs.

### Parity Validation
- Run the same preprocessed input through both models, compare raw logits.
- Use **both** max diff + `np.allclose` together — they answer different questions:
  - `max_diff = np.max(np.abs(a - b))` — tells you the worst-case deviation. Useful for debugging: if parity fails, you know how bad it is.
  - `np.allclose(a, b, atol=1e-4, rtol=1e-4)` — checks every element satisfies `|a - b| <= atol + rtol * |b|`. The `rtol` term scales the tolerance by the magnitude of `b` — a diff of 0.01 on a logit of 100.0 is tiny (0.01%), but the same diff on a logit of 0.01 is 100% error. Flat `atol` alone doesn't distinguish these.
  - `np.allclose` alone = black-box True/False. Max diff alone = ignores relative magnitude. Use both.
- Slight numerical differences are expected — different FP precision in op implementations. Big differences = conversion bug.
- Most parity failures are preprocessing mismatches, not conversion bugs. Rule out that first.
- If trace missed a conditional: try `jit.script`, put model in export mode (`return_dict=False`, `use_cache=False`), or use HuggingFace `optimum`.

### HuggingFace optimum
- HuggingFace's export library. Knows how to correctly export each model family — sets the right config flags, handles model-specific quirks, validates parity.
- Has dedicated exporters for ONNX, CoreML, TensorRT, OpenVINO.
- Use it instead of hand-rolling trace for any complex HuggingFace model.

### Platform Limitations (Real-World)
- `coremltools` supports Linux for **conversion** but NOT for **inference** (removed in coremltools 8.0+).
- CoreML inference requires macOS 10.13+ — the framework is Apple-only.
- On Linux: convert, inspect spec, validate structurally. Actual `predict()` requires macOS.
- Workaround for parity on Linux: use ONNX as a proxy — both PyTorch and ONNX run on Linux.

### File Formats & Size on Disk
- `.mlpackage` is a **directory bundle** — separate files for spec (Protobuf), weights, and metadata. `os.path.getsize()` returns the directory entry size (a few bytes), not the total. Use `du -sh` (or `subprocess.run(["du", "-sh", path])`) to walk the tree and sum.
- `.onnx` is a **single flatbuffer file** (Protobuf-encoded). `os.path.getsize()` works fine.
- CoreML's `mlProgram` format stores weights in **float16 by default** → ~half the size of an ONNX export, which defaults to float32. ResNet-18: CoreML ≈ 23 MB, ONNX ≈ 47 MB, both encoding the same 11.7M parameters.

### Misc Python & Tools
- `transforms.Compose` returns a callable object — implements `__call__`. `preprocess(image)` = `preprocess.__call__(image)`. Same pattern as PyTorch models (`model(x)` calls `model.__call__(x)` which calls `forward(x)`).
- `subprocess.run(["du", "-sh", "path"], capture_output=True, text=True)` — run shell commands from Python. `capture_output=True` = don't print to terminal. `text=True` = return string not bytes.
- `du -sh` = disk usage, summary, human-readable. Use instead of `os.path.getsize()` for directories.
- `spec.WhichOneof("Type")` — Protobuf method. Returns the name of whichever variant of a `oneof` field is currently set. Same concept as `instanceof` for a tagged union.
- `softmax(logits, dim=1)` — `dim=1` applies softmax across the 1000 class scores per image, not across the batch dimension.

---

## Day 5 — CLIP Embeddings + Photo Clustering

### Embeddings
- An embedding is a dense vector representing semantic content. Each dimension captures some learned feature — not hand-crafted, learned from training data.
- ML convention: **rows = samples, columns = features**. A dataset of 200 photos with 512-dim embeddings = `(200, 512)` matrix. Always.
- Two images with similar content have embeddings that **point in the same direction** — this is why cosine similarity (angle between vectors) is used, not Euclidean distance. Magnitude varies (brightness, contrast), direction captures semantics.

### CLIP vs CNN Embeddings
- CNN backbone (ResNet): trained to classify 1000 ImageNet classes. Embeddings are good at distinguishing those classes but the semantic space is coarse.
- CLIP: two encoders (image + text) trained contrastively on 400M image-text pairs. Pulls matching pairs together, pushes mismatched pairs apart.
- CLIP embeddings are semantically richer — understands "beach", "wedding", "food" as concepts, not just ImageNet categories. Two beach photos with different lighting/people still land close together.
- CLIP enables 3 operations out of the box: (1) image→text search, (2) text→image search, (3) zero-shot classification.
- **CLIP is not always superior** — ResNet wins for: (1) fine-tuned tasks (ResNet fine-tuned on your specific dataset can beat general CLIP), (2) low-level visual similarity (style, texture, duplicate/burst shot detection — where visual composition matters more than semantic content). CLIP can also be fine-tuned and is very powerful when fine-tuned on domain-specific data.

### ViT Architecture (Vision Transformer)
- ViT cuts the image into a grid of non-overlapping patches, flattens each into a vector, feeds all patches as tokens into a transformer.
- Patch size is the size of each square: ViT-B/32 = 32×32 patches, ViT-B/16 = 16×16 patches.
- Bigger patch size = fewer patches = fewer transformer tokens = faster. ViT-B/32 is faster than ViT-B/16.
  - ViT-B/32: 224×224 image → 7×7 grid = 49 patches
  - ViT-B/16: 224×224 image → 14×14 grid = 196 patches
- Speed order fastest→slowest: `RN50 → ViT-B/32 → ViT-B/16 → ViT-L/14`
- A patch is NOT a filter. A filter slides across the image detecting patterns. A patch is just a dumb fixed crop — the transformer does all the work across patches together.

### CLIP API
- `model, preprocess = clip.load("ViT-B/32", device=device)` — returns model AND the exact preprocess transform together. No guessing mean/std.
- `preprocess` does the full pipeline: resize → center crop to 224×224 → tensor → normalize with CLIP's own mean/std (different from ImageNet's).
- `model.encode_image(tensor)` → 512-dim embedding vector per image.
- Model identifiers are plain strings in openai-clip — no enum. Typo = RuntimeError at load time.
- `model.visual.input_resolution` = 224. `model.visual.output_dim` = 512.

### Embedding Loop Pattern
- Loop over images one at a time: `Image.open → preprocess → unsqueeze(0) → .to(device) → encode_image → .cpu().numpy()`
- `unsqueeze(0)`: adds batch dimension. `(3, 224, 224)` → `(1, 3, 224, 224)`. Required because model always expects a batch.
- `.cpu()` before `.numpy()` — numpy can't read GPU memory. No-op on CPU, essential on GPU.
- `np.vstack(list_of_arrays)` — stacks a list of `(1, 512)` arrays into `(200, 512)`. Vertical stack = rows on top of each other.
- `np.hstack` = horizontal stack (axis=1). `np.vstack` = vertical stack (axis=0).

### K-Means Clustering
- Algorithm: initialize k centroids → assign each point to nearest centroid → recompute centroids as mean of cluster → repeat until no assignments change.
- Convergence = when no point switches clusters between iterations.
- k-means++ initialization: place first centroid randomly, then each subsequent centroid is chosen with probability proportional to its distance from existing centroids. Avoids bad random starts.
- `sklearn.cluster.KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)` — sklearn uses k-means++ by default.
- `n_init=10` — runs the full algorithm 10 times with different initializations, keeps the best result. Reduces sensitivity to initialization.
- **Inertia** = sum of squared distances from each point to its assigned centroid. Lower = tighter clusters. Always decreases as k increases.
- **Elbow method** — plot inertia vs k, pick the k where the curve flattens (diminishing returns on adding more clusters). No hard rule — visual judgment call.
- **Elbow method fails in high dimensions** — in 512D space, distances between points are nearly uniform (curse of dimensionality). Inertia decreases at a constant rate regardless of k → linear plot, no visible elbow. This is a visualization problem, not a clustering problem.
- **PCA as a fix for elbow visualization** — compress 512D → 50D before plotting inertia. Low-dimensional distances are meaningful again, elbow emerges. Use PCA only for finding k — then run actual k-means on the full 512D embeddings. PCA discards information that CLIP deliberately encoded; don't cluster on compressed embeddings.
- **PCA (Principal Component Analysis)** — finds directions of maximum variance in the data, projects everything onto those directions. Keeps top N components, discards the rest. Lossy. Motivations beyond dimension reduction: noise removal (low-variance dims are often noise), visualization (compress to 2D/3D to plot), decorrelation (PCA components are orthogonal). Limitation: linear only — misses non-linear cluster structure. `sklearn.decomposition.PCA(n_components=50).fit_transform(embeddings_norm)`.

### Dataset Choice & Clustering Quality

- **food101 failed as a clustering demo** — all images are food. CLIP embeddings form one dense blob with no inter-cluster gaps. Inertia dropped linearly regardless of k. The elbow method produced no signal. The 2D PCA scatter showed no separation (only 12.7% variance explained in 2D).
- **Root cause: semantic narrowness** — CLIP clusters by semantic domain. Within a single domain (all food), subtle differences between pizza and sushi are real but small compared to food vs. animals vs. vehicles. No algorithm can create gaps that don't exist in the data.
- **Takeaway for production**: dataset choice matters as much as the algorithm. For meaningful photo clustering (e.g. automatic album generation in a consumer photo product), you need photos spanning genuinely different semantic domains — people, landscapes, food, events, pets. A user's camera roll has this diversity naturally. A curated single-category dataset doesn't.
- **Fix**: combine multiple datasets across domains (food + pets + landscapes). Each domain lands in a distinct region of CLIP's embedding space → clear cluster separation → elbow visible → grid visualization makes sense.

### Visualization of High-Dimensional Embeddings
- **PCA to 2D fails for CLIP** — only captures ~12% of 512D variance. The remaining 88% of cluster structure is invisible. Scatter plot looks like noise.
- **t-SNE (t-distributed Stochastic Neighbor Embedding)** — non-linear dimensionality reduction designed for visualization. How it works: (1) in high-D space, compute a probability distribution over neighbors using a Gaussian — close points get high probability; (2) initialize random 2D positions, compute same distribution using a t-distribution (fatter tails); (3) gradient descent moves 2D points until the distributions match. The t-distribution's fatter tails push far-apart points more aggressively apart than they were in high-D — this creates visible gaps between clusters. No eigenvectors, no closed-form solution — pure iterative optimization.
- t-SNE vs PCA: PCA is linear (fast, deterministic, preserves global structure). t-SNE is non-linear (slower, non-deterministic, preserves local cluster structure). For visualizing clusters, t-SNE wins.
- t-SNE axes have no meaning — only relative positions matter. You can't read "PC1" off a t-SNE plot.
- `sklearn.manifold.TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings_norm)`
- **UMAP** is a faster, more modern alternative to t-SNE with similar cluster-preserving properties. Standard in industry for embedding visualization.
- **t-SNE still looks messy with sparse data** — 2 images per class across 125 classes means k-means has almost no data per cluster. For clean t-SNE blobs: 10-20 images per class minimum. Small source images (64×64) also degrade CLIP signal since upscaling loses detail.
- **Image grid is a better sanity check than scatter plot** — look at whether photos in the same row are visually similar. If yes, the clustering is working even if the 2D plot looks noisy.

### At Scale — Production Clustering
- **FAISS (Facebook AI Similarity Search)** — library for fast nearest neighbor search and clustering at scale. Two uses: (1) `faiss.Kmeans` — GPU-accelerated k-means, handles millions of vectors, manages its own batching internally. Much faster than sklearn at scale. (2) similarity search — find nearest neighbor vectors to a query, O(log n) via HNSW index vs O(n) brute force in a RDBMS.
- **sklearn KMeans doesn't support cosine distance** — hardcoded Euclidean. The centroid update step (mean of cluster members) is only mathematically valid in Euclidean space. Fix: L2 normalize before clustering. Spherical k-means or `faiss` with cosine index for native cosine clustering.
- **Per-user clustering** — never cluster all users' photos together. Each user's photos are clustered independently. 10M users = 10M small jobs, not one giant job.
- **Distributed task queue** — for running 10M clustering jobs: push a job per user to a queue, worker machines pick up and process. **Celery** = Python task queue framework (`@app.task` decorator, `.delay()` to enqueue). **RabbitMQ/Redis** = the broker underneath (message transport). **SQS** = AWS's managed broker. Celery is the framework; RabbitMQ/SQS is the transport. Unlike `deque` (in-process, single machine), distributed queues work across many machines.
- **Vector databases** (Pinecone, Weaviate) — store embeddings with fast nearest-neighbor index. Use when you need to search across all users' embeddings ("find photos similar to this one"). For per-user clustering, plain numpy arrays are fine — only 500 vectors per user.
- **Multi-signal clustering** — pure CLIP is semantics only. Two birthday parties 6 months apart land in the same cluster. Production systems combine: CLIP embedding + timestamp + GPS location + face embeddings. Dataset choice matters as much as the algorithm.
- **GPU batch size** — determined by VRAM. Each image tensor takes memory; at some batch size you OOM and crash. 256 is a safe default for most GPUs. Tune up until OOM, then back off. Not a fixed number.
- **MiniBatchKMeans** — sklearn's large-scale k-means. Manages mini-batching internally (`batch_size` param). Slightly less accurate than full k-means but much faster on large data. CPU alternative to `faiss.Kmeans`.
- **Incremental clustering** — re-clustering all photos from scratch when a user uploads 10 new photos is expensive. Production strategy: assign new photos to nearest existing centroid immediately, re-cluster periodically or when enough new photos accumulate.
- **Redis as Celery broker** — uses lists (LPUSH/BRPOP), not Pub/Sub. Producer pushes to list, worker blocks waiting to pop. Message persists until consumed. Redis Pub/Sub = fire-and-forget, multiple subscribers, message lost if no one listening. Different Redis feature, same server.

### HuggingFace Datasets
- `streaming=True` — returns an iterator instead of downloading the full dataset. Pull images one at a time, stop when you have enough. Nothing else hits your disk.
- Without streaming: full dataset downloads before you can access a single sample.
- Newer `datasets` library dropped support for loading scripts (`.py` files). Only accepts parquet-format datasets. `frgfm/imagenette` uses an old script → crashes. `food101` is parquet → works.
- Parquet = columnar binary file format. Stores structured data as static files — no code needed to load, just read the file.

---

## Day 6 — Transfer Learning & Fine-Tuning MobileNet

### Transfer Learning — Core Idea
- **Transfer learning reuses the learned *features*, not the learned *classes*.** A model pretrained on ImageNet (1.2M images, 1000 classes) has learned a general visual hierarchy. We keep that and relearn only the final mapping to our task.
- **Early layers = generic, late layers = task-specific.** Early conv layers detect edges/textures/colors — universal to all vision. The deepest layers specialize into ImageNet-specific semantics (e.g. "this texture combo = golden retriever"). This asymmetry drives everything: when fine-tuning we unfreeze **from the back**, because the late layers are the most ImageNet-biased and most need to re-adapt; the early edge detectors can stay frozen forever.
- **Two modes, done in sequence:**
  - **Feature extraction** — freeze the whole backbone, train only a new head. Bets that the existing 1280-dim features already separate your classes with a linear map. Fast, low overfit risk, can't damage pretrained weights.
  - **Fine-tuning** — unfreeze some backbone layers and train them too with a tiny LR. Lets the features themselves shift to specialize for your task.
- **Which you need depends on task distance from ImageNet + data volume.** Close to ImageNet (cats/dogs) + little data → head-only often suffices. Far (medical, satellite) or lots of data → must fine-tune the backbone. The project does both and *measures the delta* — sometimes fine-tuning buys almost nothing, and knowing that is the lesson.

### The Head Swap
- A pretrained classifier's final layer maps features → its original classes (MobileNetV2: `Linear(1280 → 1000)`). There is **no single "cat" or "dog" output neuron** — ImageNet has ~120 dog breeds + 5 cat-breed classes scattered across the 1000. So you can't argmax over "cat vs dog" — those outputs don't exist.
- **Fix: delete the entire final layer, bolt on a fresh `Linear(1280 → 2)`.** The new head taps the **1280-dim feature vector one layer earlier** (before the old head) — it does NOT read/merge the old class logits. The old 1000-class mapping is discarded entirely.
- The input dim (1280) is **forced** (it's what the backbone emits); the output dim (2) is **our choice** (num classes). Read `in_features` off the model (`model.classifier[1].in_features`) — never hardcode.
- This generalizes to classes *not in ImageNet at all* (defect detection, tumor vs healthy) — you're never reading the old outputs, just the features beneath them.

### FC head vs MLP head (head design)
- **FC head** = one `Linear` (a linear probe). **MLP head** = two+ `Linear` with a non-linearity between (can learn non-linear feature combos).
- **Default to a single `Linear`.** The backbone already did the non-linear work — its 1280-dim vector is mostly linearly separable, so a linear classifier is usually enough. Extra hidden layers mostly add params that overfit small datasets.
- **The key insight:** if a single Linear head underperforms, the better fix is usually **unfreeze more backbone** (adapt the powerful conv features) — not **deepen the head** (stack weak FC layers). Conv adaptation beats FC capacity.
- MLP heads shine in self-supervised learning (SimCLR/MoCo projection heads) and large-data fine-tuning where overfitting isn't the bottleneck.

### MobileNet & Depthwise Separable Convolution
- MobileNet is built for edge/phones (the reason it matters for on-device photo processing). Its signature is the **depthwise separable convolution**.
- **A normal conv does two jobs at once** (expensive): spatial filtering (look at a KxK neighborhood) AND channel mixing (combine all input channels). Cost ≈ `H·W·C_in·C_out·K·K`.
- **Depthwise separable splits these into two cheap steps:**
  - **Depthwise conv** — one KxK filter *per input channel*, independently. Spatial only, no channel mixing. Cost ≈ `H·W·C_in·K·K` (no C_out term).
  - **Pointwise conv** — a 1×1 conv that mixes channels C_in → C_out. Channel mixing only, no spatial window. Cost ≈ `H·W·C_in·C_out` (no K·K term).
- Net: **~8–9× fewer operations** for 3×3 kernels (ratio `1/C_out + 1/K²`). That ~9× is the entire reason MobileNet runs in real time on a phone and lands in CoreML/TFLite instead of ResNet.
- Interview one-liner: *normal conv mixes space and channels in one expensive op; depthwise separable factors it into a per-channel spatial filter + a 1×1 channel mixer, ~8–9× cheaper.*
- MobileNetV2 structure: `model.features` (depthwise-separable backbone → `(1280,7,7)` feature map → pooled to 1280-dim vector) + `model.classifier` (`Sequential(Dropout(0.2), Linear(1280→1000))`). Only ~2.2M params total vs ResNet-18's ~11.7M.

### Freezing — Two Separate Locks
- **Lock 1 — `requires_grad = False`** on a param freezes its **weights only**: no gradient computed, optimizer can't move it. (Covers conv weights AND BatchNorm's learnable γ/β.)
- **Lock 2 — `.eval()` mode** is needed *separately* to freeze BatchNorm's **running stats**. Those (`running_mean`/`running_var`) are **buffers, not parameters** — `requires_grad` does not touch them. In `train()` mode they keep drifting via EMA regardless of `requires_grad`.
- **The gotcha:** freeze the backbone with `requires_grad=False` but leave it in `train()` mode, and its BN running stats silently drift from ImageNet values toward your small/narrow dataset's stats → the "frozen" backbone computes different features each epoch. Especially harmful with small batches from a narrow domain.
- **Proper feature extraction sets modes per-section:** `model.features.eval()` (freeze BN stats) + `model.classifier.train()` (keep Dropout active). A global `model.train()` would un-freeze BN; a global `model.eval()` would kill the head's Dropout.
- One-liner: **`requires_grad=False` freezes the weights; only `.eval()` freezes BatchNorm's running stats. Two independent locks.**

### BatchNorm — Batch Stats vs Running Stats (EMA)
- BN normalizes activations to ~mean 0 / var 1 so the next layer sees a stable distribution. Question is: mean/var over *what*?
- **`train()` mode:** normalizes using the **current batch's** mean/var (measured live from the 32 images in front of it). As a *side effect*, updates the running stats via EMA. The thing that "drifts" is the running buffer, not the normalization-in-use.
- **`eval()` mode:** normalizes using the stored **running stats** (the accumulated EMA), and stops updating them.
- **EMA (Exponential Moving Average):** `running = (1-momentum)*running + momentum*batch_stat` (PyTorch momentum default 0.1 → 10% new, 90% history). A cheap, smoothly-updating estimate of the dataset-wide mean/var that exponentially forgets old values.
- **Why two sets:** at training you have full batches → use real batch stats; quietly accumulate the dataset-wide EMA in the background. At inference you might have 1 image (no meaningful batch variance) → fall back on the EMA. Bonus: eval becomes **batch-independent / deterministic** — the same image gets the same prediction regardless of what's batched alongside it.

### Train / Validation / Test — Three Roles
- **Train** — model learns from it (backprop updates weights). **Validation** — evaluated mid-training (forward pass, no backprop) to *steer decisions* (when to stop, which checkpoint is "best", hyperparameter choices). **Test** — touched once at the very end for an honest final number.
- The split is about **who is tuned by what:** train tunes *weights* (gradient descent); validation tunes *your choices* (hyperparameters, model selection); test tunes *nothing*.
- **Why val ≠ test:** the moment a set is used to *make a decision* it's "used up" — you've fit to it (at the hyperparameter level). Reporting val would be optimistic because we selected the model that maximizes exactly that number. Test influenced no decision → unbiased estimate.
- Analogy: train = homework, validation = practice exam (adjust how you study), test = the real final (graded once).
- **Overfitting is read from the train-vs-val gap:** train metric ↑ while val plateaus/↓ = memorizing. Validation is a read-only probe of the current model state used for early stopping, checkpointing, and tuning.

### Data Augmentation & Two-Pipeline Split
- **Augment training only.** Augmentation (random crop/flip/jitter) diversifies training data → less memorization → better generalization. Applying it to val/test would measure performance on distorted inputs you'll never see → noisy, non-reproducible metric. (Narrow exception: deliberate test-time augmentation/TTA, applied at prediction time, not to define the metric.)
- Val/test use a **deterministic** pipeline (no randomness) so the metric reflects the real distribution and is reproducible.
- **`RandomResizedCrop(224)` is a self-contained resize+crop** — it samples a random area/aspect from the full-res original and resizes that crop to 224. No prior `Resize` needed (one would just discard resolution before the crop). Eval path *does* need `Resize(256) → CenterCrop(224)` because `CenterCrop` is dumb (grabs middle pixels, no scaling) and needs scale standardized first. Asymmetry is intentional: random get-to-224 for train, fixed get-to-224 for eval.
- **Two-dataset trick:** a transform is bound to the dataset object, but train and val often come from the *same* split. Build **two dataset objects over that split** (one augmented, one clean) and give them **disjoint indices** via `Subset`. Train indices read the augmented copy; val indices read the clean copy.
- **`Subset(dataset, indices)`** is a lazy re-numbered view: `subset[k]` → `dataset[indices[k]]`. It forwards to the underlying dataset (and its transform). It's a `Dataset` (`__getitem__`/`__len__`), which is exactly what `DataLoader` needs — unlike `zip`, which is a one-shot iterator with no indexing/len and would force eager loading (breaking shuffle, multi-epoch, and per-set transforms).
- Indexing is **lazy end-to-end**: `_bin_labels`-style label lists let you plan a balanced split using integers only; no JPEG is decoded until a DataLoader batch is actually requested. That's why datasets bigger than RAM still train.
- **Balanced subsetting via per-class loop:** loop over class labels (`for cls in [0,1]`), `np.where(labels==cls)` to get that class's positions, shuffle (seeded), cap per class, then slice into disjoint train/val pieces. Looping over *classes* (not images) is what gives per-class count control on a naturally imbalanced dataset (Oxford Pet is ~2:1 dog-heavy).

### Optimizer / Loss / Autograd — Three Decoupled Actors
- Build the optimizer over **only trainable params**: `optim.Adam([p for p in model.parameters() if p.requires_grad], lr=...)`. Filter by the *property* (`requires_grad`), not a *location* (`model.classifier`) — they coincide in Phase 1 but diverge once Phase 2 unfreezes backbone blocks (location-based would silently omit them → optimizer never updates the layers you meant to fine-tune; a no-error bug).
- Why filter (beyond correctness): Adam allocates per-param state (momentum + variance buffers) for every registered param — wasteful on frozen ones. And **weight decay can still nudge a "frozen" param** that's in the optimizer's list (the decay term `-lr*wd*param` doesn't depend on `.grad`). `requires_grad` filtering removes that landmine. Belt-and-suspenders: `requires_grad` blocks the gradient, the param list blocks registration.
- **The optimizer does NOT call `backward()`.** Three separate actors communicating through shared tensors:
  - `criterion(logits, labels)` → **measures** wrongness (the loss). Decoupled from the optimizer — swap loss or optimizer independently.
  - `loss.backward()` → **autograd** walks the recorded computation graph and writes `.grad` on every trainable param.
  - `optimizer.step()` → **reads** those `.grad` values and applies the update; only touches its registered params.
  - The link between backward and step is **not a function call — it's the shared `.grad` field.** backward writes it, step reads it.
- `zero_grad()` vs `no_grad()` are nearly opposites: `zero_grad` *erases* gradients already computed (needed every train batch because PyTorch accumulates `.grad`); `no_grad` *prevents* gradient computation entirely (used in eval — no graph built). You'd never use both in the same scope.
- CrossEntropyLoss takes **raw logits** (applies softmax internally); never softmax before it.

### Loss vs F1 — They Measure Different Things
- **F1/accuracy judge the discrete decision** (argmax → right/wrong). Confidence is irrelevant — 51% correct and 99% correct both count as one "correct".
- **Cross-entropy loss judges confidence continuously** (`-log(p_correct)`). Even a correct prediction pays a penalty unless it was made with ~100% confidence.
- So **high F1 + non-zero loss is normal**: the model is reliably correct but not maximally confident, and/or a few *confident* errors inflate loss (loss is unbounded per mistake — a confident wrong answer can contribute ~4; F1 caps each error as one flipped verdict).
- **Loss keeps dropping after F1 plateaus** = same predictions, rising confidence. F1 is a step function (flat until a prediction flips); loss is smooth.
- We **optimize loss** (smooth, differentiable — can't backprop through argmax) but **report F1/accuracy** (the decision-level business metric).

### Checkpointing — Best, Not Last
- Save `{model.state_dict(), optimizer.state_dict(), epoch, val_f1}` so a run is fully resumable (optimizer state matters for Adam's momentum/variance buffers).
- **Checkpoint on best *val* metric, not every epoch and not the last epoch** — the last epoch may be overfit; the best-generalizing epoch may be earlier. Reloading the best checkpoint (not current in-memory weights) is the whole point of checkpointing.
- **Which metric to select on is a design choice:** match it to the deployment objective. Report/serve hard labels → select on F1 (loss as tie-breaker). Use probabilities (ranking/thresholding/calibration) → select on loss. "More confident" only helps if you actually consume the probabilities; and "loss down, F1 flat" can be an early overfitting signature.

### Precision / Recall / F1 / Confusion Matrix
- **Recall = the class's perspective:** "of all actual Xs, how many did I catch?" `TP/(TP+FN)`. Misses (false negatives) hurt recall.
- **Precision = the prediction's trustworthiness:** "when I say X, how often am I right?" `TP/(TP+FP)`. False alarms (false positives) hurt precision.
- **They trade off:** making the model eager to predict X raises X's recall but lowers X's precision. **F1 = harmonic mean** punishes sacrificing either → honest single number.
- A clean `classification_report` lets you **reconstruct the confusion matrix**: predicted-count = TP/precision, actual-count = support, fill in the off-diagonal from recall.
- **`average="macro"`** = unweighted mean of each class's F1 → both classes count equally regardless of imbalance. (Weighted avg would let the majority class dominate.)
- **Why F1 > accuracy (interview):** on an imbalanced set, "always predict majority" can score high accuracy (e.g. 98.3% if 590/600 are dogs) while having 0% recall on the minority class. Accuracy hides this; macro-F1 craters (minority F1 = 0, averaged in) and exposes it. On a *balanced* set the gap doesn't show — but that imbalanced scenario is exactly why the metric exists.

### Results (this run)
- Oxford-IIIT Pet, binary cat/dog, balanced subset (train 800 / val 200 / test 600), MobileNetV2, CPU.
- **Phase 1 (head only, backbone frozen):** untrained baseline val acc 16.5% (random head can be below chance — it's an arbitrary linear projection). After 4 epochs: val macro-F1 0.985, **test macro-F1 0.9917 / acc 99.2%** (5 errors / 600), near-symmetric per-class. Cats/dogs are squarely in ImageNet's wheelhouse → frozen features + a 2,562-param linear head nearly solve it. Best epoch was 3, not 4 (checkpointing recovered it).
- Even with a frozen backbone, every batch still does a full forward pass through it (the expensive part) — a known optimization is to pre-compute and cache features once, then train the head on cached vectors.

### Progressive Unfreezing — Why From the Back, Why Not All At Once
- **Unfreeze from the output end inward.** The last blocks hold the most task-specific (most ImageNet-biased) features and benefit most from re-adapting; the early edge/texture detectors are universal and can stay frozen forever. We unfroze MobileNetV2's `features[17]` (last inverted-residual block) and `features[18]` (final 1×1 pointwise conv → 1280 channels), leaving 0–16 frozen.
- **Don't unfreeze everything on a small dataset.** Unfreezing blocks 17–18 already exposed **886K params (39.9% of the model) to 800 training images** — an aggressive ratio. More unfrozen params with too little data → the optimizer drags pretrained weights *away* from their good ImageNet state without enough signal to land somewhere better (a soft form of catastrophic forgetting). Fewer trainable params is the safer default; scale up only with more data/epochs.
- **Per-block mode must follow per-block `requires_grad`.** In Phase 2 the model is *mixed* — some blocks trainable, some frozen — so a single "freeze backbone" flag can't express it. Drive each block's `.train()`/`.eval()` off `any(p.requires_grad for p in block.parameters())`: unfrozen blocks `train()` (their BN learns from batches), frozen blocks `eval()` (BN stays on ImageNet stats), head always `train()` (Dropout on).

### Differential / Discriminative Learning Rate (LLRD)
- **Different param groups get different LRs in one optimizer**, via a list of dicts: `optim.Adam([{"params": backbone, "lr": 1e-5}, {"params": head, "lr": 1e-4}])`. `"params"` is the only required key; any other key (`lr`, `weight_decay`, `betas`, …) overrides the optimizer-level default *for that group only*. Not arbitrary kwargs — the keys are exactly the optimizer constructor's args.
- **Why split:** the backbone is pretrained and near-optimal → it needs only a tiny nudge (`1e-5`), a big LR would cause catastrophic forgetting. The head is cheap/replaceable → it can take a larger step (`1e-4`). We used backbone = `FINETUNE_LR/10`, head = `FINETUNE_LR`.
- **LLRD (Layer-wise LR Decay)** generalizes this: every layer gets a geometrically decaying LR, smallest at the input, largest at the head (e.g. block18 `lr`, block17 `lr·0.8`, block16 `lr·0.64`, …). The scale-up technique when you unfreeze many layers instead of just two.
- **Must rebuild the optimizer after changing `requires_grad`.** An optimizer's param list is a **construction-time snapshot**. The Phase 1 optimizer holds only the head; flipping backbone blocks to trainable does *not* retroactively register them — the old optimizer would never update them. Construct a fresh `optim.Adam` with the new param groups.

### Adam Cold-Start Instability at Phase Boundaries
- **The training curve spiked at the Phase 1→2 boundary:** train loss jumped ~0.12 → 0.17 and train F1 *dipped* on the first fine-tune epoch, even though we lowered the LR. Cause: the rebuilt optimizer's Adam state (per-param momentum `m` and variance `v` buffers) is **re-initialized to zero** — no accumulated gradient history. The backbone weights start moving for the first time, momentarily disrupting the representations the head had learned to read.
- **Val barely moved** (F1 flat at 0.985, val loss kept dropping) — the backbone nudge at `1e-5` was small enough not to break generalization, just enough to cause a transient *training* wobble.
- This is the motivation for **LR warm-up** in full fine-tuning: ramp the LR from near-zero over the first epochs so Adam accumulates gradient statistics before taking large steps. We skipped it here (3 epochs is too short to matter).

### When Fine-Tuning Doesn't Help (the actual result)
- **Phase 2 *regressed*: test macro-F1 0.9917 → 0.9833 (−0.0083), acc 99.2% → 98.3%.** Fine-tuning made it *worse*, and recognizing that is the lesson.
- **Why:** (1) the task is too easy / too close to ImageNet — Phase 1 already left only ~5 errors on 600 images, so there was no useful signal left for the backbone to learn; (2) too many unfrozen params (886K) for too little data (800 imgs); (3) val set too small (200 imgs) → high-variance checkpoint selection, so the "best" Phase 2 epoch isn't reliably the best-generalizing one.
- **Takeaway:** **feature extraction (head-only) is often the right answer for small datasets on near-ImageNet tasks.** Backbone fine-tuning earns its keep when the domain is *far* from ImageNet (medical, satellite, documents) or you have thousands of examples per class. Always *measure the delta* rather than assuming more training = better.
- **Keep phase checkpoints separate** (`best_model.pt` vs `best_model_ft.pt`) precisely so you can compare and fall back — here you'd ship the Phase 1 model.
