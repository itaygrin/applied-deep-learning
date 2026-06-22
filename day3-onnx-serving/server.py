import io
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
import urllib.request

url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
with urllib.request.urlopen(url) as r:
    LABELS = json.load(r)

session = ort.InferenceSession("resnet18_fp32.onnx")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    tensor = preprocess(image)
    tensor = tensor.unsqueeze(0)
    inp = tensor.numpy()

    logits = session.run(None, {"input": inp})[0]
    probs  = np.exp(logits) / np.exp(logits).sum()

    top5_idx = probs[0].argsort()[-5:][::-1]
    top5 = [{"class": LABELS[i], "confidence": float(probs[0][i])} for i in top5_idx]

    return {"predictions": top5}
