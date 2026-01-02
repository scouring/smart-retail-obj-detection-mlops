from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io
import torch
import base64

app = FastAPI()

device = "cpu"
model = YOLO("models/best.pt").to(device)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model.predict(source=img, device=device, conf=0.25)

    draw = ImageDraw.Draw(img)

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls)
            label = model.names[cls]
            conf = float(box.conf)

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 10), f"{label} {conf:.2f}", fill="red")

            detections.append({
                "class": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    return {
        "detections": detections,
        "annotated_image_base64": img_base64
    }
