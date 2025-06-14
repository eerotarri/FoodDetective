from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

# Allow frontend (React) access later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model (you can use yolov5s.pt or yolov8n.pt)
model = YOLO("yolov8n.pt")  # Smallest model; works without GPU

@app.post("/detect/")
async def detect_items(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run YOLOv8 inference
        results = model(image)

        # Extract labels
        detected_items = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            label = results[0].names[cls_id]
            detected_items.append(label)

        return {"items": list(set(detected_items))}

    except Exception as e:
        return {"error": str(e)}
