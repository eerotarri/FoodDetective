import os
from fastapi import FastAPI, UploadFile, File
from google.cloud import vision

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-credentials.json"

app = FastAPI()
client = vision.ImageAnnotatorClient()

CONFIDENCE_THRESHOLD = 0.7

@app.post("/detect")
async def detect_food_items(file: UploadFile = File(...)):
    content = await file.read()
    image = vision.Image(content=content)

    # 1. Web Detection
    web_entities = []
    web_response = client.web_detection(image=image)
    if web_response.web_detection and web_response.web_detection.web_entities:
        web_entities = [
            entity.description.lower()
            for entity in web_response.web_detection.web_entities
            if entity.score >= 0.6 and entity.description
        ]

    # # 2. Text Detection (OCR)
    # ocr_texts = []
    # text_response = client.text_detection(image=image)
    # if text_response.text_annotations:
    #     # The first annotation is usually the full block
    #     ocr_texts = [
    #         annotation.description.lower()
    #         for annotation in text_response.text_annotations[1:]  # skip full block
    #     ]

    # Combine and deduplicate
    combined = web_entities # + ocr_texts
    unique = list(dict.fromkeys(combined))  # Preserves order

    return {
        "raw_results": unique
    }