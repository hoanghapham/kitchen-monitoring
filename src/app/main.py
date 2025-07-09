import uvicorn
from fastapi import FastAPI
import torch
from ultralytics import YOLO
from pydantic import ValidationError

from kitchen.inference import process_video
from utils.schemas import PredictionInput, PredictionOutput


# Init model
detector        = YOLO("models/detector//train/weights/best.pt")
dish_classifier = YOLO("models/dish_classifier/train/weights/best.pt")
tray_classifier = YOLO("models/tray_classifier/train/weights/best.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


app = FastAPI()

@app.get("/")
def read_root():
    return {"app_name": "Kitchen Dispatch Monitoring"}

@app.post("/predict/")
def predict(content: dict):
    try:
        pred_input = PredictionInput(**content)
        result = process_video(
            pred_input["input_video"],
            pred_input["output_video"],
            detector, dish_classifier, tray_classifier,
            pred_input["conf"],
            pred_input["iou"],
            DEVICE
        )
        return result
    except ValidationError as e:
        print(e)
        return PredictionOutput(output_video="")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
