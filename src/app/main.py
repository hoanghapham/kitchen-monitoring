import uvicorn
from fastapi import FastAPI
from ultralytics import YOLO
from pydantic import ValidationError
from pathlib import Path

from kitchen.inference import process_video
from utils.schemas import PredictionInput, PredictionOutput


# Init model
PROJECT_DIR     = Path(__file__).parent.parent.parent 
CACHE_DIR       = PROJECT_DIR / "cache"
detector        = YOLO(PROJECT_DIR / "models/detector/best.pt")
dish_classifier = YOLO(PROJECT_DIR / "models/dish_classifier/best.pt")
tray_classifier = YOLO(PROJECT_DIR / "models/tray_classifier/best.pt")

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TODO: Clear cache when starting up

if CACHE_DIR.exists():
    for file in CACHE_DIR.glob("*"):
        file.unlink()
else:
    CACHE_DIR.mkdir(parents=True)

app = FastAPI()


@app.get("/")
def read_root():
    return {"app_name": "Kitchen Monitoring"}


@app.post("/predict/", response_model=PredictionOutput)
def predict(content: dict):
    try:
        pred_input = PredictionInput(**content)
        result = process_video(
            pred_input.in_video_path,
            pred_input.out_video_path,
            detector, 
            dish_classifier, 
            tray_classifier,
            pred_input.conf,
            pred_input.iou,
            pred_input.device
        )
        return result
    except ValidationError as e:
        print(e)
        return PredictionOutput(output_video="")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)

