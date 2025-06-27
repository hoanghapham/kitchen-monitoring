from pathlib import Path
from argparse import ArgumentParser
from ultralytics import YOLO
import torch

if __name__ == "__main__":

    PROJECT_DIR = Path(__file__).parent.parent

    parser = ArgumentParser()
    parser.add_argument("--base-model", type=str, required=True, help="Name of base model, for example: yolov11m.pt, yolo11m-cls.pt")
    parser.add_argument("--data", "-d", type=str, required=True)
    parser.add_argument("--image-size", "-imgsz", type=int, required=True)
    parser.add_argument("--out-model-dir", "-o", type=str, required=True)
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--save", "-s", default="false", choices=["true", "false"])
    parser.add_argument("--save-period", "-sp", type=int, default=1)
    parser.add_argument("--batch", "-b", type=int, default=10)

    args            = parser.parse_args()
    BASE_MODEL      = args.base_model
    DATA            = Path(args.data)
    IMAGE_SIZE      = args.image_size
    OUT_MODEL_DIR   = Path(args.out_model_dir)
    EPOCHS          = int(args.epochs)
    BATCH           = int(args.batch)
    SAVE            = args.save == "true"
    SAVE_PERIOD     = int(args.save_period)
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLO(f"./models/base/{BASE_MODEL}")

    model.train(
        data=DATA, 
        imgsz=IMAGE_SIZE,
        epochs=EPOCHS, 
        batch=BATCH,
        save=SAVE,
        project=OUT_MODEL_DIR,
        save_period=SAVE_PERIOD,
        device=DEVICE
    )