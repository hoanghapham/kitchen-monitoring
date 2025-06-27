import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from kitchen.visual_tasks import crop_image


def init_video_in_out(input_video, output_video):
    cap     = cv2.VideoCapture(input_video)
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = int(cap.get(cv2.CAP_PROP_FPS))
    out     = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"MP4V"), fps, (width, height))

    return cap, out, width, height


def process_video(
        cap: cv2.VideoCapture, 
        out: cv2.VideoWriter, 
        width: int,
        height: int,
        detector: YOLO, 
        dish_classifier: YOLO, 
        tray_classifier: YOLO,
        conf: float = 0.25,
        iou: float = 0.7,
        device="cpu"
    ):
    track_history = defaultdict(lambda: [])

    # Iterate through the frames in the video
    while True:
        has_frame, frame = cap.read()

        if not has_frame:
            print("Done")
            break
        
        # Initate an annotator to draw ion the frame
        annotator           = Annotator(frame, line_width=2, font_size=20)
        tracking_results    = detector.track(
            frame, 
            persist=True, 
            conf=conf, 
            iou=iou, 
            imgsz=(width, height),
            device=device
        )

        if tracking_results[0].boxes.is_track and tracking_results[0].boxes is not None:
            boxes       = tracking_results[0].boxes.xyxy
            classes     = tracking_results[0].boxes.cls.int()
            names       = [detector.names[i.item()] for i in  classes]
            track_ids   = tracking_results[0].boxes.id.int().cpu().tolist()

            # Iterate through tracking results, classify dish and tray
            for bbox, cls, name, track_id in zip(boxes, classes, names, track_ids):
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                
                # Track center of the box
                track_history[track_id].append((center_x, center_y))
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)

                cropped = crop_image(frame, bbox)
                # cropped = cropped.resize((int(cropped.width * 0.5), cropped.height))

                if cls == 0:    # dish
                    subclass = dish_classifier(cropped, device=device)
                    subclass_name = dish_classifier.names[subclass[0].probs.top1]
                elif cls == 1:  # tray
                    subclass == tray_classifier(cropped, device=device)
                    subclass_name = tray_classifier.names[subclass[0].probs.top1]
                
                # Draw bbox
                track_color = colors(int(track_id), True)
                annotator.box_label(box=bbox, color=track_color, label=f"{name}-{subclass_name}")

                # Draw tracking line
                points = np.hstack(track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=track_color, thickness=5)

        out.write(frame)
    
    out.release()
    cap.release()

if __name__ == "__main__":

    PROJECT_DIR = Path(__file__).parent.parent

    parser = ArgumentParser()
    parser.add_argument("--input-video", "-i", type=str, required=True)
    parser.add_argument("--output-video", "-o", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cpu")


    args            = parser.parse_args()
    INPUT_VIDEO     = Path(args.input_video)
    OUTPUT_VIDEO    = Path(args.output_video)
    CONF            = float(args.conf)
    IOU             = float(args.iou)
    DEVICE          = args.device

    if OUTPUT_VIDEO.exists():
        overwrite = input("Output video already exists. Overwrite? (y/n) ")
        if overwrite.lower() != "y":
            exit()
        else:
            OUTPUT_VIDEO.unlink()

    detector        = YOLO("models/detector//train/weights/best.pt")
    dish_classifier = YOLO("models/dish_classifier/train/weights/best.pt")
    tray_classifier = YOLO("models/tray_classifier/train/weights/best.pt")

    cap, out, width, height = init_video_in_out(INPUT_VIDEO, OUTPUT_VIDEO)
    process_video(
        cap, out, 
        width, height, 
        detector, dish_classifier, tray_classifier,
        CONF,
        IOU,
        DEVICE
    )
