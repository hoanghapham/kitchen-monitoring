import cv2
import numpy as np
import subprocess
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from kitchen.visual_tasks import crop_image
from utils.schemas import PredictionOutput


temp_video = "cache/temp_video.mp4"


def process_video(
    input_video: str,
    output_video: str,
    detector: YOLO, 
    dish_classifier: YOLO, 
    tray_classifier: YOLO,
    conf: float = 0.25,
    iou: float = 0.7,
    device="cpu"
):
    cap     = cv2.VideoCapture(input_video)
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = int(cap.get(cv2.CAP_PROP_FPS))
    
    out     = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*"MP4V"), fps, (width, height))

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

                # Crop image using detected box to be fed into the classifiers
                cropped = crop_image(frame, bbox)

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

    # Convert cv2 format to x264
    subprocess.call(args=f"ffmpeg -y -i {temp_video} -c:v libx264 {output_video}".split(" "))
    return PredictionOutput(output_video=output_video)
