import cv2
from pathlib import Path
from argparse import ArgumentParser
from ultralytics import YOLO
from PIL import Image
from ultralytics.utils.plotting import Annotator, colors
from kitchen.visual_tasks import crop_image


def init_video_in_out(input_video, output_video):
    cap     = cv2.VideoCapture(input_video)
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = int(cap.get(cv2.CAP_PROP_FPS))
    out     = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"h264"), fps, (width, height))

    return cap, out, width, height


def process_video(
        cap: cv2.VideoCapture, 
        out: cv2.VideoWriter, 
        width: int,
        height: int,
        detector: YOLO, 
        dish_classifier: YOLO, 
        tray_classifier: YOLO
    ):

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
            conf=0.1, 
            iou=0.7, 
            imgsz=(width, height)
        )

        if tracking_results[0].boxes.is_track and tracking_results[0].boxes is not None:
            boxes       = tracking_results[0].boxes.xyxy
            classes     = tracking_results[0].boxes.cls.int()
            names       = [detector.names[i.item()] for i in  classes]
            track_ids   = tracking_results[0].boxes.id.int().cpu().tolist()

            # Iterate through tracking results, classify dish and tray
            for bbox, cls, name, track_id in zip(boxes, classes, names, track_ids):
                cropped = crop_image(frame, bbox)
                cropped = cropped.resize((int(cropped.width * 0.5), cropped.height))

                if cls == 0:    # dish
                    subclass = dish_classifier(cropped)
                    subclass_name = dish_classifier.names[subclass[0].probs.top1]
                elif cls == 1:  # tray
                    subclass == tray_classifier(cropped)
                    subclass_name = tray_classifier.names[subclass[0].probs.top1]
                
                annotator.box_label(box=bbox, color=colors(int(track_id), True), label=f"{name}-{track_id}-{subclass_name}")

        out.write(frame)
    
    out.release()
    cap.release()

if __name__ == "__main__":

    PROJECT_DIR = Path(__file__).parent.parent

    parser = ArgumentParser()
    parser.add_argument("--input-video", "-i", type=str, required=True)
    parser.add_argument("--output-video", "-o", type=str, required=True)

    args            = parser.parse_args()
    INPUT_VIDEO     = Path(args.input_video)
    OUTPUT_VIDEO    = Path(args.output_video)

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
    process_video(cap, out, width, height, detector, dish_classifier, tray_classifier)
