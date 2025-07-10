import numpy as np
import uuid
import gradio as gr
from pathlib import Path
from PIL import Image
from gradio_image_annotation import image_annotator

from kitchen.inference import get_video_frame
from kitchen.visual_tasks import bbox_xyxy_to_yolo_format
from utils.file_tools import write_json_file, write_list_to_text_file


PROJECT_DIR         = Path(__file__).parent.parent.parent
DATA_DIR            = PROJECT_DIR / "data/"


def display_first_frame(video_path: str):
    return get_video_frame(video_path, 1)


def get_bbox(annotations):
    return annotations["boxes"]


def get_annotator(ori_frame_disp):
    annotator = image_annotator(
        value={"image": ori_frame_disp},
        show_remove_button=True,
        show_clear_button=True
    )
    return annotator


def save_detection_ann(image: np.ndarray, json_ann: dict):
    out_ann_id = uuid.uuid4()
    json_dir = DATA_DIR / f"retrain/detection/json/"
    image_dir = DATA_DIR / f"retrain/detection/image"
    label_dir = DATA_DIR / f"retrain/detection/label"

    if not json_dir.exists():
        json_dir.mkdir(parents=True)

    if not image_dir.exists():
        image_dir.mkdir(parents=True)

    if not label_dir.exists():
        label_dir.mkdir(parents=True)

    pil_image = Image.fromarray(image)
    pil_image.save(image_dir / f"{out_ann_id}.png")
    write_json_file(json_ann, json_dir / f"{out_ann_id}.json")

    # Convert json to yolo format
    label2id = {
        "dish": 0,
        "tray": 1
    }
    bboxes = []
    for box in json_ann:
        object_label = box["label"].split("-")[0]
        bboxes.append(bbox_xyxy_to_yolo_format(
            (box["xmin"], box["ymin"], box["xmax"], box["ymax"]),
            pil_image.width,
            pil_image.height,
            label2id[object_label]
        ))

    write_list_to_text_file(bboxes, label_dir / f"{out_ann_id}.txt") 
    gr.Info(f"Saved image to 'retrain/detection'")


with gr.Blocks() as reannotate_block:
    gr.Markdown("Drag the slider to select a frame you want to reannotate, then draw bounding box and add labels in the lower image.")

    video_stats = gr.State()
    in_video = gr.Video(visible=False)
    out_video = gr.Video(visible=False)
    out_frame_disp = gr.Image(label="Annotated frame", interactive=False)
    ori_frame_disp = gr.Image(interactive=False, visible=False)

    # Only render the frame slider when video stats is changed
    @gr.render(inputs=[video_stats], triggers=[video_stats.change])
    def frame_slider(video_stats):
        frame_idx = gr.Slider(label="Frame", minimum=1, maximum=video_stats["n_frames"], step=1)  
        frame_idx.change(get_video_frame, inputs=[out_video, frame_idx], outputs=[out_frame_disp])
        frame_idx.change(get_video_frame, inputs=[in_video, frame_idx], outputs=[ori_frame_disp])
        frame_idx.change(get_annotator, [ori_frame_disp], [annotator])

    # Render an annotator with a blank placeholder image
    annotator = image_annotator(
        value={"image": np.ones((500, 1000, 3), dtype=np.uint8) * 255},
        show_remove_button=True,
        show_clear_button=True,
        label_list=[
            "dish-empty", 
            "dish-not_empty",
            "dish-kakigori",
            "tray-empty", 
            "tray-not_empty",
            "tray-kakigori",
        ],
    )

    with gr.Row():
        with gr.Column():
            get_bbox_btn = gr.Button("Get bounding boxes")
            save_detection_btn = gr.Button("Save detection data")
        with gr.Column():
            json_boxes = gr.JSON()
    
    get_bbox_btn.click(get_bbox, annotator, json_boxes)

    # Button to save image and annotated boxes
    save_detection_btn.click(
        save_detection_ann, 
        inputs=[ori_frame_disp, json_boxes],
    )

    # Data flow: When in_video/out_video is updated, display the first frame
    in_video.change(display_first_frame, [in_video], [ori_frame_disp])
    out_video.change(display_first_frame, [out_video], [out_frame_disp])
    