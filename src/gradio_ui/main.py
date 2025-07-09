import os
import uuid
import torch
import requests
import gradio as gr
import uvicorn
import numpy as np
from pathlib import Path
from gradio_image_annotation import image_annotator

from app.main import app
from kitchen.inference import get_video_stats, get_video_frame
from utils.schemas import PredictionInput


APP_URL             = "http://0.0.0.0:8000"
PROJECT_DIR         = Path(__file__).parent.parent.parent
GRADIO_CACHE_DIR    = ".gradio_cache"
GRADIO_CUSTOM_PATH  = "/gradio"
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["GRADIO_CACHE_DIR"]  = GRADIO_CACHE_DIR


def set_video_stats(video_path):
    stats = get_video_stats(video_path)
    stats["in_video_path"] = video_path
    return stats


def detect_objects(in_video, conf, iou, result_collection: list):
    progress = gr.Progress()

    if in_video is None:
        raise gr.Error("Please upload a video first.", duration=2)
    
    temp_out_video = str(PROJECT_DIR / f"cache/{uuid.uuid4()}.mp4")
    pred_input = PredictionInput(
        in_video_path=in_video,
        out_video_path=temp_out_video,
        conf=conf,
        iou=iou,
        device=DEVICE
    )

    progress(0.0, desc="Inferencing...")
    response = requests.post(
        APP_URL + "/predict", 
        json=pred_input.model_dump(), 
        timeout=300
    )
    progress(1, "Done")

    if response.status_code == 200:
        output_video = response.json()["output_video"]
        return result_collection + [output_video]
    else:
        raise gr.Error(f"API Error: {response.status_code}")


def to_reannotate_tab():
    return gr.Tabs(selected=1)


def activate(collection):
    return gr.update(interactive=collection is not None)


def update_out_video(collection):
    return collection[-1]


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

with gr.Blocks() as demo:
    result_collection = gr.State([])
    video_stats = gr.State({})

    with gr.Tabs(key="all_tabs") as tabs:
        with gr.Tab("Upload", key="upload", id=0) as tab_upload:
            with gr.Row():
                in_video = gr.Video(label="Upload video", sources="upload")
                out_video = gr.Video(label="Inference Result", interactive=False)

            with gr.Row():
                with gr.Column():
                    conf = gr.Slider(label="Confidence", minimum=0, maximum=1, value=0.25, interactive=True)
                    iou = gr.Slider(label="IoU", minimum=0, maximum=1, value=0.7, interactive=True)
                
                with gr.Column():
                    submit_btn = gr.Button(value="Detect Objects", interactive=False)
                    reannotate_btn = gr.Button(value="Reannotate", interactive=False)

        with gr.Tab("Reannotate", key="reannotate", id=1, interactive=False) as tab_reannotate:
            out_frame_disp = gr.Image(label="Annotated frame", interactive=False)
            ori_frame_disp = gr.Image(interactive=False, visible=False)

            # Only render the frame slider when video stats is changed
            @gr.render(inputs=[video_stats], triggers=[video_stats.change])
            def frame_slider(video_stats):
                frame_idx = gr.Slider(label="Frame", minimum=1, maximum=video_stats["n_frames"], step=1)  
                frame_idx.change(get_video_frame, inputs=[out_video, frame_idx], outputs=[out_frame_disp])
                frame_idx.change(get_video_frame, inputs=[in_video, frame_idx], outputs=[ori_frame_disp])
                frame_idx.change(get_annotator, [ori_frame_disp], [annotator])

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
                label_colors=[(0, 255, 0), (255, 0, 0)],
            )
            button_get = gr.Button("Get bounding boxes")
            json_boxes = gr.JSON()
            button_get.click(get_bbox, annotator, json_boxes)

            #TODO Add button to save image and annotated boxes
            

    # Dataflow
    # Activate the Detect Objects button when a video is uploaded
    in_video.change(activate, [in_video], [submit_btn])
    
    # Get and set video stats having results
    out_video.change(set_video_stats, inputs=[in_video], outputs=[video_stats])

    # Inference
    submit_btn.click(
        detect_objects,
        inputs=[in_video, conf, iou, result_collection],
        outputs=[result_collection],
        show_progress="full",
        show_progress_on=[out_video]
    )


    # When a result is available:
    # - Activate Reannnotate button
    # - Update out_video placeholder
    # - Display first frame in the frame placeholders in the Reannotate tab
    result_collection.change(activate, [result_collection], [reannotate_btn])
    result_collection.change(update_out_video, [result_collection], [out_video])
    
    in_video.change(display_first_frame, [in_video], [ori_frame_disp])
    out_video.change(display_first_frame, [out_video], [out_frame_disp])
    
    # Activate the Reannotate tab and move there
    reannotate_btn.click(activate, inputs=[result_collection], outputs=[tab_reannotate])
    reannotate_btn.click(to_reannotate_tab, [], [tabs])


# Mount the Gradio demo on top of base app
app = gr.mount_gradio_app(app, demo, path=GRADIO_CUSTOM_PATH)


if __name__ == "__main__":
    uvicorn.run("gradio_ui.main:app", host="0.0.0.0", port=8000)
    