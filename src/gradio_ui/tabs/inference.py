import os
import uuid
import torch
import requests
import gradio as gr
from pathlib import Path

from kitchen.inference import get_video_stats, get_video_frame
from utils.schemas import PredictionInput


APP_URL             = "http://0.0.0.0:8000"
PROJECT_DIR         = Path(__file__).parent.parent.parent.parent

GRADIO_CACHE_DIR    = ".gradio_cache"
GRADIO_CUSTOM_PATH  = "/gradio"
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR            = PROJECT_DIR / "data/"

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
    )
    progress(1, "Done")

    if response.status_code == 200:
        output_video = response.json()["output_video"]
        return result_collection + [output_video]
    else:
        raise gr.Error(f"API Error: {response.status_code}")


def activate(collection):
    return gr.update(interactive=collection is not None)

def deactivate():
    return gr.update(interactive=False)

def update_out_video(collection):
    return collection[-1]


def display_first_frame(video_path: str):
    return get_video_frame(video_path, 1)


def sync_gradio_object_state(input_value, target_state_value):
    target_state_value = input_value
    return target_state_value if target_state_value is not None else gr.skip()


with gr.Blocks() as inference_block:
    gr.Markdown("Upload a video to start")
    result_collection = gr.State([])
    video_stats = gr.State({})
    
    with gr.Row():
        in_video = gr.Video(label="Upload video", sources="upload")
        out_video = gr.Video(label="Inference Result", interactive=False, )

    with gr.Row():
        with gr.Column():
            conf = gr.Slider(label="Confidence", minimum=0, maximum=1, value=0.25, interactive=True)
            iou = gr.Slider(label="IoU", minimum=0, maximum=1, value=0.7, interactive=True)
        
        with gr.Column():
            submit_btn = gr.Button(value="Detect Objects", interactive=False)
            reannotate_btn = gr.Button(value="Reannotate", interactive=False)


    # Dataflow
    # Activate the Detect Objects button when a video is uploaded
    in_video.upload(activate, [in_video], [submit_btn])
    in_video.clear(deactivate, [], [submit_btn])
    # Get and set video stats having results
    out_video.change(set_video_stats, inputs=[in_video], outputs=[video_stats])

    # Click button -> deactivate btn and run task -> then activate again
    submit_btn.click(deactivate, [], [submit_btn]).then(
        detect_objects,
        inputs=[in_video, conf, iou, result_collection],
        outputs=[result_collection],
        show_progress="full",
        show_progress_on=[out_video]
    ).then(activate, [], [submit_btn])


    # When a result is available:
    # - Activate Reannnotate button
    # - Update out_video placeholder
    result_collection.change(activate, [result_collection], [reannotate_btn])
    result_collection.change(update_out_video, [result_collection], [out_video])
    