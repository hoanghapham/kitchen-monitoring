import os
import uuid
import torch
import requests
import time
from pathlib import Path
import gradio as gr

from app.main import app
from utils.schemas import PredictionInput


APP_URL             = "http://0.0.0.0:8000"
PROJECT_DIR         = Path(__file__).parent.parent.parent
GRADIO_CACHE_DIR    = ".gradio_cache"
GRADIO_CUSTOM_PATH  = "/gradio"
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["GRADIO_CACHE_DIR"]  = GRADIO_CACHE_DIR


def detect_objects(in_video, conf, iou):
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

    if response.status_code == 200:
        progress(1.0, "Done")
        output_video = response.json()["output_video"]
        return output_video
    else:
        raise gr.Error(f"API Error: {response.status_code}")
    

with gr.Blocks() as demo:
    with gr.Tab("Upload", key="upload"):
        in_video = gr.Video(label="Upload video", sources="upload")

        with gr.Row():
            conf = gr.Slider(label="Confidence", minimum=0, maximum=1, value=0.25, interactive=True)
            iou = gr.Slider(label="IoU", minimum=0, maximum=1, value=0.7, interactive=True)
        
        submit_btn = gr.Button(value="Detect Objects")

    with gr.Tab("Inference", key="inference"):
        out_video_container = gr.Video(label="Inference Result")

    # TODO: Add a tab for user feedback:
    # - Process output video as images with drawn bbox
    # - Display input video as clean images, on top of a drawable canvas

    # App logic
    submit_btn.click(
        detect_objects,
        inputs=[in_video, conf, iou],
        outputs=[out_video_container],
        show_progress="full",
        show_progress_on=[in_video]
    )

    # Switch tab when finishing
    gr.Info("Inference done, switching tab...")
    time.sleep(1)
    out_video_container.change(lambda: gr.Tabs(selected="inference"))


# Mount the Gradio demo on top of base app
app = gr.mount_gradio_app(app, demo, path=GRADIO_CUSTOM_PATH)

if __name__ == "__main__":
    demo.launch()