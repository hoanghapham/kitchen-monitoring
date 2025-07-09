import os
import uuid
import torch
import requests
import time
from pathlib import Path
import gradio as gr

from app.main import app
from kitchen.inference import get_video_stats, get_video_frame
from utils.schemas import PredictionInput


APP_URL             = "http://0.0.0.0:8000"
PROJECT_DIR         = Path(__file__).parent.parent.parent
GRADIO_CACHE_DIR    = ".gradio_cache"
GRADIO_CUSTOM_PATH  = "/gradio"
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["GRADIO_CACHE_DIR"]  = GRADIO_CACHE_DIR


def to_reannotate_tab():
    return gr.Tabs(selected=1)


def set_video_stats(video_path):
    stats = get_video_stats(video_path)
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


def activate(collection):
    return gr.update(interactive=collection is not None)


def update_video_container(collection):
    return collection[-1]


def display_first_frame(video_path: str):
    return get_video_frame(video_path, 0)


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
            out_frame_disp = gr.Image(label="Frame")

            @gr.render(inputs=[video_stats], triggers=[reannotate_btn.click])
            def frame_slider(video_stats):
                frame_idx = gr.Slider(label="Frame", minimum=1, maximum=video_stats["n_frames"], step=1)  
                frame_idx.change(get_video_frame, inputs=[out_video, frame_idx], outputs=[out_frame_disp])
        
    # App logic
    submit_btn.click(
        detect_objects,
        inputs=[in_video, conf, iou, result_collection],
        outputs=[result_collection],
        show_progress="full",
        show_progress_on=[out_video]
    )

    # Get & set video stats
    in_video.change(activate, [in_video], [submit_btn])
    submit_btn.click(set_video_stats, inputs=[in_video], outputs=[video_stats])
    
    result_collection.change(activate, [result_collection], [reannotate_btn])
    result_collection.change(update_video_container, [result_collection], [out_video])
    result_collection.change(display_first_frame, [in_video], [out_frame_disp])
    
    reannotate_btn.click(activate, inputs=[result_collection], outputs=[tab_reannotate])
    reannotate_btn.click(to_reannotate_tab, [], [tabs])


    # result_collection.change(show_reannotate_tab)

    


# Mount the Gradio demo on top of base app
app = gr.mount_gradio_app(app, demo, path=GRADIO_CUSTOM_PATH)

if __name__ == "__main__":
    demo.launch()