import os
import torch
import gradio as gr
import uvicorn
import webbrowser
import time
from pathlib import Path
from urllib.parse import urljoin

from app.main import app

from gradio_ui.tabs.inference import inference_block, reannotate_btn, activate
from gradio_ui.tabs.inference import in_video as infr_in_video
from gradio_ui.tabs.inference import out_video as infr_out_video
from gradio_ui.tabs.inference import video_stats as infr_video_stats
from gradio_ui.tabs.inference import result_collection as infr_result_collection

from gradio_ui.tabs.reannotate import reannotate_block
from gradio_ui.tabs.reannotate import video_stats as reann_video_stats
from gradio_ui.tabs.reannotate import in_video as reann_in_video
from gradio_ui.tabs.reannotate import out_video as reann_out_video

APP_URL             = "http://0.0.0.0:8000"
PROJECT_DIR         = Path(__file__).parent.parent.parent
GRADIO_CACHE_DIR    = ".gradio_cache"
GRADIO_CUSTOM_PATH  = "/gradio"
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR            = PROJECT_DIR / "data/"
GRADIO_URL          = urljoin("http://0.0.0.0:8000/", GRADIO_CUSTOM_PATH)

os.environ["GRADIO_CACHE_DIR"]  = GRADIO_CACHE_DIR


def open_in_browser():
    time.sleep(1)  # Wait a bit for server to start
    webbrowser.open(GRADIO_URL)


def start_app():
    uvicorn.run("gradio_ui.main:app", host="0.0.0.0", port=8000)


def to_reannotate_tab():
    return gr.Tabs(selected=1)


def sync_gradio_object_state(input_value, target_state_value):
    target_state_value = input_value
    return target_state_value if target_state_value is not None else gr.skip()


theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
    # font=[
    #     gr.themes.GoogleFont("Open Sans"),
    #     "ui-sans-serif",
    #     "system-ui",
    #     "sans-serif",
    # ],
)

content = """Detect dish and trays, and classify dish and trays into three sub-categories (empty, not_empty, kakigori).
After inference, user can also choose to reannotate some frames to create new training data.
"""

with gr.Blocks(
    title="Dispatch Monitoring",
    theme=theme
) as demo:
    
    gr.Markdown("# Dispatch Monitoring")
    gr.Markdown(value=content)

    with gr.Tabs(key="all_tabs") as tabs:
        with gr.Tab("Upload", key="upload", id=0) as tab_inference:
            inference_block.render()

        with gr.Tab("Reannotate", key="reannotate", id=1, interactive=False) as tab_reannotate:
            reannotate_block.render()
        
    
    # Button to activate the Reannotate tab and move there
    reannotate_btn.click(activate, inputs=[infr_result_collection], outputs=[tab_reannotate])
    reannotate_btn.click(to_reannotate_tab, [], [tabs])

    # Sync objects between the main app and the reannotate tab
    infr_in_video.change(
        sync_gradio_object_state,
        inputs=[infr_in_video, reann_in_video],
        outputs=[reann_in_video]
    )

    infr_out_video.change(
        sync_gradio_object_state,
        inputs=[infr_out_video, reann_out_video],
        outputs=[reann_out_video]
    )

    infr_video_stats.change(
        sync_gradio_object_state,
        inputs=[infr_video_stats, reann_video_stats],
        outputs=[reann_video_stats]
    )

# Mount the Gradio demo on top of base app
app = gr.mount_gradio_app(app, demo, path=GRADIO_CUSTOM_PATH)


if __name__ == "__main__":
    print(f"\nTo view the app, go to: {GRADIO_URL}\n")
    start_app()