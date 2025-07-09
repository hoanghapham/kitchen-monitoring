import gradio as gr
from PIL import Image
import time
import threading
# from main import process_video

# Assume this is provided
# def process_video(video_path): ...
#     return frame (PIL.Image), bboxes (List[Tuple[int, int, int, int]])

def process_video():
    return None

# Shared state
state = {
    "video_path": None,
    "is_playing": False,
    "current_frame_idx": 0,
    "skip_n": 1,
    "video_data": None,  # Assume process_video loads this
    "play_thread": None
}

def set_video_path(video):
    state["video_path"] = video
    return f"Video loaded: {video}"

def update_skip(value):
    state["skip_n"] = int(value)

def prev_frame():
    state["current_frame_idx"] = max(0, state["current_frame_idx"] - state["skip_n"])
    frame, _ = process_video(state["video_path"], state["current_frame_idx"])
    return frame

def play_video(frame_display):
    state["is_playing"] = not state["is_playing"]

    def run():
        while state["is_playing"]:
            frame, _ = process_video(state["video_path"], state["current_frame_idx"])
            frame_display.update(value=frame)
            state["current_frame_idx"] += state["skip_n"]
            time.sleep(0.1)

    if state["is_playing"]:
        # Launch thread
        state["play_thread"] = threading.Thread(target=run)
        state["play_thread"].start()

    return gr.Button.update(value="Pause" if state["is_playing"] else "Play")

# UI Definition

with gr.Blocks("Inference") as inference:
    video_path_status = gr.Textbox(label="Status", interactive=False)
    load_button = gr.Button("Load Video for Inference")
    frame_display = gr.Image(label="Frame Display")

    # Data flow:
    # Display the first frame
    # If play is true, while cap.isOpened(), read frame, and display frame.
    # At the final frame, set frame_idx, frame_msecs, and set paused to True
    # Read the state of the play/pause button. If paused, break while
    # While displaying, constantly update lastfarme, frame_idx, frame_msecs

    with gr.Row():
        prev_btn = gr.Button("Prev")
        play_pause_btn = gr.Button("Play")
        skip_dropdown = gr.Dropdown(
            choices=["1", "2", "5", "10", "20"],
            value="1",
            label="Frame skip")

    # load_button.click(fn=set_video_path, inputs=video_input, outputs=video_path_status)
    skip_dropdown.change(fn=update_skip, inputs=skip_dropdown)

    prev_btn.click(fn=prev_frame, outputs=frame_display)
    play_pause_btn.click(fn=play_video, inputs=frame_display, outputs=play_pause_btn)
