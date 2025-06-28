import streamlit as st
import cv2
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Inference")


# State management
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
    st.session_state.frame_msec = 0.0   

if "video_uploaded" not in st.session_state:
    st.session_state.video_uploaded = False

def calc_frame_msec(frame_idx: int):
    return frame_idx * 1000 / fps

def play_pause_callback():
    if "paused" not in st.session_state:
        st.session_state.paused = False
    else:
        if st.session_state.paused is False:
            st.session_state.paused = True
        else:
            st.session_state.paused = False


def prev_callback():
    st.session_state.frame_idx = max(0, st.session_state.frame_idx - skip_frames)
    st.session_state.frame_msec = calc_frame_msec(st.session_state.frame_idx)


def forward_callback():
    st.session_state.paused = True
    # st.session_state.forward = True
    st.session_state.frame_idx = min(st.session_state.frame_count - 1, st.session_state.frame_idx + skip_frames)
    st.session_state.frame_msec = calc_frame_msec(st.session_state.frame_idx)


def display_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_display.image(frame_rgb, caption=f"Frame {st.session_state.frame_idx}")



st.markdown("# Inference")
play_frames = st.selectbox("Display every n frames:", options=[1, 5, 10, 20, 50, 100, 500], index=0)
skip_frames = st.selectbox("Forward/backward n frames", options=[1, 2, 5, 10, 50, 100], index=0)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    prev = st.button("Prev", on_click=prev_callback)

with col2:
    play_pause = st.button("Play/Pause", on_click=play_pause_callback)

with col3:
    forward = st.button("Forward", on_click=forward_callback)

image_display = st.empty()


# Display video
if st.session_state.video_uploaded:

    # Read info
    cap = cv2.VideoCapture(st.session_state.video_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_length_s = frame_count / fps
    spf = vid_length_s / frame_count
    st.session_state.frame_count = frame_count

    
    # Display image
    cap.set(cv2.CAP_PROP_POS_MSEC, st.session_state.frame_msec)
    ret, frame = cap.read()
    display_frame(frame)
    st.session_state.frame_idx += play_frames
    st.session_state.frame_msec = calc_frame_msec(st.session_state.frame_idx)

    # Play/pause logic
    if play_pause:

        while cap.isOpened():
            ret, frame = cap.read()

            if frame is not None:
                display_frame(frame)

            if not ret:
                st.session_state.paused = True
                st.session_state.frame_idx = 0
                st.session_state.frame_msec = 0
                break
            
            st.session_state.frame_idx += play_frames
            st.session_state.frame_msec = calc_frame_msec(st.session_state.frame_idx)

            if st.session_state.paused is True:
                display_frame(frame)
                break

    cap.release()