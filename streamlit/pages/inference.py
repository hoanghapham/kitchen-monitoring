import streamlit as st
import cv2
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.set_page_config(page_title="Inference")


# State management
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
    st.session_state.frame_msec = 0.0   

if "video_uploaded" not in st.session_state:
    st.session_state.video_uploaded = False

if "paused" not in st.session_state:
    st.session_state.paused = False

def calc_frame_msec(frame_idx: int):
    return frame_idx * 1000 / fps

def play_pause_callback():
    if "paused" not in st.session_state:
        st.session_state.paused = True
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

@st.fragment
def display_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_display.image(frame_rgb, caption=f"Frame {st.session_state.frame_idx}", width=frame_rgb.shape[1] // 2)

@st.fragment
def annotate(frame):
    frame_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    with st.form(f"annotate"):
        if st.session_state.paused:
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=2,
                background_image=frame_img,
                drawing_mode="rect",
                update_streamlit=True,
                key="annotation",
                height=frame_img.height // 2,
                width=frame_img.width // 2,
            )
            submitted = st.form_submit_button("Confirm")


@st.fragment
def set_frame():
    cap.set(cv2.CAP_PROP_POS_MSEC, st.session_state.frame_msec)
    ret, frame = cap.read()
    return ret, frame


st.markdown("# Inference")

with st.container(height=650):
    image_display = st.empty()

play_frames = st.selectbox("Display every n frames:", options=[1, 5, 10, 20, 50, 100, 500], index=0)
skip_frames = st.selectbox("Forward/backward n frames", options=[1, 2, 5, 10, 50, 100], index=0)


lcol, rcol = st.columns([1, 1])

with lcol:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        prev = st.button("Prev", on_click=prev_callback)

    with col2:
        play_pause = st.button("Play/Pause", on_click=play_pause_callback)

    with col3:
        forward = st.button("Forward", on_click=forward_callback)

with rcol:
    st.markdown("# Annotate")

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
        ret, frame = set_frame()
        if frame is not None:
            st.session_state.last_frame = frame
        display_frame(frame)
        # st.session_state.frame_idx += play_frames
        # st.session_state.frame_msec = calc_frame_msec(st.session_state.frame_idx)
        # annotate(frame)

        # Play/pause logic
        if play_pause:
            while cap.isOpened():
                # progress_display.caption(st.session_state.frame_idx)

                ret, frame = cap.read()

                if st.session_state.paused is True:
                    display_frame(frame)
                    break

                if frame is not None:
                    display_frame(frame)
                    st.session_state.last_frame = frame
                    st.session_state.frame_idx += play_frames
                    st.session_state.frame_msec = calc_frame_msec(st.session_state.frame_idx)

                if not ret:
                    st.session_state.paused = True
                    st.session_state.frame_idx = 0
                    st.session_state.frame_msec = 0
                    break

                
        cap.release()
        
        annotate(st.session_state.last_frame)