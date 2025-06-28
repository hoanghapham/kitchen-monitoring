import streamlit as st
import tempfile
import cv2

st.set_page_config("Upload video")
st.markdown("# Upload video")
uploaded_file = st.file_uploader(label="Video", type=[".mp4"])

st.markdown("# Preview")


if uploaded_file is not None:
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded_file.read())
    st.session_state.video_uploaded = True
    st.session_state.video_path = temp.name

    st.video(st.session_state.video_path)

    cap = cv2.VideoCapture(st.session_state.video_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_length_s = frame_count / fps
    spf = vid_length_s / frame_count
    st.session_state.frame_count = frame_count

    st.write(f"Total frames: {frame_count}, total length: {vid_length_s:.2f} seconds.")