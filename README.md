Dispatch Monitoring System
- Develop an intelligent monitoring system for a commercial kitchen's dispatch area. 
- Using the provided video and dataset (https://drive.google.com/drive/folders/1Ji4RVZveHcBkLngfgH0eexLIWiUp2c78?usp=drive_link), 
- Build a complete solution capable of tracking items within the dispatch area. 
- The system should also include functionality to improve model performance based on user feedback

python main.py --input-video data/first.mp4 --output-video data/tracked.mp4 --conf 0.7 --iou 0.7 --device cuda

User feedback: an annotation module?

Must have streaming / online mode

Flask/fastapi for backend
react for UI

or streamlit, gradio


UI Flow:
- Upload video
- Preview source video
- Inference:
    - Play all video frame by frame (have a dropdown to control how many frames to jump)
        - Has a play/pause button
        - Play video from the current_frame_idx
    - Has a prev button: go back to x frames (has a dropdown to control)
    - Has a Next button: Advance 
- Annotation:
    - Missing items: 
        - can draw new bbox and add new label
        - Can save result to a pair of image - label, image is a frame
    - Flag false positive
        