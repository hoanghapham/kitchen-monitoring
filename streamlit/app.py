import streamlit as st

st.set_page_config(page_title="Tracking System", layout="wide")

page_1 = st.Page("pages/upload.py", title="Upload")
page_2 = st.Page("pages/inference.py", title="Inference & Correction")

nav = st.navigation([page_1, page_2])

nav.run()