import gradio as gr

css = """
.gr-info {
    background-color: rgba(0, 255, 0, 1) !important;  /* solid white */
    color: black;
    border: 1px solid #ccc;
}
"""

def click_callback():
    gr.Info("This is a non-transparent info box.")

theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[
        gr.themes.GoogleFont("Open Sans"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
).set(
    background_fill_primary="white",
    error_background_fill="white"
)

with gr.Blocks(theme=theme) as demo:
    btn = gr.Button("click here")
    btn.click(click_callback)

demo.launch()