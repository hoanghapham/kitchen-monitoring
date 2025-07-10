import multiprocessing
from src.gradio_ui.main import open_in_browser, start_app


if __name__ == "__main__":
    multiprocessing.Process(target=open_in_browser).start()
    start_app()