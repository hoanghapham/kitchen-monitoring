# Kitchen Monitoring System

This project simulates a monitoring system for kitchens, which performs object detection and tracking for two types of items (dish, tray), and further classifies the items into one of the three sub-categories (empty, not_empty, kakigori - a type of Japanese shaved ice dessert).

Components:

- Backend: FastAPI
- UI: Gradio
- Video / Image processing: OpenCV
- Models:
    - Detector: `yolo11m`
    - Classifier: `yolo11m-cls`

Inference flow:

![](assets/inference_flow.png)

# Workflow

In reality the system should be able to handle video stream from camera, but for demo purpose, it process a video in an offline manner. 

After uploading the video and click **Detect Objects**, the video will be processed and an output video with bounding boxes & labels will be displayed on the right.

![](assets/upload.png)

The user can also choose to slice through the video and do annotation themselves to create new training data points.

![](assets/reannotate.png)


## Manual Setup & Running

The project can be installed using `uv` or `pip`. Here are the steps:

- Clone this project from the GitHub repo
- Create a virtual environment using `venv`, `pyenv`, or `conda`, and activate the environment.

- **(Recommended)** Install required packages with [uv project manager](https://docs.astral.sh/uv/#installation):

```bash
# Install uv (MacOS, Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install packages
uv sync --frozen

# Or, install the whole project in editable mode
uv pip install -e .
```

- Install with `pip`:

```bash
# Install packages
pip install -r requirements.txt

# Or, install the whole project in editable mode
pip install -e .
```

To run the app:

```bash
python run.py
```

The script with launch the application in a new browser tab.


## Setting up & Running with Docker

To set up and run the app using Docker, for the first time you will need to build the image:

```bash
docker compose -f docker/compose.yaml build
```

To run the app after building:

```bash
docker compose -f docker/compose.yaml up
```

These `docker compose` commands will do the following:
- Build the Docker images for the backend and frontend services
- Run the two services in isolation

On the first run you can also combine the two commands like so:

```bash
docker compose -f docker/compose.yaml up --build
```

When the app startup is done, it can be accesses via this address: 

`http://0.0.0.0:8000/gradio`


## Folder structure

- `data`:
    - `classification`: training data for the dish classifier and tray classifier to detect sub-classes
- `docker`: Containing Docker-related files to build and run the Docker image for the app.
- `models`: Containing `.pt` files of the trained YOLO models
- `scrips`: Containing scripts to train the models
- `src`:
    - `app`: The backend of the system, exposing a single API endpoint for prediction
    - `gradio_ui`: The app UI built with Gradio
    - `kitchen`: Containing functions to process images, videos and perform inference using the YOLO models
    - `utils`: Helper functions and schemas for input/output contents, to assist the communication between frontend and backend
