from pydantic import BaseModel


class PredictionInput(BaseModel):
    in_video_path: str
    out_video_path: str
    conf: float
    iou: float
    device: str


class PredictionOutput(BaseModel):
    output_video: str