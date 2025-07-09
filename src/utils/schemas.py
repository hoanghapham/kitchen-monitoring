from pydantic import BaseModel


class PredictionInput(BaseModel):
    input_video: str
    conf: float
    iou: float


class PredictionOutput(BaseModel):
    output_video: str