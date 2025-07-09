import numpy as np
from PIL import Image


# Conversion between normal format and YOLO

def bbox_xyxy_to_yolo_format(bbox: tuple | list, img_width: int, img_height: int, class_id=0) -> str:
    """Convert xyxy bbox to yolo string

    Parameters
    ----------
    bbox : tuple | list
    img_width : int
    img_height : int
    class_id : int, optional
        Class of the instance, by default 0

    Returns
    -------
    str
        Format: {class_id} {x_center} {y_center} {width} {height}
    """
    assert len(bbox) == 4, f"bbox has {len(bbox)} elements"
    (xmin, ymin, xmax, ymax) = bbox
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return f"{class_id} {x_center} {y_center} {width} {height}"


def bboxes_xyxy_to_yolo_format(bboxes: list[tuple], img_width: int, img_height: int, class_id=0) -> list[str]:
    """Accept a list of bboxes in xyxy format and convert to YOLO format:
        {class_id} {x_center} {y_center} {width} {height}

    Parameters
    ----------
    bboxes : list[tuple]
    img_width : int
    img_height : int
    class_id : int, optional
        Class of the object, by default 0

    Returns
    -------
    list[str]
        List of YOLO formatted annotations
    """
    yolo_annotations = []
    for bbox in bboxes:
        yolo_str = bbox_xyxy_to_yolo_format(bbox, img_width, img_height, class_id=class_id)
        yolo_annotations.append(yolo_str)
    return yolo_annotations


def yolo_to_bbox_xyxy(yolo_str: str, img_width: int, img_height: int) -> tuple:
    """Convert YOLO format to xyxy format"""
    class_id, x_center, y_center, width, height = yolo_str.split(" ")
    xmin = (float(x_center) - float(width) / 2) * img_width
    ymin = (float(y_center) - float(height) / 2) * img_height
    xmax = (float(x_center) + float(width) / 2) * img_width
    ymax = (float(y_center) + float(height) / 2) * img_height
    return (xmin, ymin, xmax, ymax)


def crop_image(img, bbox: list[tuple[int, int]]):
    """Crops an image based on the provided polygon coordinates. 
    Apply a white background for areas outside of the polygon.
    """
    image_array = np.array(img)

    points = np.array(bbox).astype(int)
    cropped = image_array[points[1]:points[3], points[0]:points[2], :]

    return Image.fromarray(cropped).convert("RGB")
