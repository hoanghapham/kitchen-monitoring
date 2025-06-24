import numpy as np
from PIL import Image as PILImage
from shapely.geometry import Polygon


IMAGE_EXTENSIONS = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tif",
            ".tiff",
            ".JPG",
            ".JPEG",
            ".PNG",
            ".GIF",
            ".BMP",
            ".TIF",
            ".TIFF",]


def bbox_xyxy_to_polygon(bbox: list[tuple]) -> Polygon:
    """Convert bbox in xyxy format to polygon in counter-clockwise order"""
    x1, y1, width, height = bbox_xyxy_to_xywh(bbox)

    # Order polygon points counter-clockwise
    x2 = x1 
    y2 = y1 + height

    x3 = x1 + width
    y3 = y1 + height

    x4 = x1 + width
    y4 = y1

    return Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])


def bbox_xyxy_to_xywh(bbox):
    """Convert bbox in xyxy format to xywh format"""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return x1, y1, w, h


def bbox_xywh_to_xyxy(bbox):
    """Convert bbox in xywh format to xyxy format"""
    x1, y1, width, height = bbox
    x2 = x1 + width
    y2 = y1 + height
    return x1, y1, x2, y2


def polygon_to_bbox_xyxy(polygon: Polygon | list[tuple[int, int]]):
    """Generate bbox in xyxy format from provided polygon"""
    if isinstance(polygon, Polygon):
        boundary = polygon.boundary.coords
    else:
        boundary = polygon

    x_coords = [tup[0] for tup in boundary]
    y_coords = [tup[1] for tup in boundary]

    x1 = min(x_coords)
    y1 = min(y_coords)
    x2 = max(x_coords)
    y2 = max(y_coords)
    return x1, y1, x2, y2



def crop_image(img, bbox: list[tuple[int, int]]):
    """Crops an image based on the provided polygon coordinates. 
    Apply a white background for areas outside of the polygon.
    """
    image_array = np.array(img)

    points = np.array(bbox).astype(int)
    cropped = image_array[points[1]:points[3], points[0]:points[2], :]

    return PILImage.fromarray(cropped).convert("RGB")

