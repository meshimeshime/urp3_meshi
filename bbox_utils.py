import cv2
import numpy as np

def mask_to_bbox(mask):
    """
    binary mask (0/255) -> bounding box 좌표 (x1, y1, x2, y2)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return x, y, x + w, y + h

def draw_bbox(img, bbox, color=(0, 255, 0), thickness=2):
    if bbox is None:
        return img
    x1, y1, x2, y2 = bbox
    return cv2.rectangle(img.copy(), (x1, y1), (x2, y2), color, thickness)
