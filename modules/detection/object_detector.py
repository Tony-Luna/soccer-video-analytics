# modules/detection/object_detector.py
# -*- coding: utf-8 -*-
"""
Object Detector.

Implements YOLO-based object detection.

@author: Tony-Luna
"""

import numpy as np
from ultralytics import YOLO
from modules.common.models import BoundingBox
from typing import List

class ObjectDetector:
    """
    Wraps YOLO detection, returning bounding boxes and labels.
    """

    def __init__(self, model_path: str) -> None:
        """
        Args:
            model_path (str): Path to the YOLO model file.
        """
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray, conf: float = 0.7, imgsz: int = 640) -> List[tuple]:
        """
        Performs object detection on a single BGR frame.

        Args:
            frame (np.ndarray): BGR image.
            conf (float): Confidence threshold.
            imgsz (int): YOLO inference resolution.

        Returns:
            list of (BoundingBox, int): (bbox, label_id)
        """
        results = self.model.predict(frame, conf=conf, imgsz=imgsz)
        bboxes_labels = self._parse(results)
        return bboxes_labels

    def _parse(self, results) -> List[tuple]:
        out = []
        if not results or not results[0].boxes:
            return out

        boxes = results[0].boxes.xyxy
        labels = results[0].boxes.cls

        for lbl, xyxy in zip(labels, boxes):
            lbl_int = int(lbl.cpu().numpy())
            x1, y1, x2, y2 = xyxy.cpu().numpy()
            out.append((BoundingBox(int(x1), int(y1), int(x2), int(y2)), lbl_int))

        return out
