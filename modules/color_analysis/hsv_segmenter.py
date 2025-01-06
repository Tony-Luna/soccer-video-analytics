# modules/color_analysis/hsv_segmenter.py
# -*- coding: utf-8 -*-
"""
HSV Segmenter.

Provides a class for interactive HSV segmentation using OpenCV trackbars.
"""

import cv2
import numpy as np

class HSVSegmenter:
    """
    Interactive tool for adjusting HSV thresholds on an image via OpenCV trackbars.
    """

    def __init__(self, window_name: str = "HSV Segmentation") -> None:
        self.window_name = window_name
        cv2.namedWindow(self.window_name)
        self._create_trackbars()
        self.lower_bound = np.array([0, 0, 0])
        self.upper_bound = np.array([179, 255, 255])

    def _create_trackbars(self) -> None:
        cv2.createTrackbar("H Min", self.window_name, 0, 179, lambda x: None)
        cv2.createTrackbar("H Max", self.window_name, 179, 179, lambda x: None)
        cv2.createTrackbar("S Min", self.window_name, 0, 255, lambda x: None)
        cv2.createTrackbar("S Max", self.window_name, 255, 255, lambda x: None)
        cv2.createTrackbar("V Min", self.window_name, 0, 255, lambda x: None)
        cv2.createTrackbar("V Max", self.window_name, 255, 255, lambda x: None)

    def update_segmentation(self, image: np.ndarray) -> tuple:
        """
        Applies trackbar values to segment the given image in real-time.

        Args:
            image (np.ndarray): The BGR image to segment.

        Returns:
            (lower_bound, upper_bound): The updated HSV range.
        """
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.lower_bound = np.array([
            cv2.getTrackbarPos("H Min", self.window_name),
            cv2.getTrackbarPos("S Min", self.window_name),
            cv2.getTrackbarPos("V Min", self.window_name)
        ])
        self.upper_bound = np.array([
            cv2.getTrackbarPos("H Max", self.window_name),
            cv2.getTrackbarPos("S Max", self.window_name),
            cv2.getTrackbarPos("V Max", self.window_name)
        ])

        mask = cv2.inRange(hsv_img, self.lower_bound, self.upper_bound)
        segmented = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow(self.window_name, segmented)

        return (self.lower_bound, self.upper_bound)

    def reset_trackbars(self) -> None:
        """Resets all trackbars to default min/max values."""
        cv2.setTrackbarPos("H Min", self.window_name, 0)
        cv2.setTrackbarPos("H Max", self.window_name, 179)
        cv2.setTrackbarPos("S Min", self.window_name, 0)
        cv2.setTrackbarPos("S Max", self.window_name, 255)
        cv2.setTrackbarPos("V Min", self.window_name, 0)
        cv2.setTrackbarPos("V Max", self.window_name, 255)
