# modules/geometry/homography_setup.py
# -*- coding: utf-8 -*-
"""
Homography Setup.

Allows users to define corresponding points on a layout image and 
the first frame of the video to compute a homography matrix.

@author: Tony-Luna
"""

import os
import cv2
import numpy as np
from typing import Optional

class HomographySetup:
    """
    Manages manual point selection for computing homography.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        usage = config['usage_parameters']
        homography = config['homography_parameters']

        self.video_path = usage['input_video_path']
        self.layout_image_path = homography['input_layout_image']

        self.output_dir = self._ensure_output_dir()
        self.h_matrix_path = os.path.join(self.output_dir, "h_matrix.npy")

        # Will be assigned upon loading
        self.layout_img = None
        self.first_frame = None

        self.points_layout = []
        self.points_frame = []

        self.padded_layout = None
        self.padded_frame = None

    def compute_homography_matrix(self) -> Optional[np.ndarray]:
        """
        Returns:
            np.ndarray or None: The homography matrix if computed, else None.
        """
        if os.path.exists(self.h_matrix_path):
            return np.load(self.h_matrix_path)

        self._load_images()
        self._prepare_images()
        concat_img = np.concatenate((self.padded_layout, self.padded_frame), axis=1)

        cv2.namedWindow("Homography Selection", cv2.WINDOW_NORMAL)
        cv2.imshow("Homography Selection", concat_img)
        max_w = self.padded_layout.shape[1]
        cv2.setMouseCallback("Homography Selection", self._on_mouse, (concat_img, max_w))

        print("\n[Homography Setup] Instructions:")
        print("- Click corresponding points on layout (left) and video (right).")
        print("- Press 'y' to compute homography.")
        print("- Press 'r' to remove last point pair.")
        print("- Press 'Esc' to exit without saving.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return None
            elif key == ord('y'):
                if len(self.points_layout) >= 4 and len(self.points_frame) >= 4:
                    H, _ = cv2.findHomography(np.array(self.points_frame), np.array(self.points_layout))
                    np.save(self.h_matrix_path, H)
                    cv2.destroyAllWindows()
                    return H
                else:
                    print("Not enough points to compute homography.")
            elif key == ord('r'):
                if self.points_layout and self.points_frame:
                    self.points_layout.pop()
                    self.points_frame.pop()
                    self._refresh_display(concat_img, max_w)

    def _ensure_output_dir(self) -> str:
        base_dir = self.config['usage_parameters']['output_base_dir']
        name = os.path.splitext(os.path.basename(self.video_path))[0]
        out_dir = os.path.join(base_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _load_images(self) -> None:
        self.layout_img = cv2.imread(self.layout_image_path)
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise IOError("Could not load first frame from video.")
        self.first_frame = frame

    def _prepare_images(self) -> None:
        max_h = max(self.layout_img.shape[0], self.first_frame.shape[0])
        max_w = max(self.layout_img.shape[1], self.first_frame.shape[1])

        self.padded_layout = cv2.copyMakeBorder(
            self.layout_img, 0, max_h - self.layout_img.shape[0],
            0, max_w - self.layout_img.shape[1],
            cv2.BORDER_CONSTANT, value=(0,0,0)
        )
        self.padded_frame = cv2.copyMakeBorder(
            self.first_frame, 0, max_h - self.first_frame.shape[0],
            0, max_w - self.first_frame.shape[1],
            cv2.BORDER_CONSTANT, value=(0,0,0)
        )

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            concat_img, layout_w = param
            if x < layout_w:
                self.points_layout.append((x, y))
            else:
                self.points_frame.append((x - layout_w, y))
            self._refresh_display(concat_img, layout_w)

    def _refresh_display(self, concat_img, layout_w: int) -> None:
        concat_img[:, :] = np.concatenate((self.padded_layout, self.padded_frame), axis=1)
        # Draw layout points (blue)
        for pt in self.points_layout:
            cv2.circle(concat_img, pt, 5, (255,0,0), -1)
        # Draw frame points (green)
        for pt in self.points_frame:
            cv2.circle(concat_img, (pt[0] + layout_w, pt[1]), 5, (0,255,0), -1)
        # Connect pairs with red line
        for pl, pf in zip(self.points_layout, self.points_frame):
            cv2.line(concat_img, pl, (pf[0] + layout_w, pf[1]), (0,0,255), 2)
        cv2.imshow("Homography Selection", concat_img)
