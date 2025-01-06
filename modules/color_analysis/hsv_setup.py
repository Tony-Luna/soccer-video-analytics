# modules/color_analysis/hsv_setup.py
# -*- coding: utf-8 -*-
"""
HSV Setup.

High-level class to manage HSV color setup for multiple classes (teams).

@author: Tony-Luna
"""

import os
import cv2
import json
import numpy as np
import string
import sys

from typing import Dict
from modules.color_analysis.hsv_segmenter import HSVSegmenter
from modules.common.frame_sampler import sample_random_frames

class HSVRangeSetup:
    """
    Manages the process of defining HSV ranges for multiple team classes.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        usage = config['usage_parameters']
        self.n_classes = usage['n_classes']
        self.video_path = usage['input_video_path']
        self.output_dir = self._ensure_output_dir()
        self.hsv_path = os.path.join(self.output_dir, "hsv_ranges.json")

    def setup_hsv_ranges(self) -> Dict[str, dict]:
        """
        If existing HSV file is found, load it. Otherwise, interactively define new ranges.
        
        Returns:
            dict: A dict of unique_id -> { lower_bound, upper_bound, bgr_color, team_letter }
        """
        if os.path.exists(self.hsv_path):
            return self._load_ranges()

        # Sample frames from video
        frames = sample_random_frames(self.video_path, num_frames=5)
        # Combine them into one large mosaic
        mosaic = self._create_mosaic(frames)

        segmenter = HSVSegmenter("HSV Segmenter")
        hsv_dict = {}
        letters = iter(string.ascii_lowercase)

        print("\n[HSV Setup] Instructions:")
        print("- Press 'y' when satisfied with the current HSV range (to move to next class).")
        print("- Press 'Esc' to exit.")

        for _ in range(self.n_classes):
            while True:
                lower, upper = segmenter.update_segmentation(mosaic)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('y'):
                    team_letter = next(letters, None)
                    if not team_letter:
                        raise ValueError("Too many classes, out of letters.")
                    # Generate unique key
                    unique_id = f"{lower[0]}_{lower[1]}_{lower[2]}__{upper[0]}_{upper[1]}_{upper[2]}"

                    # Middle hue => approximate color
                    mid_hue = int((lower[0] + upper[0]) / 2)
                    mid_hsv = np.array([[[mid_hue, 255, 255]]], dtype=np.uint8)
                    mid_bgr = cv2.cvtColor(mid_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()

                    hsv_dict[unique_id] = {
                        'lower_bound': lower.tolist(),
                        'upper_bound': upper.tolist(),
                        'bgr_color': mid_bgr,
                        'team_letter': team_letter
                    }
                    break
                elif key == 27:  # ESC
                    cv2.destroyAllWindows()
                    sys.exit(0)

            segmenter.reset_trackbars()

        cv2.destroyAllWindows()
        self._save_ranges(hsv_dict)
        return self._load_ranges()

    def _ensure_output_dir(self) -> str:
        base_dir = self.config['usage_parameters']['output_base_dir']
        name = os.path.splitext(os.path.basename(self.video_path))[0]
        out_dir = os.path.join(base_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _create_mosaic(self, frames):
        """
        Creates a mosaic from a list of frames with a fixed number of columns 
        (max_columns). If the last row doesn't have enough frames to match 
        max_columns, we pad it with black images so that each row has the 
        same total width.
        """
        if not frames:
            return np.zeros((300, 400, 3), dtype=np.uint8)
    
        max_columns = 3
        thumbnail_w = 200
        thumbnail_h = 150
    
        # Calculate how many total rows needed
        import math
        num_rows = math.ceil(len(frames) / max_columns)
    
        # Pad frames if the last row is not complete
        remainder = len(frames) % max_columns
        if remainder != 0:
            needed = max_columns - remainder
            # Create dummy black frames to pad
            dummy_frame = np.zeros((thumbnail_h, thumbnail_w, 3), dtype=np.uint8)
            frames += [dummy_frame] * needed
    
        # Now frames is guaranteed to be multiple of max_columns
        row_images = []
        for row_idx in range(num_rows):
            # Extract the frames for this row
            row_start = row_idx * max_columns
            row_end   = row_start + max_columns
            row_frames = frames[row_start:row_end]
    
            # Resize each frame to consistent shape
            col_images = [cv2.resize(f, (thumbnail_w, thumbnail_h)) for f in row_frames]
            # Stack horizontally
            row_img = np.hstack(col_images)
            row_images.append(row_img)
    
        # Finally, stack all rows vertically
        mosaic = np.vstack(row_images)
        return mosaic


    def _save_ranges(self, data: dict) -> None:
        with open(self.hsv_path, 'w') as f:
            json.dump(data, f, indent=4)

    def _load_ranges(self) -> dict:
        with open(self.hsv_path, 'r') as f:
            data = json.load(f)
        # Convert to numpy arrays
        for k, v in data.items():
            v['lower_bound'] = np.array(v['lower_bound'])
            v['upper_bound'] = np.array(v['upper_bound'])
            v['bgr_color'] = tuple(map(int, v['bgr_color']))
        return data
