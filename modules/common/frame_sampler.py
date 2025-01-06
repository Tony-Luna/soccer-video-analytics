# modules/common/frame_sampler.py
# -*- coding: utf-8 -*-
"""
Frame Sampler.

Provides functionality to sample random frames from a video.

@author: Tony-Luna
"""

import cv2
import random
from typing import List
import numpy as np
from tqdm import tqdm

def sample_random_frames(video_path: str, num_frames: int = 5) -> List[np.ndarray]:
    """
    Randomly samples up to `num_frames` frames from the given video.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): How many frames to sample.

    Returns:
        list of np.ndarray: A list of sampled frames (BGR).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_samples = min(num_frames, total_frames)
    chosen_indices = random.sample(range(total_frames), actual_samples)

    frames = []
    for idx in tqdm(sorted(chosen_indices), desc="Sampling frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    return frames
