# modules/domain/objects_entities.py
# -*- coding: utf-8 -*-
"""
Domain Entities.

Defines classes for representing players (TeamPlayer) and the ball (Ball).

@author: Tony-Luna
"""

import cv2
import numpy as np
from typing import List, Tuple
from modules.common.models import BoundingBox

class TeamPlayer:
    """
    Represents a player belonging to a specific HSV-based class/team.
    """

    def __init__(self, config: dict, team_id: str, teams_dict: dict) -> None:
        usage = config['usage_parameters']
        self.labels_of_interest = usage['player_labels']
        self.team_id = team_id
        self.team_letter = teams_dict[team_id]['team_letter']
        self.color = teams_dict[team_id]['bgr_color']
        self.max_bbox: BoundingBox = None
        self.center_point: Tuple[int, int] = None

        # Collect all HSV ranges from the entire dictionary
        self.hsv_ids, self.hsv_ranges = self._collect_hsv_data(teams_dict)

    def _collect_hsv_data(self, teams_dict):
        hsv_ids = []
        hsv_ranges = []
        for k, v in teams_dict.items():
            if 'lower_bound' in v and 'upper_bound' in v:
                hsv_ids.append(k)
                hsv_ranges.append((v['lower_bound'], v['upper_bound']))
        return hsv_ids, hsv_ranges

    def update_draw_location(self, detections: List[tuple], frame: np.ndarray) -> None:
        """
        Filters detections by label, checks color classification, 
        picks the largest bounding box.
        Args:
            detections (List[tuple]): [(BoundingBox, label_id), ...]
            frame (np.ndarray): The current video frame.
        """
        candidate_boxes = [(bbox, lbl) for (bbox, lbl) in detections if lbl in self.labels_of_interest]
        same_team_boxes = [bbox for (bbox, _) in candidate_boxes if self._is_team_box(bbox, frame)]

        if same_team_boxes:
            # Pick largest bounding box by area
            areas = [b.width * b.height for b in same_team_boxes]
            idx_max = int(np.argmax(areas))
            self.max_bbox = same_team_boxes[idx_max]
            cx, cy = self.max_bbox.center_bottom()
            self.center_point = (cx, cy)
            self._draw(frame)
        else:
            self.max_bbox = None
            self.center_point = None

    def _is_team_box(self, bbox: BoundingBox, frame: np.ndarray) -> bool:
        """
        Classifies the bounding box by HSV color.
        Returns True if it matches self.team_id, else False.
        """
        crop = frame[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
        predicted_id = self._classify_color(crop)
        return predicted_id == self.team_id

    def _classify_color(self, crop: np.ndarray) -> str:
        """Applies multiple HSV ranges; picks the best via mask sums."""
        hsv_img = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        scores = []
        for (low, high) in self.hsv_ranges:
            mask = cv2.inRange(hsv_img, low, high)
            scores.append(np.sum(mask))
        best_idx = int(np.argmax(scores))
        return self.hsv_ids[best_idx]

    def _draw(self, frame: np.ndarray) -> None:
        """Draw bounding box and label on frame."""
        color = self.color
        if self.max_bbox:
            x1, y1, x2, y2 = self.max_bbox.x1, self.max_bbox.y1, self.max_bbox.x2, self.max_bbox.y2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, self.center_point, 5, color, -1)
            label_str = f"Team {self.team_letter.upper()}"
            (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label_str, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)


class Ball:
    """
    Represents the ball. Tracks last possession by checking overlap 
    with player bounding boxes.
    """

    def __init__(self, config: dict):
        usage = config['usage_parameters']
        self.labels_of_interest = usage['ball_labels']
        self.color = (255,255,255)
        self.center_point: Tuple[int, int] = None
        self.last_team_id: str = None
        self.last_team_letter: str = None

    def update_draw_location(self, players: List[TeamPlayer], detections: List[tuple], frame: np.ndarray) -> None:
        """
        Finds the largest ball bounding box, updates center, checks for possession.
        """
        ball_boxes = [(bbox, lbl) for (bbox, lbl) in detections if lbl in self.labels_of_interest]
        if not ball_boxes:
            self.center_point = None
            return

        # Pick largest by area
        areas = [(b.width * b.height) for (b, _) in ball_boxes]
        idx_max = int(np.argmax(areas))
        max_bbox = ball_boxes[idx_max][0]
        cx, cy = max_bbox.center_bottom()
        self.center_point = (cx, cy)

        # Check if near bottom of a player's bounding box => possession
        for pl in players:
            if pl.max_bbox:
                if pl.max_bbox.x1 < cx < pl.max_bbox.x2 and (pl.max_bbox.y1 + 2*(pl.max_bbox.height)//3) < cy < pl.max_bbox.y2:
                    self.color = pl.color
                    self.last_team_id = pl.team_id
                    self.last_team_letter = pl.team_letter
        self._draw_ball(frame, max_bbox)

    def _draw_ball(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        """Draws the ball as a filled circle."""
        r = min(bbox.width, bbox.height) // 2
        center_x = bbox.x1 + (bbox.width // 2)
        center_y = bbox.y1 + (bbox.height // 2)
        cv2.circle(frame, (center_x, center_y), r, self.color, -1)
