# modules/visualization/layout_projector.py
# -*- coding: utf-8 -*-
"""
Layout Projector.

Projects player/ball points onto the layout image with heatmaps,
and keeps track of ball possession time.

@author: Tony-Luna
"""

import cv2
import numpy as np
import math

class LayoutProjector:
    """
    Projects on-field positions onto a layout image via homography,
    computes heatmaps, and tracks ball possession per team.
    """

    def __init__(self, config: dict, H: np.ndarray, teams_dict: dict):
        self.config = config
        self.H = H
        self.teams_dict = teams_dict

        layout_path = config['homography_parameters']['input_layout_image']
        self.layout_image = cv2.imread(layout_path)
        self.layout_img_gray = cv2.cvtColor(cv2.cvtColor(self.layout_image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

        h, w = self.layout_image.shape[:2]
        self.heatmaps_dict = {v['team_letter']: np.zeros((h, w), dtype=np.float32) for v in teams_dict.values()}
        self.overlay_heatmaps_dict = {k: None for k in self.heatmaps_dict}

        # Ball possession times
        self.ball_poss_dict = {
            v['team_letter']: {'color': v['bgr_color'], 'frames_count': 0, 'time': "00:00"}
            for v in teams_dict.values()
        }

        # For CSV logging
        self.layout_dict = {v['team_letter']: [] for v in teams_dict.values()}
        self.layout_dict["ball"] = []
        self.layout_dict["ball_possession"] = []

        self.fps = 30.0  # updated from video

    def update_draw_layout_dict(self, players, ball_object):
        """
        Updates layout data for each frame, transforms points, updates heatmaps.
        Returns (drawn_layout, overlay_heatmaps).
        """
        temp_dict = {}

        # Transform player centers
        for p in players:
            p_center = self._apply_homography(p.center_point)
            temp_dict[p.team_letter] = {"point": p_center, "color": p.color}

        # Transform ball center
        b_center = self._apply_homography(ball_object.center_point)
        temp_dict["ball"] = {"point": b_center, "color": ball_object.color}

        # Save to layout_dict for CSV
        for k, v in temp_dict.items():
            if k not in self.layout_dict:
                self.layout_dict[k] = []
            self.layout_dict[k].append(v["point"])

        # Ball possession
        if ball_object.last_team_letter:
            self.ball_poss_dict[ball_object.last_team_letter]["frames_count"] += 1
            self._convert_frames_to_time()

        self.layout_dict["ball_possession"].append(ball_object.last_team_letter)

        drawn_layout = self._draw_points_and_heatmaps(temp_dict)
        return drawn_layout, self.overlay_heatmaps_dict

    def _apply_homography(self, point):
        if point is None:
            return None
        point_h = np.array([point[0], point[1], 1], dtype=float)
        transformed = self.H @ point_h
        if transformed[2] != 0:
            return (transformed[0] / transformed[2], transformed[1] / transformed[2])
        return None

    def _draw_points_and_heatmaps(self, temp_dict):
        base_img = self.layout_image.copy()
        r = 10
        border = 3

        for k, v in temp_dict.items():
            pt = v["point"]
            if pt is None:
                continue
            x, y = int(pt[0]), int(pt[1])
            color = v["color"]

            if k == "ball":
                border_col = (255,255,255)
            else:
                border_col = (0,0,0)
                # Update heatmap
                self._update_heatmap(k, x, y, r)

            # Draw
            cv2.circle(base_img, (x, y), r+border, border_col, -1)
            cv2.circle(base_img, (x, y), r, color, -1)

        return base_img

    def _update_heatmap(self, team_letter: str, x: int, y: int, radius: int) -> None:
        h, w = self.layout_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (x, y), radius, 1.0, -1)
        self.heatmaps_dict[team_letter] += mask
        self._generate_overlay(team_letter)

    def _generate_overlay(self, team_letter: str, base: float = 10, alpha: float = 0.5) -> None:
        heatmap = self.heatmaps_dict[team_letter]
        log_img = np.log1p(heatmap) / math.log(base)
        norm_img = cv2.normalize(log_img, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_u8 = norm_img.astype(np.uint8)
        colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        overlayed = cv2.addWeighted(self.layout_img_gray, 1 - alpha, colored, alpha, 0)
        self.overlay_heatmaps_dict[team_letter] = overlayed

    def _convert_frames_to_time(self):
        for data in self.ball_poss_dict.values():
            total_seconds = data["frames_count"] / self.fps
            mins = int(total_seconds // 60)
            secs = int(total_seconds % 60)
            data["time"] = f"{mins:02d}:{secs:02d}"

    def update_draw_possession_time(self, frame: np.ndarray) -> np.ndarray:
        """
        Draws the ball possession times at top-right of the main frame.
        """
        h, w = frame.shape[:2]
        offset_x, offset_y = 10, 10
        label_h = 20
        box_w, box_h = 100, 50
        scale = 0.5
        thick = 2

        n_teams = len(self.ball_poss_dict)
        total_w = n_teams * box_w
        top_right_x = w - offset_x - total_w

        # "POSSESSION" label
        cv2.rectangle(frame, (top_right_x, offset_y), (w - offset_x, offset_y + label_h), (0,0,0), -1)
        txt = "POSSESSION"
        t_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0]
        tx = top_right_x + (total_w - t_size[0])//2
        ty = offset_y + label_h - (label_h - t_size[1])//2
        cv2.putText(frame, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thick)

        start_y = offset_y + label_h
        sorted_keys = sorted(self.ball_poss_dict.keys(), reverse=True)
        for idx, letter in enumerate(sorted_keys):
            color = self.ball_poss_dict[letter]["color"]
            time_str = self.ball_poss_dict[letter]["time"]

            cell_x = w - offset_x - (box_w * (idx+1))
            # Label box
            cv2.rectangle(frame, (cell_x, start_y), (cell_x+box_w, start_y+label_h), color, -1)
            lbl_text = f"({letter.upper()})"
            l_size = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0]
            lx = cell_x + (box_w - l_size[0])//2
            ly = start_y + label_h - (label_h - l_size[1])//2
            cv2.putText(frame, lbl_text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thick)

            # White box for possession time
            cv2.rectangle(frame, (cell_x, start_y+label_h), (cell_x+box_w, start_y+label_h+box_h), (255,255,255), -1)
            tm_size = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            tx2 = cell_x + (box_w - tm_size[0])//2
            ty2 = start_y + label_h + (box_h + tm_size[1])//2 - 5
            cv2.putText(frame, time_str, (tx2, ty2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

        return frame
