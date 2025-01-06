# modules/domain/goal_polygon.py
# -*- coding: utf-8 -*-
"""
Goal Polygon.

Allows user to draw a polygon representing the goal area in the first frame.
Tracks ball entering to increment scores.

@author: Tony-Luna
"""

import os
import cv2
import numpy as np
from typing import List, Tuple

class GoalPolygon:
    """
    Manages the user-defined polygon that marks a goal region.
    """

    def __init__(self, config: dict, teams_dict: dict) -> None:
        self.config = config
        usage = config['usage_parameters']
        self.video_path = usage['input_video_path']
        self.output_dir = self._ensure_output_dir()
        self.polygon_path = os.path.join(self.output_dir, "goal_polygon.npy")

        self.teams_dict = teams_dict
        self.scores_dict = {v['team_letter']: 0 for v in teams_dict.values()}
        for val in self.teams_dict.values():
            val['score'] = 0

        self.first_frame = self._load_first_frame()
        self.polygon: List[Tuple[int, int]] = []
        self.ball_in = False

        self._load_or_draw_polygon()

    def _ensure_output_dir(self) -> str:
        base_dir = self.config['usage_parameters']['output_base_dir']
        vid_name = os.path.splitext(os.path.basename(self.config['usage_parameters']['input_video_path']))[0]
        out_dir = os.path.join(base_dir, vid_name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _load_first_frame(self) -> np.ndarray:
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise IOError("Could not load first frame from video.")
        return frame

    def _load_or_draw_polygon(self) -> None:
        if os.path.exists(self.polygon_path):
            self.polygon = np.load(self.polygon_path, allow_pickle=True).tolist()
            if not isinstance(self.polygon, list):
                self.polygon = list(self.polygon)
        else:
            self._interactive_polygon()

    def _interactive_polygon(self) -> None:
        cv2.namedWindow("Goal Polygon")
        cv2.setMouseCallback("Goal Polygon", self._on_mouse)

        print("\n[Goal Polygon] Instructions:")
        print("- Click points to define the goal region.")
        print("- Press 'y' to finalize & save.")
        print("- Press 'r' to remove last point.")
        print("- Press 'Esc' to exit without saving.")

        while True:
            tmp = self.first_frame.copy()
            if len(self.polygon) > 1:
                cv2.polylines(tmp, [np.array(self.polygon, dtype=np.int32)], False, (0,255,0), 2)
            for pt in self.polygon:
                cv2.circle(tmp, pt, 5, (0,255,0), -1)

            cv2.imshow("Goal Polygon", tmp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('y'):
                if len(self.polygon) > 2 and self.polygon[0] != self.polygon[-1]:
                    self.polygon.append(self.polygon[0])
                np.save(self.polygon_path, np.array(self.polygon, dtype=object))
                break
            elif key == 27:
                cv2.destroyAllWindows()
                return
            elif key == ord('r'):
                if self.polygon:
                    self.polygon.pop()

        cv2.destroyAllWindows()

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.polygon.append((x, y))

    def draw_polygon_on_frame(self, frame: np.ndarray) -> None:
        """
        Draws the polygon (filled if ball is inside) on the main frame.
        """
        if len(self.polygon) < 2:
            return
        pts = np.array(self.polygon, dtype=np.int32)
        if self.ball_in:
            cv2.fillPoly(frame, [pts], (0,255,0))
        else:
            cv2.polylines(frame, [pts], True, (0,255,0), 3)

    def update_draw_score(self, ball_object, frame: np.ndarray) -> None:
        """
        Checks if ball is inside the polygon to update scores.

        Args:
            ball_object (Ball): The ball entity with center_point.
            frame (np.ndarray): The main BGR frame for drawing.
        """
        if len(self.polygon) > 2 and ball_object.center_point is not None:
            pts = np.array(self.polygon, dtype=np.int32)
            inside = cv2.pointPolygonTest(pts, ball_object.center_point, False)
            if not self.ball_in and inside == 1:
                self.ball_in = True
                if ball_object.last_team_id in self.teams_dict:
                    letter = self.teams_dict[ball_object.last_team_id]['team_letter']
                    self.teams_dict[ball_object.last_team_id]['score'] += 1
                    self.scores_dict[letter] = self.teams_dict[ball_object.last_team_id]['score']
            if self.ball_in and inside < 0:
                self.ball_in = False

        self._draw_score_boxes(frame)

    def _draw_score_boxes(self, frame: np.ndarray) -> None:
        """
        Renders the scoreboard at the top-left corner.
        """
        offset_x, offset_y = 10, 10
        label_height = 20
        box_width, box_height = 100, 50

        # Number of teams
        num_teams = len(self.teams_dict)
        total_w = num_teams * box_width

        # "SCORE" label
        cv2.rectangle(frame, (offset_x, offset_y), (offset_x + total_w, offset_y + label_height), (0,0,0), -1)
        text = "SCORE"
        scale = 0.5
        thick = 2
        t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0]
        tx = offset_x + (total_w - t_size[0])//2
        ty = offset_y + label_height - (label_height - t_size[1])//2
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thick)

        current_y = offset_y + label_height

        for idx, (team_id, data) in enumerate(self.teams_dict.items()):
            letter = data['team_letter']
            color = data['bgr_color']
            score_val = data.get('score', 0)

            # Label box
            x0 = offset_x + idx * box_width
            cv2.rectangle(frame, (x0, current_y), (x0 + box_width, current_y + label_height), color, -1)
            label_text = f"({letter.upper()})"
            lbl_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0]
            lx = x0 + (box_width - lbl_size[0])//2
            ly = current_y + label_height - (label_height - lbl_size[1])//2
            cv2.putText(frame, label_text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thick)

            # Score box
            cv2.rectangle(frame, (x0, current_y + label_height), (x0 + box_width, current_y + label_height + box_height), (255,255,255), -1)
            sc_text = str(score_val)
            sc_size = cv2.getTextSize(sc_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            scx = x0 + (box_width - sc_size[0])//2
            scy = current_y + label_height + (box_height + sc_size[1])//2 - 5
            cv2.putText(frame, sc_text, (scx, scy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
