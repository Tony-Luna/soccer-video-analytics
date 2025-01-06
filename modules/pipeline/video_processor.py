# modules/pipeline/video_processor.py
# -*- coding: utf-8 -*-
"""
Video Processor.

Coordinates the per-frame analysis: detection, scoring, layout projection,
and writes CSV/JSON analytics.

@author: Tony-Luna
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from modules.analytics.csv_writer import CsvWriter
from modules.analytics.report_writer import ReportWriter

class VideoProcessor:
    """
    Main class orchestrating video processing for soccer analysis.
    """

    def __init__(self, config, object_detector, goal_polygon, team_players, ball_object, layout_projector) -> None:
        self.config = config
        self.detector = object_detector
        self.goal_polygon = goal_polygon
        self.team_players = team_players
        self.ball_object = ball_object
        self.layout_projector = layout_projector

        self.output_dir = self._ensure_output_dir()
        self.csv_writer = CsvWriter(os.path.join(self.output_dir, "soccer_analytics.csv"))
        self.report_writer = ReportWriter(os.path.join(self.output_dir, "soccer_analytics_report.json"))

        self.scores_video_path = os.path.join(self.output_dir, "scores_video.mp4")
        self.layout_video_path = os.path.join(self.output_dir, "layout_video.mp4")
        self.heatmap_writers = {}
        self.layout_writer = None

        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.fps = 0.0

    def _ensure_output_dir(self) -> str:
        base_dir = self.config["usage_parameters"]["output_base_dir"]
        name = os.path.splitext(os.path.basename(self.config["usage_parameters"]["input_video_path"]))[0]
        out_dir = os.path.join(base_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def process_video(self) -> None:
        """
        Reads video frames, applies detection and domain logic, then writes final results.
        """
        cap = cv2.VideoCapture(self.config["usage_parameters"]["input_video_path"])
        if not cap.isOpened():
            print(f"Error opening video: {self.config['usage_parameters']['input_video_path']}")
            return

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.layout_projector.fps = self.fps

        # Writers
        main_writer = cv2.VideoWriter(self.scores_video_path, self.fourcc, self.fps, (frame_w, frame_h))
        layout_h, layout_w = self.layout_projector.layout_image.shape[:2]
        self.layout_writer = cv2.VideoWriter(self.layout_video_path, self.fourcc, self.fps, (layout_w, layout_h))

        # Heatmap writers
        for letter in self.layout_projector.heatmaps_dict.keys():
            hm_path = os.path.join(self.output_dir, f"heatmap_video_{letter}.mp4")
            self.heatmap_writers[letter] = cv2.VideoWriter(hm_path, self.fourcc, self.fps, (layout_w, layout_h))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=total_frames, desc="Processing Video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed, layout, overlays = self._process_frame(frame)
                # Write results
                main_writer.write(processed)
                self.layout_writer.write(layout)
                for k, v in overlays.items():
                    self.heatmap_writers[k].write(v)

                # Optionally visualize
                cv2.imshow("Detections", processed)
                cv2.imshow("Layout", layout)
                for k, overlay in overlays.items():
                    # Check that 'overlay' is valid and not empty
                    if overlay is not None and overlay.shape[0] > 0 and overlay.shape[1] > 0:
                        cv2.imshow(f"Heatmap_{k}", overlay)


                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

                pbar.update(1)

        cap.release()
        main_writer.release()
        self.layout_writer.release()
        for writer in self.heatmap_writers.values():
            writer.release()
        cv2.destroyAllWindows()

        # Save final heatmaps as images
        last_overlays = overlays  # the final iteration overlays
        for k, img in last_overlays.items():
            hm_img_path = os.path.join(self.output_dir, f"heatmap_image_{k}.png")
            cv2.imwrite(hm_img_path, img)

    def _process_frame(self, frame: np.ndarray):
        # 1. Detect
        detections = self.detector.detect(frame)

        # 2. Draw polygon
        self.goal_polygon.draw_polygon_on_frame(frame)

        # 3. Update each team's location
        for player in self.team_players:
            player.update_draw_location(detections, frame)

        # 4. Update ball location
        self.ball_object.update_draw_location(self.team_players, detections, frame)

        # 5. Check scoring
        self.goal_polygon.update_draw_score(self.ball_object, frame)

        # 6. Update layout
        layout_img, overlays = self.layout_projector.update_draw_layout_dict(self.team_players, self.ball_object)

        # 7. Draw ball possession times
        frame = self.layout_projector.update_draw_possession_time(frame)

        # 8. Update CSV
        self.csv_writer.update_csv(self.layout_projector.layout_dict)

        # 9. Update JSON report
        self.report_writer.update_report(self.goal_polygon.scores_dict, self.layout_projector.ball_poss_dict)

        return frame, layout_img, overlays
