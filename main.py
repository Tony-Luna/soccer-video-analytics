# main.py
# -*- coding: utf-8 -*-
"""
Main script for Soccer Video Analysis.

Implements the pipeline: config loading, logging, homography, HSV setup,
entity instantiation, and final video processing.

@author: anlun
"""

import sys
from modules.utils.config_loader import load_config
from modules.utils.logger_manager import get_logger

from modules.color_analysis.hsv_setup import HSVRangeSetup
from modules.geometry.homography_setup import HomographySetup
from modules.detection.object_detector import ObjectDetector
from modules.domain.goal_polygon import GoalPolygon
from modules.domain.objects_entities import TeamPlayer, Ball
from modules.visualization.layout_projector import LayoutProjector
from modules.pipeline.video_processor import VideoProcessor

def main() -> None:
    """Executes the soccer video analysis pipeline."""
    config_path = "config.yaml"
    config = load_config(config_path)

    # Set up logger
    verbosity = config['usage_parameters'].get('verbosity', 1)
    logger = get_logger(verbosity)

    logger.info("Starting Soccer Video Analysis...")

    # 1) HSV Ranges setup
    logger.info("Setting up HSV color ranges...")
    hsv_setup = HSVRangeSetup(config)
    teams_dict = hsv_setup.setup_hsv_ranges()

    # 2) YOLO Object Detector
    logger.info("Initializing object detector...")
    model_path = config['detection_parameters']['yolo_model_path']
    object_detector = ObjectDetector(model_path)

    # 3) Homography
    logger.info("Computing/loading homography matrix...")
    homography = HomographySetup(config)
    h_matrix = homography.compute_homography_matrix()
    if h_matrix is None:
        logger.warning("Homography not computed. Exiting.")
        sys.exit(1)

    # 4) Layout projector
    layout_proj = LayoutProjector(config, h_matrix, teams_dict)

    # 5) Goal polygon
    logger.info("Setting up goal polygon...")
    goal_polygon = GoalPolygon(config, teams_dict)

    # 6) Create domain entities
    logger.info("Initializing team players & ball objects...")
    team_players = [TeamPlayer(config, key, teams_dict) for key in teams_dict.keys()]
    ball_object = Ball(config)

    # 7) Process Video
    logger.info("Starting video processing...")
    processor = VideoProcessor(config, object_detector, goal_polygon, team_players, ball_object, layout_proj)
    processor.process_video()

    logger.info("Soccer Video Analysis complete.")

if __name__ == "__main__":
    main()
