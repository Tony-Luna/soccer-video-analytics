# modules/utils/config_loader.py
# -*- coding: utf-8 -*-
"""
Configuration Loader.

Loads config parameters from a YAML file.

@author: Tony-Luna
"""

import os
import yaml

def load_config(config_path: str) -> dict:
    """
    Load parameters from a YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: The loaded configuration.

    Raises:
        FileNotFoundError: If the config_path does not exist.
        ValueError: If required sections are missing.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Basic validation checks
    for required in ['usage_parameters', 'detection_parameters', 'homography_parameters']:
        if required not in config:
            raise ValueError(f"Missing '{required}' in config.yaml.")

    return config
