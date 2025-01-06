# modules/analytics/csv_writer.py
# -*- coding: utf-8 -*-
"""
CSV Writer.

Writes per-frame analytics to a CSV file.

@author: Tony-Luna
"""

import os
import csv
from typing import Dict, List, Any

class CsvWriter:
    """Writes analytics data into a CSV file, one row per frame."""

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self._reset_file()

    def _reset_file(self) -> None:
        if os.path.isfile(self.csv_path):
            os.remove(self.csv_path)

    def update_csv(self, analytics_dict: Dict[str, List[Any]]) -> None:
        """
        Updates the CSV by adding one row from the last data point of each key.

        Args:
            analytics_dict (dict): e.g., { "teamA": [pt1, pt2,...], "ball": [...], ... }
        """
        if not analytics_dict:
            return

        # frame number => length of the first list
        frame_num = len(next(iter(analytics_dict.values())))  
        # If file is empty, write headers
        if not os.path.isfile(self.csv_path) or os.stat(self.csv_path).st_size == 0:
            self._init_csv(list(analytics_dict.keys()))

        self._append_data(frame_num, analytics_dict)

    def _init_csv(self, headers: list) -> None:
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame"] + headers)

    def _append_data(self, frame_num: int, analytics_dict: Dict[str, List[Any]]) -> None:
        row = [frame_num]
        for key in analytics_dict:
            data_list = analytics_dict[key]
            val = data_list[-1] if data_list else None
            row.append(val)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
