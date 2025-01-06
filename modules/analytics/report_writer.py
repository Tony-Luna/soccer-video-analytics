# modules/analytics/report_writer.py
# -*- coding: utf-8 -*-
"""
Report Writer.

Writes a final JSON report with scoring and ball possession data.

@author: Tony-Luna
"""

import os
import json
from typing import Dict

class ReportWriter:
    """
    Creates a JSON file summarizing final scores and ball possession times.
    """

    def __init__(self, report_path: str) -> None:
        self.report_path = report_path
        self._reset_file()

    def _reset_file(self) -> None:
        if os.path.isfile(self.report_path):
            os.remove(self.report_path)

    def update_report(self, scores_dict: Dict[str, int], ball_poss_dict: Dict[str, dict]) -> None:
        """
        Writes the final report merging scores and possession times.

        Args:
            scores_dict (dict): { "A": 1, "B": 2, ... }
            ball_poss_dict (dict): { "A": {"time":"00:05"}, ... }
        """
        report_data = {}
        for letter, score_val in scores_dict.items():
            report_data[letter] = {"score": score_val}

        for letter, data in ball_poss_dict.items():
            if letter not in report_data:
                report_data[letter] = {}
            report_data[letter]["time"] = data.get("time", "00:00")

        with open(self.report_path, 'w') as f:
            json.dump(report_data, f, indent=4)
