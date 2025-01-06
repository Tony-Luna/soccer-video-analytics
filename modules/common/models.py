# modules/common/models.py
# -*- coding: utf-8 -*-
"""
Common Data Models.

Holds shared data structures used by multiple modules.
"""

from dataclasses import dataclass
from typing import Tuple

@dataclass
class BoundingBox:
    """
    Represents a bounding box in x1, y1, x2, y2 format.
    """
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def center_bottom(self) -> Tuple[int, int]:
        """Returns the (x, y) at the bottom center of this box."""
        x_mid = self.x1 + (self.width // 2)
        return (x_mid, self.y2)
