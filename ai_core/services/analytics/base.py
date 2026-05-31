"""
Base abstractions for analytics services.

After removing counting/mapping/backend integration, AI Core only keeps tracking in the local runtime.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from core.config import OCSortConfig


class BaseTrackingService(ABC):
    """Interface chung cho tracking service."""

    @abstractmethod
    def __init__(self, config: OCSortConfig) -> None:
        """Initialize the tracking service from OCSortConfig."""
        pass

    @abstractmethod
    def update(
        self,
        detections: np.ndarray,
        img_info: Tuple[int, int],
        img_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Update the tracker with new detections.

        Args:
            detections: Array [x1, y1, x2, y2, score, class_id].
            img_info: Original frame size (width, height).
            img_size: Tracker input frame size (width, height).

        Returns:
            Array [x1, y1, x2, y2, track_id, class_id, score].
        """
        pass
