"""
Base abstractions for analytics services.

Sau khi cắt counting/mapping/backend, AI Core chỉ còn tracking trong runtime local.
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from core.config import OCSortConfig


class BaseTrackingService(ABC):
    """Interface chung cho tracking service."""

    @abstractmethod
    def __init__(self, config: OCSortConfig) -> None:
        """Khởi tạo tracking service từ OCSortConfig."""
        pass

    @abstractmethod
    def update(
        self,
        detections: np.ndarray,
        img_info: Tuple[int, int],
        img_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Update tracker với detections mới.

        Args:
            detections: Array [x1, y1, x2, y2, score, class_id].
            img_info: Kích thước frame gốc (width, height).
            img_size: Kích thước frame đầu vào tracker (width, height).

        Returns:
            Array [x1, y1, x2, y2, track_id, class_id, score].
        """
        pass
