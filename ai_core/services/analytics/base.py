"""
Base Abstractions for Analytics Services.
Defines the interface contract that all analytics service implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from core.config import OCSortConfig, CountingConfig, MappingConfig


class BaseTrackingService(ABC):
    @abstractmethod
    def __init__(self, config: OCSortConfig) -> None:
        pass

    @abstractmethod
    def update(
        self,
        detections: np.ndarray,
        img_info: Tuple[int, int],
        img_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Update tracker with new detections.

        Args:
            detections: Array of [x1, y1, x2, y2, score, class_id].
            img_info: Original image dimensions (height, width).
            img_size: Network input dimensions (height, width).

        Returns:
            Array of [x1, y1, x2, y2, track_id, class_id, score].
        """
        pass


class BaseCountingService(ABC):
    @abstractmethod
    def update(self, bboxes: List[List[float]], track_ids: List[int]) -> None:
        pass

    @abstractmethod
    def get_counts(self) -> Dict[str, int]:
        pass


class BaseMappingService(ABC):
    @abstractmethod
    def __init__(self, config: MappingConfig) -> None:
        pass

    @abstractmethod
    def project_bboxes(
        self,
        bboxes: List[List[float]],
        frame_size: Tuple[int, int],
        map_size: Tuple[int, int],
    ) -> List[Tuple[float, float]]:
        pass
