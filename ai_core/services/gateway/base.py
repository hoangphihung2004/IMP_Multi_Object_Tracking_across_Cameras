"""
Base Abstractions for Gateway Services.
"""

from abc import ABC, abstractmethod
from schemas.schemas import FrameData


class BaseResultPublisher(ABC):
    """
    Abstract Interface for publishing pipeline results (e.g., WebSocket, MQTT, HTTP).
    """

    @abstractmethod
    def send_frame(self, frame_data: FrameData) -> None:
        """
        Publish frame data containing tracks, counts, and system status.
        """
        pass
