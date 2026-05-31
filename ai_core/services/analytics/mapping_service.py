"""
Mapping Service Implementation.
Projects camera pixel coordinates to 2D map coordinates using
a Homography matrix computed from user-defined correspondences.

Logic is preserved 100% from the legacy ai_core MappingService.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from services.analytics.base import BaseMappingService
from core.config import MappingConfig


class MappingService(BaseMappingService):
    """
    Maps bounding box foot-points from camera pixel space to 2D map space
    using a RANSAC-computed Homography matrix.

    Input:  bboxes List[[x1, y1, x2, y2]], frame_size, map_size
    Output: List of (map_x, map_y) coordinates on the 2D floor plan.
    """

    def __init__(self, config: MappingConfig) -> None:
        """
        Initialize the mapping service.

        Args:
            config: Pydantic model loaded from mapping_config.json.
                    Must contain 'correspondences', 'image_size', and 'map_size'.
        """
        self.config = config

        # Extract camera and map points from correspondences
        if not config.correspondences or len(config.correspondences) < 4:
            raise ValueError(
                f"Need at least 4 correspondence points, got {len(config.correspondences)}."
            )

        self._camera_points = np.array(
            [c.camera for c in config.correspondences], dtype=np.float32
        )
        self._map_points = np.array(
            [c.map for c in config.correspondences], dtype=np.float32
        )

        # Store reference sizes from config
        self._camera_size: Tuple[int, int] = (
            config.image_size["width"],
            config.image_size["height"],
        )
        self._map_size: Tuple[int, int] = (
            config.map_size["width"],
            config.map_size["height"],
        )

        # Compute homography matrix via RANSAC
        self._homography_matrix, self._inlier_mask = cv2.findHomography(
            self._camera_points, self._map_points, cv2.RANSAC, 5.0
        )
        if self._homography_matrix is None:
            raise RuntimeError("Failed to compute homography matrix from correspondence points.")

    def _project_point(self, x: float, y: float) -> List[float]:
        """
        Project a single (x, y) point using the homography matrix.

        Args:
            x: Pixel x in camera space.
            y: Pixel y in camera space.

        Returns:
            [map_x, map_y] in map space.
        """
        point = np.array([[[x, y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, self._homography_matrix)
        return [float(mapped[0][0][0]), float(mapped[0][0][1])]

    def project(
        self,
        x: float,
        y: float,
        current_frame_size: Tuple[int, int],
        current_map_size: Tuple[int, int],
    ) -> List[float]:
        """
        Project a single camera-space point to map-space,
        with runtime scaling for both frame and map resolutions.

        Args:
            x: Pixel x in current frame.
            y: Pixel y in current frame.
            current_frame_size: (width, height) of the live camera frame.
            current_map_size: (width, height) of the runtime map display.

        Returns:
            [map_x, map_y] scaled to current_map_size.
        """
        cur_w, cur_h = current_frame_size
        cam_w, cam_h = self._camera_size

        # Scale pixel coordinates from runtime frame to reference (label-time) frame
        scaled_x = x * cam_w / cur_w
        scaled_y = y * cam_h / cur_h

        # Apply homography
        mapped = self._project_point(scaled_x, scaled_y)

        # Scale map output to runtime map size
        map_w_label, map_h_label = self._map_size
        map_w_runtime, map_h_runtime = current_map_size
        mapped[0] = mapped[0] * map_w_runtime / map_w_label
        mapped[1] = mapped[1] * map_h_runtime / map_h_label

        return mapped

    def project_bbox(
        self,
        bbox: List[float],
        current_frame_size: Tuple[int, int],
        current_map_size: Tuple[int, int],
    ) -> List[float]:
        """
        Project a bounding box's foot-point (bottom-center) to map space.

        Args:
            bbox: [x1, y1, x2, y2] in current frame pixels.
            current_frame_size: (width, height) of the live camera frame.
            current_map_size: (width, height) of the runtime map display.

        Returns:
            [map_x, map_y] on the 2D floor plan.
        """
        x1, y1, x2, y2 = bbox
        foot_x = (x1 + x2) / 2.0
        foot_y = y2
        return self.project(foot_x, foot_y, current_frame_size, current_map_size)

    def project_bboxes(
        self,
        bboxes: List[List[float]],
        frame_size: Tuple[int, int],
        map_size: Tuple[int, int],
    ) -> List[Tuple[float, float]]:
        """
        Project multiple bounding boxes to map coordinates.

        Args:
            bboxes: List of [x1, y1, x2, y2].
            frame_size: Current camera frame dimensions.
            map_size: Current map display dimensions.

        Returns:
            List of (map_x, map_y) tuples.
        """
        if not bboxes:
            return []
        return [
            tuple(self.project_bbox(bbox, frame_size, map_size))
            for bbox in bboxes
        ]

    @property
    def is_ready(self) -> bool:
        """Check if the homography matrix is valid."""
        return self._homography_matrix is not None

    @property
    def num_correspondences(self) -> int:
        """Number of calibration point pairs."""
        return len(self._camera_points)

    @property
    def num_inliers(self) -> int:
        """Number of RANSAC inliers used in homography estimation."""
        return int(np.sum(self._inlier_mask)) if self._inlier_mask is not None else 0

    def update_mapping(
        self,
        camera_points: List[List[float]],
        map_points: List[List[float]],
        camera_size: Tuple[int, int],
        map_size: Tuple[int, int],
    ) -> None:
        """
        Hot-reload mapping data at runtime (called by SettingsSubscriber).
        Replaces correspondence points and recomputes the Homography matrix.

        Args:
            camera_points: New list of [x, y] camera calibration points.
            map_points: New list of [x, y] map calibration points.
            camera_size: (width, height) of the camera frame used during labeling.
            map_size: (width, height) of the 2D map.
        """
        if len(camera_points) != len(map_points):
            raise ValueError("camera_points and map_points must have the same length.")
        if len(camera_points) < 4:
            raise ValueError(f"Need at least 4 correspondence points, got {len(camera_points)}.")

        cam_w, cam_h = camera_size
        map_w, map_h = map_size
        if cam_w <= 0 or cam_h <= 0:
            raise ValueError(f"Invalid camera_size: {camera_size}")
        if map_w <= 0 or map_h <= 0:
            raise ValueError(f"Invalid map_size: {map_size}")

        self._camera_points = np.array(camera_points, dtype=np.float32)
        self._map_points = np.array(map_points, dtype=np.float32)
        self._camera_size = (int(cam_w), int(cam_h))
        self._map_size = (int(map_w), int(map_h))

        H, mask = cv2.findHomography(self._camera_points, self._map_points, cv2.RANSAC, 5.0)
        if H is None:
            raise RuntimeError("Failed to recompute homography matrix.")
        self._homography_matrix = H
        self._inlier_mask = mask
