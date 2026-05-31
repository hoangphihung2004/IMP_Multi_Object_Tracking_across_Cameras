"""
Counting Service Implementation.
State-based line-crossing counter that tracks objects entering/exiting
across a user-defined counting line.

Includes Debouncing (min_hits) and EMA Smoothing (ema_alpha) to prevent
false counts caused by bounding box jitter.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple

from services.analytics.base import BaseCountingService
from core.config import CountingConfig


class LineCounter:
    """
    Core line-crossing detection engine.
    Each tracked object is assigned an 'inside' or 'outside' state.
    A count is registered only when a clear state transition occurs
    across the crossing margin band and maintains that state for a minimum
    number of frames (debouncing).
    """

    def __init__(
        self,
        start_point: List[float],
        end_point: List[float],
        inside_point: List[float],
        crossing_margin: float = 10.0,
        min_hits: int = 3  # Thêm tham số lọc nhiễu frame
    ) -> None:
        """
        Initialize line geometry and direction vectors.

        Args:
            start_point: Line start coordinate [x, y].
            end_point: Line end coordinate [x, y].
            inside_point: A reference point on the 'inside' direction.
            crossing_margin: Half-width of the no-decision band (pixels).
            min_hits: Minimum consecutive frames required to confirm a state change.
        """
        self.start_point = np.array(start_point, dtype=np.float64)
        self.end_point = np.array(end_point, dtype=np.float64)
        self.crossing_margin = crossing_margin
        self.min_hits = min_hits

        # Compute direction vectors
        self.line_vector = self.end_point - self.start_point
        self.line_length = np.linalg.norm(self.line_vector)
        if self.line_length == 0:
            raise ValueError("Line start and end points must not be identical.")
        self.unit_line_vector = self.line_vector / self.line_length
        self.normal_vector = np.array(
            [-self.unit_line_vector[1], self.unit_line_vector[0]]
        )

        # Determine which side is 'inside'
        inside_distance = self._signed_distance(inside_point)
        self.in_sign = 1 if inside_distance > 0 else -1

        # State tracking per object ID
        self.object_states: Dict[int, str] = {}  # Trạng thái đã confirm
        self.raw_states: Dict[int, str] = {}     # Trạng thái thô hiện tại
        self.hit_streaks: Dict[int, int] = {}    # Đếm số frame liên tiếp

        self.entry_count: int = 0
        self.exit_count: int = 0

    def _signed_distance(self, point: List[float]) -> float:
        """Calculate signed perpendicular distance from point to line."""
        pt = np.array(point, dtype=np.float64)
        return float(np.dot(pt - self.start_point, self.normal_vector))

    def update(self, object_id: int, center_point: Tuple[float, float]) -> str:
        """
        Update object position and detect line crossings with debouncing.

        Args:
            object_id: Track ID from TrackingService.
            center_point: (x, y) foot-point of the bounding box.

        Returns:
            'enter' if object crossed from outside to inside,
            'exit' if object crossed from inside to outside,
            '' (empty string) otherwise.
        """
        current_distance = self._signed_distance(list(center_point))
        relative_dist = current_distance * self.in_sign

        # Determine current raw state based on margin band
        current_raw_state = None
        if relative_dist > self.crossing_margin:
            current_raw_state = "inside"
        elif relative_dist < -self.crossing_margin:
            current_raw_state = "outside"

        # Inside the no-decision band -> skip logic, state streak remains
        if current_raw_state is None:
            return ""

        # Update streak counter
        if self.raw_states.get(object_id) != current_raw_state:
            self.raw_states[object_id] = current_raw_state
            self.hit_streaks[object_id] = 1
        else:
            self.hit_streaks[object_id] += 1

        # Only process state change if streak meets min_hits
        if self.hit_streaks[object_id] >= self.min_hits:
            prev_confirmed_state = self.object_states.get(object_id)

            # First time seeing this ID confirmed -> store initial state, do not count
            if prev_confirmed_state is None:
                self.object_states[object_id] = current_raw_state
                return ""

            # Detect clear, confirmed state transitions
            if prev_confirmed_state == "outside" and current_raw_state == "inside":
                self.entry_count += 1
                self.object_states[object_id] = "inside"
                return "enter"

            if prev_confirmed_state == "inside" and current_raw_state == "outside":
                self.exit_count += 1
                self.object_states[object_id] = "outside"
                return "exit"

        # State unconfirmed or unchanged
        return ""

    def reset(self) -> None:
        """Reset all counting state."""
        self.object_states.clear()
        self.raw_states.clear()
        self.hit_streaks.clear()
        self.entry_count = 0
        self.exit_count = 0

    def update_line(
        self,
        start_point: List[float],
        end_point: List[float],
        inside_point: List[float],
        crossing_margin: Optional[float] = None,
    ) -> None:
        """Update line geometry and derived vectors at runtime."""
        self.start_point = np.array(start_point, dtype=np.float64)
        self.end_point = np.array(end_point, dtype=np.float64)
        if crossing_margin is not None:
            self.crossing_margin = crossing_margin

        self.line_vector = self.end_point - self.start_point
        self.line_length = np.linalg.norm(self.line_vector)
        if self.line_length == 0:
            raise ValueError("start_point and end_point must not be identical.")

        self.unit_line_vector = self.line_vector / self.line_length
        self.normal_vector = np.array(
            [-self.unit_line_vector[1], self.unit_line_vector[0]]
        )

        inside_distance = self._signed_distance(inside_point)
        self.in_sign = 1 if inside_distance > 0 else -1

        # Clear previous states because line geometry changed
        self.reset()


class CountingService(BaseCountingService):
    """
    High-level Counting Service.
    Receives tracked bounding boxes and IDs from TrackingService,
    applies EMA smoothing to foot-points, and delegates to LineCounter.

    Input:  bboxes List[[x1, y1, x2, y2]], track_ids List[int]
    Output: get_counts() -> {'in': N, 'out': M, 'current': K}
    """

    def __init__(
        self,
        counting_config: Optional[CountingConfig] = None,
        *,
        line_start: Optional[List[float]] = None,
        line_end: Optional[List[float]] = None,
        inside_point: Optional[List[float]] = None,
        crossing_margin: float = 10.0,
        source_frame_size: Optional[Tuple[int, int]] = None,
        current_frame_size: Optional[Tuple[int, int]] = None,
        normalized: bool = False,
        min_hits: int = 3,       # Tham số Debouncing
        ema_alpha: float = 0.6   # Tham số EMA (0.0 -> 1.0)
    ) -> None:
        """
        Initialize with a CountingConfig or explicit geometry.
        """
        if counting_config is not None:
            line_start = counting_config.line_start
            line_end = counting_config.line_end
            inside_point = counting_config.inside_point
            crossing_margin = counting_config.crossing_margin
            normalized = getattr(counting_config, "normalized", False)
            if getattr(counting_config, "frame_width", None) and getattr(counting_config, "frame_height", None):
                source_frame_size = (counting_config.frame_width, counting_config.frame_height)

        if line_start is None or line_end is None or inside_point is None:
            raise ValueError("line_start, line_end, and inside_point are required.")

        start, end, inside = self._scale_geometry(
            line_start, line_end, inside_point,
            source_frame_size=source_frame_size,
            current_frame_size=current_frame_size,
            normalized=normalized,
        )

        # Lưu thông số làm mượt
        self.ema_alpha = ema_alpha
        self.point_history: Dict[int, Tuple[float, float]] = {}

        self.line_counter = LineCounter(
            start_point=list(start),
            end_point=list(end),
            inside_point=list(inside),
            crossing_margin=crossing_margin,
            min_hits=min_hits
        )

    @staticmethod
    def _scale_geometry(
        line_start: List[float],
        line_end: List[float],
        inside_point: List[float],
        source_frame_size: Optional[Tuple[int, int]],
        current_frame_size: Optional[Tuple[int, int]],
        normalized: bool,
    ) -> Tuple[tuple, tuple, tuple]:
        """
        Scale geometry from source frame to current frame
        or from normalized coordinates.
        """
        if normalized:
            if current_frame_size is None:
                raise ValueError("current_frame_size is required when using normalized coordinates.")
            dst_w, dst_h = current_frame_size
            scale = (dst_w, dst_h)
            pts = np.array([line_start, line_end, inside_point], dtype=np.float64) * np.array(scale, dtype=np.float64)
            return tuple(pts[0]), tuple(pts[1]), tuple(pts[2])

        if current_frame_size is None:
            return tuple(line_start), tuple(line_end), tuple(inside_point)

        if source_frame_size is None:
            return tuple(line_start), tuple(line_end), tuple(inside_point)

        src_w, src_h = source_frame_size
        dst_w, dst_h = current_frame_size
        scale = np.array([dst_w / src_w, dst_h / src_h], dtype=np.float64)
        pts = np.array([line_start, line_end, inside_point], dtype=np.float64) * scale
        return tuple(pts[0]), tuple(pts[1]), tuple(pts[2])

    def update(self, bboxes: List[List[float]], track_ids: List[int]) -> None:
        """
        Process one frame of tracking results with EMA smoothing.

        Args:
            bboxes: List of [x1, y1, x2, y2] bounding boxes.
            track_ids: Corresponding track IDs.
        """
        for bbox, tid in zip(bboxes, track_ids):
            x1, y1, x2, y2 = bbox

            # Tính tọa độ foot-point thô (cạnh dưới cùng của box)
            raw_foot_x = (x1 + x2) / 2.0
            raw_foot_y = float(y2)
            tid_int = int(tid)

            # Áp dụng bộ lọc EMA
            if tid_int in self.point_history:
                prev_x, prev_y = self.point_history[tid_int]
                foot_x = self.ema_alpha * raw_foot_x + (1.0 - self.ema_alpha) * prev_x
                foot_y = self.ema_alpha * raw_foot_y + (1.0 - self.ema_alpha) * prev_y
            else:
                foot_x, foot_y = raw_foot_x, raw_foot_y

            # Lưu lại tọa độ mượt để dùng cho frame sau
            self.point_history[tid_int] = (foot_x, foot_y)

            # Cập nhật đếm
            self.line_counter.update(tid_int, (foot_x, foot_y))

    def get_counts(self) -> Dict[str, int]:
        """
        Retrieve current counting statistics.
        """
        return {
            "in": self.line_counter.entry_count,
            "out": self.line_counter.exit_count,
            "current": max(0, self.line_counter.entry_count - self.line_counter.exit_count),
        }

    @property
    def count_in(self) -> int:
        return self.line_counter.entry_count

    @property
    def count_out(self) -> int:
        return self.line_counter.exit_count

    @property
    def current_people(self) -> int:
        return max(0, self.line_counter.entry_count - self.line_counter.exit_count)

    def draw_margin_region(self, frame: np.ndarray, color=(255, 0, 255), thickness=1) -> np.ndarray:
        n = self.line_counter.normal_vector
        r = self.line_counter.crossing_margin
        pts_top, pts_bot = [], []

        for t in np.linspace(0, 1, 50):
            pt = self.line_counter.start_point + t * self.line_counter.line_vector
            pts_top.append((pt + n * r).astype(int))
            pts_bot.append((pt - n * r).astype(int))

        region_pts = np.array(pts_top + pts_bot[::-1], dtype=np.int32)
        cv2.polylines(frame, [region_pts], True, color, thickness)
        return frame

    def update_config(
        self,
        line_start: List[float],
        line_end: List[float],
        inside_point: List[float],
        crossing_margin: float = 10.0,
        source_frame_size: Optional[Tuple[int, int]] = None,
        current_frame_size: Optional[Tuple[int, int]] = None,
        normalized: bool = False,
    ) -> None:
        start, end, inside = self._scale_geometry(
            line_start, line_end, inside_point,
            source_frame_size=source_frame_size,
            current_frame_size=current_frame_size,
            normalized=normalized,
        )
        self.line_counter.update_line(list(start), list(end), list(inside), crossing_margin)

    def reset(self) -> None:
        """Reset all counting state and tracking history."""
        self.line_counter.reset()
        self.point_history.clear()

    @classmethod
    def from_config(
        cls,
        config: CountingConfig,
        current_frame_size: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> "CountingService":
        return cls(counting_config=config, current_frame_size=current_frame_size, **kwargs)