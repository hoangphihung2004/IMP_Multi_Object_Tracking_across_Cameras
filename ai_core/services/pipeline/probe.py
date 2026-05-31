"""
Analytics Probe for DeepStream.

Tracking-only runtime: extract detections, run OCSort per source, inject OSD metadata.
"""

import colorsys
import time
from typing import Dict, List

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

import numpy as np
import pyds
from loguru import logger

from core.config import OCSortConfig, OsdConfig
from services.analytics.tracking_service import TrackingService


class AnalyticsProbe:
    """Pad probe xử lý tracking và vẽ metadata cho từng source."""

    def __init__(
        self,
        tracking_services: Dict[int, TrackingService],
        tracking_config: OCSortConfig,
        osd_config: OsdConfig,
    ) -> None:
        self.tracking_services = tracking_services
        self.tracking_config = tracking_config
        self.osd_config = osd_config
        self.color_palette = self._generate_palette(256)
        self.frame_counts: Dict[int, int] = {source_id: 0 for source_id in tracking_services}
        self.fps_start_times: Dict[int, float] = {
            source_id: time.time() for source_id in tracking_services
        }
        self.current_fps: Dict[int, float] = {source_id: 0.0 for source_id in tracking_services}

    @staticmethod
    def _generate_palette(n: int) -> List[tuple]:
        """Sinh bảng màu tương phản cho track IDs."""
        golden_angle = 0.618033988749895
        palette = []
        for index in range(n):
            hue = (index * golden_angle) % 1.0
            saturation = 0.9 if (index % 2 == 0) else 0.7
            value = 1.0 if (index % 3 != 0) else 0.85
            red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
            palette.append((red, green, blue, 1.0))
        return palette

    def probe_callback(self, pad, info, u_data):
        """GStreamer Pad Probe callback."""
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            logger.error("[Probe] Unable to get GstBuffer from probe info")
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list

        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            self._process_frame_meta(frame_meta, batch_meta)

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    def _process_frame_meta(self, frame_meta, batch_meta) -> None:
        """Extract detections, chạy tracker đúng source, inject OSD boxes."""
        source_id = int(frame_meta.source_id)
        frame_id = int(frame_meta.frame_num)

        tracker = self.tracking_services.get(source_id)
        if tracker is None:
            logger.warning(f"[Probe] Missing tracker for source_id={source_id}; skipping frame.")
            return

        detections = self._extract_detections(frame_meta)
        detections_np = (
            np.array(detections, dtype=np.float32)
            if detections
            else np.empty((0, 6), dtype=np.float32)
        )

        frame_size = (frame_meta.source_frame_width, frame_meta.source_frame_height)
        tracked_objects = tracker.update(detections_np, frame_size, frame_size)
        filtered_tracks = self._filter_tracks(tracked_objects)

        if len(filtered_tracks) > 0:
            self._sync_track_ids(frame_meta, batch_meta, filtered_tracks)

        self._update_fps(source_id, frame_id, len(detections), len(filtered_tracks))

    def _extract_detections(self, frame_meta) -> List[List[float]]:
        """Lấy raw detections từ DeepStream metadata và ẩn box YOLO gốc."""
        detections: List[List[float]] = []
        l_obj = frame_meta.obj_meta_list

        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                rect = obj_meta.rect_params
                rect.border_width = 0
                rect.border_color.set(0.0, 0.0, 0.0, 0.0)
                obj_meta.text_params.display_text = ""
                obj_meta.text_params.set_bg_clr = 0

                detections.append(
                    [
                        float(rect.left),
                        float(rect.top),
                        float(rect.left + rect.width),
                        float(rect.top + rect.height),
                        float(obj_meta.confidence),
                        int(obj_meta.class_id),
                    ]
                )
                l_obj = l_obj.next
            except StopIteration:
                break

        return detections

    def _filter_tracks(self, tracked_objects: np.ndarray) -> np.ndarray:
        """Áp dụng filter aspect ratio/min area sau OCSort."""
        if len(tracked_objects) == 0:
            return np.empty((0, 7), dtype=np.float32)

        filtered = []
        ar_thresh = self.tracking_config.aspect_ratio_thresh
        area_thresh = self.tracking_config.min_box_area

        for track in tracked_objects:
            x1, y1, x2, y2 = track[:4]
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                continue
            is_horizontal = width / (height + 1e-6) > ar_thresh
            if width * height > area_thresh and not is_horizontal:
                filtered.append(track)

        if not filtered:
            return np.empty((0, 7), dtype=np.float32)
        return np.array(filtered, dtype=np.float32)

    def _sync_track_ids(self, frame_meta, batch_meta, tracked_objects: np.ndarray) -> None:
        """Inject NvDsObjectMeta mới với OCSort track IDs và màu theo ID."""
        if not self.osd_config.enabled:
            return

        for track_box in tracked_objects:
            track_id = int(track_box[4])
            x1, y1, x2, y2 = track_box[:4]
            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
            obj_meta.unique_component_id = 99
            obj_meta.confidence = 1.0
            obj_meta.class_id = 0
            obj_meta.object_id = track_id

            rect = obj_meta.rect_params
            rect.left = float(x1)
            rect.top = float(y1)
            rect.width = float(width)
            rect.height = float(height)

            color = self.color_palette[track_id % len(self.color_palette)]
            rect.border_color.set(*color)
            rect.border_width = self.osd_config.bbox_border_width

            text = obj_meta.text_params
            text.display_text = f"ID:{track_id}"
            text.x_offset = int(rect.left)
            text.y_offset = int(max(0, rect.top - 20))
            text.font_params.font_name = self.osd_config.font_name
            text.font_params.font_size = self.osd_config.font_size
            text.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            text.set_bg_clr = 1
            text.text_bg_clr.set(color[0], color[1], color[2], 0.6)

            pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)

    def _update_fps(self, source_id: int, frame_id: int, num_detections: int, num_tracks: int) -> None:
        """Log FPS định kỳ theo từng source."""
        self.frame_counts[source_id] = self.frame_counts.get(source_id, 0) + 1
        if self.frame_counts[source_id] % 30 != 0:
            return

        now = time.time()
        elapsed = now - self.fps_start_times[source_id]
        fps = 30.0 / elapsed if elapsed > 0 else 0.0
        self.current_fps[source_id] = fps
        self.fps_start_times[source_id] = now

        logger.info(
            f"[Probe] source={source_id} frame={frame_id} fps={fps:.1f} "
            f"detections={num_detections} tracks={num_tracks}"
        )
