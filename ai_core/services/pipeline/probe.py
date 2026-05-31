"""
Analytics Probe for DeepStream.
Orchestrates Tracking, Counting, Mapping, and Result Publishing.
Handles metadata extraction and visual injection for OSD.
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import time
from datetime import datetime, timezone
import numpy as np
import pyds
from loguru import logger

from services.analytics.tracking_service import TrackingService
from services.analytics.counting_service import CountingService
from services.analytics.mapping_service import MappingService
from services.gateway.backend_gateway import ResultPublisher
from schemas.schemas import FrameData, ObjectDetection


class AnalyticsProbe:
    """
    Pad probe handler that runs analytics on DeepStream metadata.
    """

    def __init__(
        self,
        tracking_service: TrackingService,
        counting_service: CountingService,
        mapping_service: MappingService,
        publisher: ResultPublisher,
        warning_threshold: int = 25,
        critical_threshold: int = 50,
    ):
        self.tracking_service = tracking_service
        self.counting_service = counting_service
        self.mapping_service = mapping_service
        self.publisher = publisher
        
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        # Visual styles (matching legacy)
        self.line_color = (255, 0, 255, 1.0)  # Magenta (RGBA for pyds)
        self.text_color = (1.0, 1.0, 1.0, 1.0)  # White
        
        # High-contrast colors for tracks (Zeno's Dichotomy replication simplified)
        self.color_palette = self._generate_palette(256)

        # Performance Monitoring
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0

    @staticmethod
    def _generate_palette(n: int):
        """Generate colors with maximum contrast between consecutive IDs using golden angle."""
        import colorsys
        golden_angle = 0.618033988749895  # 1 / golden_ratio
        palette = []
        for i in range(n):
            hue = (i * golden_angle) % 1.0  # golden angle spacing
            saturation = 0.9 if (i % 2 == 0) else 0.7  # alternate saturation
            value = 1.0 if (i % 3 != 0) else 0.85       # alternate brightness
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            palette.append((r, g, b, 1.0))
        return palette

    def probe_callback(self, pad, info, u_data):
        """
        GStreamer Pad Probe Callback.
        """
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            logger.error("Unable to get GstBuffer from probe info")
            return Gst.PadProbeReturn.OK

        # 1. Access Batch Metadata
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

    def _process_frame_meta(self, frame_meta, batch_meta):
        """Extract objects, run services, and update metadata."""
        frame_id = frame_meta.frame_num
        timestamp = datetime.now(timezone.utc)
        
        # 1. Extract ALL raw detections from nvinfer metadata (no filtering!)
        #    Legacy Python: outputs[0] was passed DIRECTLY to tracker.update()
        detections = []
        l_obj = frame_meta.obj_meta_list

        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                rect = obj_meta.rect_params
                
                # Hide ALL raw YOLO boxes on OSD. Only OCSort-confirmed boxes will be drawn.
                rect.border_width = 0
                rect.border_color.set(0.0, 0.0, 0.0, 0.0)
                obj_meta.text_params.display_text = ""
                obj_meta.text_params.set_bg_clr = 0
                
                # Format: [x1, y1, x2, y2, score, class_id]
                detections.append([
                    rect.left, 
                    rect.top, 
                    rect.left + rect.width, 
                    rect.top + rect.height, 
                    obj_meta.confidence, 
                    obj_meta.class_id
                ])

                l_obj = l_obj.next
            except StopIteration:
                break

        if not detections:
            # Still need to update counters if no one is in frame
            self.counting_service.update([], [])
            self._send_empty_frame(frame_meta, timestamp)
            self._draw_osd(frame_meta, batch_meta)
            return

        # DEBUG: Log detection details for first 10 frames to diagnose OCSort
        if frame_id <= 10:
            for idx, d in enumerate(detections):
                logger.debug(f"  [Frame {frame_id}] Det#{idx}: bbox=({d[0]:.0f},{d[1]:.0f},{d[2]:.0f},{d[3]:.0f}) score={d[4]:.4f} class={d[5]}")

        # 2. Run Tracking (C++ OCSort) — receives ALL raw detections, just like Python legacy
        detections_np = np.array(detections, dtype=np.float32)
        img_size = (frame_meta.source_frame_width, frame_meta.source_frame_height)
        tracked_objects = self.tracking_service.update(detections_np, img_size, img_size)
        
        # DEBUG: Log OCSort output
        if frame_id <= 10:
            logger.debug(f"  [Frame {frame_id}] OCSort returned {len(tracked_objects)} tracks")
        
        # 3. Apply Geometric Filter AFTER OCSort (exactly like Python legacy code!)
        #    Legacy Python (tracking_service.py dòng 184-191):
        #      for t in targets:
        #          tlwh = [t[0], t[1], t[2]-t[0], t[3]-t[1]]
        #          vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
        #          if tlwh[2]*tlwh[3] > min_box_area and not vertical:
        #              bboxes.append(xyxy)
        ar_thresh = self.tracking_service.config.aspect_ratio_thresh
        area_thresh = self.tracking_service.config.min_box_area
        
        bboxes_xyxy = []
        track_ids = []
        filtered_tracked = []
        
        if len(tracked_objects) > 0:
            for t in tracked_objects:
                x1, y1, x2, y2 = t[:4]
                tid = int(t[4])
                w = x2 - x1
                h = y2 - y1
                
                # Exact same formula as Python legacy: vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
                vertical = w / (h + 1e-6) > ar_thresh
                
                # Exact same condition: if tlwh[2]*tlwh[3] > min_box_area and not vertical
                if w * h > area_thresh and not vertical:
                    bboxes_xyxy.append([float(x1), float(y1), float(x2), float(y2)])
                    track_ids.append(tid)
                    filtered_tracked.append(t)
            
            # Inject ONLY filtered OCSort targets as new NvDsObjectMeta instances
            if filtered_tracked:
                self._sync_track_ids(frame_meta, batch_meta, np.array(filtered_tracked))

        # 4. Run Counting Service
        self.counting_service.update(bboxes_xyxy, track_ids)

        # 5. Run Mapping Service
        video_size = (frame_meta.source_frame_width, frame_meta.source_frame_height)
        map_size = self.mapping_service._map_size
        map_coords = self.mapping_service.project_bboxes(
            bboxes=bboxes_xyxy, 
            frame_size=video_size,
            map_size=map_size,
        )

        # 6. Construct Objects List for Backend
        objects = []
        for bbox, tid, coords in zip(bboxes_xyxy, track_ids, map_coords):
            objects.append(
                ObjectDetection(
                    track_id=tid,
                    bbox=bbox,
                    coordinates_2D=[float(coords[0]), float(coords[1])]
                )
            )

        # 7. Calculate Alert Status
        current_people = self.counting_service.current_people
        alert = "info"
        if current_people >= self.critical_threshold:
            alert = "critical"
        elif current_people >= self.warning_threshold:
            alert = "warning"

        # 7.5. Calculate FPS
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.fps_start_time
            self.current_fps = 30.0 / elapsed if elapsed > 0 else 0.0
            self.fps_start_time = time.time()
            
            # Print to log
            logger.info(f"[Probe] Frame #{frame_id} | FPS: {self.current_fps:.1f} | Detections: {len(detections)} | Tracked: {len(tracked_objects)} | In: {self.counting_service.count_in} | Out: {self.counting_service.count_out} | Occ: {current_people}")

        # 8. Create FrameData
        frame_data = FrameData(
            timestamp=timestamp,
            count_in=self.counting_service.count_in,
            count_out=self.counting_service.count_out,
            occupancy=current_people,
            height_frame=video_size[1],
            width_frame=video_size[0],
            height_2D=self.mapping_service._map_size[1] if self.mapping_service._map_size else 0,
            width_2D=self.mapping_service._map_size[0] if self.mapping_service._map_size else 0,
            objects=objects,
            alert=alert,
            fps=self.current_fps if self.current_fps > 0 else None
        )

        # 9. Publish results
        self.publisher.send_frame(frame_data)

        # 10. Draw OSD metadata (Counting Line, Counts)
        self._draw_osd(frame_meta, batch_meta)

    def _sync_track_ids(self, frame_meta, batch_meta, tracked_objects):
        """Update NvDsObjectMeta with OCSort Track IDs and per-ID colors."""
        import pyds
        
        # In the original python code, OCSort's smoothed tracking `targets` were drawn directly,
        # totally bypassing the raw YOLO boxes. Since raw YOLO boxes are already hidden 
        # (border_width=0) in DeepStream, we allocate BRAND NEW DeepStream metadata 
        # objects using the exact smoothed Kalman filter coordinates from OCSort.
        
        for t_box in tracked_objects:
            tid = int(t_box[4])
            x1, y1, x2, y2 = t_box[:4]
            width = x2 - x1
            height = y2 - y1
            
            # Avoid invalid geometries
            if width <= 0 or height <= 0:
                continue
                
            # 1. Acquire new object meta from batch
            obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
            
            # 2. Fill basic info
            obj_meta.unique_component_id = 99  # Custom ID for injected tracks
            obj_meta.confidence = 1.0
            obj_meta.class_id = 0
            obj_meta.object_id = tid
            
            # 3. Fill bounding box coordinates
            rect = obj_meta.rect_params
            rect.left = float(x1)
            rect.top = float(y1)
            rect.width = float(width)
            rect.height = float(height)
            
            # 4. Fill UI styling
            color = self.color_palette[tid % len(self.color_palette)]
            rect.border_color.set(*color)
            rect.border_width = 3
            
            # 5. Fill Text Display properties
            txt = obj_meta.text_params
            txt.display_text = f"ID:{tid}"
            txt.x_offset = int(rect.left)
            txt.y_offset = int(max(0, rect.top - 20))  # Position text above the box
            txt.font_params.font_name = "Serif"
            txt.font_params.font_size = 12
            txt.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            txt.set_bg_clr = 1
            txt.text_bg_clr.set(color[0], color[1], color[2], 0.6)
            
            # 6. Attach the new pristine tracking box back to the frame
            pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)

    def _draw_osd(self, frame_meta, batch_meta):
        """Inject custom lines and text for nvdsosd."""
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        
        # 1. Draw Counting Line
        line = self.counting_service.line_counter
        if line:
            l_params = display_meta.line_params[0]
            l_params.x1 = int(line.start_point[0])
            l_params.y1 = int(line.start_point[1])
            l_params.x2 = int(line.end_point[0])
            l_params.y2 = int(line.end_point[1])
            l_params.line_width = 3
            l_params.line_color.set(*self.line_color)
            display_meta.num_lines = 1

        # 2. Draw Counts Text
        txt_params = display_meta.text_params[0]
        in_c = self.counting_service.count_in
        out_c = self.counting_service.count_out
        occ = self.counting_service.current_people
        txt_params.display_text = f"In: {in_c} | Out: {out_c} | Occ: {occ}"
        txt_params.x_offset = 20
        txt_params.y_offset = 40
        txt_params.font_params.font_name = "Serif"
        txt_params.font_params.font_size = 15
        txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        txt_params.set_bg_clr = 1
        txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.5)
        display_meta.num_labels = 1
        
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    def _send_empty_frame(self, frame_meta, timestamp):
        """Send empty metadata if no detections present."""
        frame_data = FrameData(
            timestamp=timestamp,
            count_in=self.counting_service.count_in,
            count_out=self.counting_service.count_out,
            occupancy=self.counting_service.current_people,
            height_frame=frame_meta.source_frame_height,
            width_frame=frame_meta.source_frame_width,
            objects=[],
            alert="none"
        )
        self.publisher.send_frame(frame_data)
