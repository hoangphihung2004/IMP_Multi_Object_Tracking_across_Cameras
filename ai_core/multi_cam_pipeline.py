import os
import cv2
import threading
import queue
import numpy as np
from services.tracking_service import TrackingService
from services.embedding_service import EmbeddingService
from services.reid_service import ReIDService
from config.data_config import OCSortConfig

PANEL_BG_COLOR = (20, 20, 20)


def resize_with_padding(frame, target_width, target_height):
    """Resize a frame to fit inside a fixed panel without changing aspect ratio."""
    panel = np.full((target_height, target_width, 3), PANEL_BG_COLOR, dtype=np.uint8)

    height, width = frame.shape[:2]
    scale = min(target_width / width, target_height / height)
    resized_width = int(width * scale)
    resized_height = int(height * scale)
    resized = cv2.resize(frame, (resized_width, resized_height))

    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2
    panel[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized
    return panel


def draw_panel_title(frame, title):
    cv2.rectangle(frame, (0, 0), (180, 38), (0, 0, 0), -1)
    cv2.putText(frame, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def make_combined_frame(frame1, frame2, panel_width, panel_height):
    if frame1 is None:
        left = np.full((panel_height, panel_width, 3), PANEL_BG_COLOR, dtype=np.uint8)
    else:
        left = resize_with_padding(frame1, panel_width, panel_height)

    if frame2 is None:
        right = np.full((panel_height, panel_width, 3), PANEL_BG_COLOR, dtype=np.uint8)
    else:
        right = resize_with_padding(frame2, panel_width, panel_height)

    draw_panel_title(left, "CAM 1")
    draw_panel_title(right, "CAM 2")
    return cv2.hconcat([left, right])


class CameraStream:
    def __init__(self, camera_id, video_path, tracker_overrides=None):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        tracker_args = OCSortConfig("config/tracking_config.json")
        if tracker_overrides:
            for key, value in tracker_overrides.items():
                if not hasattr(tracker_args, key):
                    raise ValueError(f"Unknown tracker override for {camera_id}: {key}")
                setattr(tracker_args, key, value)

        self.tracker = TrackingService(args=tracker_args)
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.stopped = False
        self.finished = False

    def read_frames(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            self.frame_queue.put(frame)
        self.cap.release()

def process_camera(cam_stream, embedder, reid_service):
    frame_count = 0
    try:
        while not cam_stream.stopped or not cam_stream.frame_queue.empty():
            try:
                frame = cam_stream.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # 1. Detection & Tracking
            frame_count += 1
            bboxes_tlwh, local_ids, scores, _ = cam_stream.tracker.predict_with_scores(
                frame_id=frame_count,
                frame=frame,
                box_type="tlwh"
            )

            bboxes = []
            for bbox in bboxes_tlwh:
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                bboxes.append([x1, y1, x2, y2])

            # 2. Tracklet-level ReID / Global matching
            global_ids = reid_service.update(
                camera_id=cam_stream.camera_id,
                frame_id=frame_count,
                frame=frame,
                local_track_ids=local_ids,
                bboxes_xyxy=bboxes,
                scores=scores,
                embedder=embedder
            )

            # 4. Visualization (replace local ID with global ID)
            display_frame = frame.copy()
            for bbox, local_id, gid in zip(bboxes, local_ids, global_ids):
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"G_ID:{gid}" if gid is not None else f"L_ID:{int(local_id)}"
                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cam_stream.result_queue.put(display_frame)
    finally:
        cam_stream.finished = True

def main():
    video1_path = "videos/cam131.mp4"
    video2_path = "videos/cam132.mp4"

    print("Initializing services...")
    embedder = EmbeddingService(model_name='osnet_x1_0', device='cpu') # Use cpu or cuda
    reid_service = ReIDService(
        tracklet_window_size=10,
        min_window_samples=6,
        min_tracklet_frames=30,
        min_detection_conf=0.50,
        match_threshold=0.56,
        match_margin=0.01,
        association_batch_size=2,
        debug=True
    )

    try:
        cam1 = CameraStream("cam1", video1_path)
        cam2 = CameraStream(
            "cam2",
            video2_path,
            tracker_overrides={
                "track_thresh": 0.60,
                "iou_thresh": 0.20,
                "use_byte": True,
                "max_age": 60,
                "min_hits": 2,
            }
        )
    except ValueError as e:
        print(e)
        return

    print("Starting processing threads...")
    # Read threads
    t_read1 = threading.Thread(target=cam1.read_frames)
    t_read2 = threading.Thread(target=cam2.read_frames)

    # Process threads
    t_proc1 = threading.Thread(target=process_camera, args=(cam1, embedder, reid_service))
    t_proc2 = threading.Thread(target=process_camera, args=(cam2, embedder, reid_service))

    t_read1.start()
    t_read2.start()
    t_proc1.start()
    t_proc2.start()

    os.makedirs("output", exist_ok=True)

    panel_width = max(cam1.width, cam2.width)
    panel_height = max(cam1.height, cam2.height)
    output_path = "output/combined_result.mp4"
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        min(cam1.fps, cam2.fps),
        (panel_width * 2, panel_height)
    )

    if not writer.isOpened():
        cam1.stopped = True
        cam2.stopped = True
        raise RuntimeError("Cannot create output video writer")

    written1 = 0
    written2 = 0
    combined_written = 0
    pending_frame1 = None
    pending_frame2 = None

    print(f"Writing combined output video: {output_path}")
    last_log_total = 0
    while True:
        cam1_done = cam1.finished and cam1.result_queue.empty()
        cam2_done = cam2.finished and cam2.result_queue.empty()

        if pending_frame1 is None and not cam1_done:
            try:
                pending_frame1 = cam1.result_queue.get(timeout=0.1)
                written1 += 1
            except queue.Empty:
                pass

        if pending_frame2 is None and not cam2_done:
            try:
                pending_frame2 = cam2.result_queue.get(timeout=0.1)
                written2 += 1
            except queue.Empty:
                pass

        if pending_frame1 is not None and pending_frame2 is not None:
            combined = make_combined_frame(pending_frame1, pending_frame2, panel_width, panel_height)
            writer.write(combined)
            combined_written += 1
            pending_frame1 = None
            pending_frame2 = None
        elif pending_frame1 is not None and cam2_done:
            combined = make_combined_frame(pending_frame1, None, panel_width, panel_height)
            writer.write(combined)
            combined_written += 1
            pending_frame1 = None
        elif pending_frame2 is not None and cam1_done:
            combined = make_combined_frame(None, pending_frame2, panel_width, panel_height)
            writer.write(combined)
            combined_written += 1
            pending_frame2 = None

        if cam1_done and cam2_done and pending_frame1 is None and pending_frame2 is None:
            break

        total_written = written1 + written2
        if total_written - last_log_total >= 60:
            print(
                f"Written frames | combined: {combined_written} | "
                f"cam1: {written1}/{cam1.total_frames} | cam2: {written2}/{cam2.total_frames}"
            )
            last_log_total = total_written

    writer.release()
    reid_db_path = "output/reid_vectors.npz"
    reid_service.save_database(reid_db_path)
    t_read1.join()
    t_read2.join()
    t_proc1.join()
    t_proc2.join()
    print(f"Pipeline finished. Output saved to: {output_path}")
    print(f"ReID vector database saved to: {reid_db_path}")

if __name__ == "__main__":
    main()
