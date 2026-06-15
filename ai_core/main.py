import os
import time
import cv2
import numpy as np
from datetime import datetime, timezone
from loguru import logger

from config.data_config import OCSortConfig
from services.tracking_service import TrackingService
from utils.visualize import sample_colors


def draw_tracking_results(frame, bboxes, track_ids, fps_info):
    """Draw bounding boxes and tracking info on frame."""
    vis_frame = frame.copy()

    # Draw each tracked object
    for bbox, track_id in zip(bboxes, track_ids):
        x1, y1, w, h = bbox
        x2 = int(x1 + w)
        y2 = int(y1 + h)
        x1, y1 = int(x1), int(y1)

        # Get color based on track_id
        color = sample_colors[int(track_id) % len(sample_colors)]

        # Draw bounding box
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color=color, thickness=2)

        # Draw track ID
        label = f"ID:{int(track_id)}"
        cv2.putText(vis_frame, label, (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw FPS and tracking count info
    info_text = f"FPS: {fps_info:.1f} | Tracks: {len(track_ids)}"
    cv2.putText(vis_frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return vis_frame


def main():
    logger.info("Starting tracking application...")

    # Paths
    video_path = "videos/demo.mp4"
    output_path = "output/result.mp4"
    os.makedirs("output", exist_ok=True)

    # Setup tracking service
    logger.info("Initializing tracking service...")
    tracking_cfg = OCSortConfig("config/tracking_config.json")
    tracking_service = TrackingService(args=tracking_cfg)
    logger.success("Tracking service initialized.")

    # Open video capture
    logger.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        raise RuntimeError(f"Cannot open video {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video info: {width}x{height}, {fps:.2f} FPS, {total_frames} frames")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        logger.error("Failed to create video writer")
        raise RuntimeError("Cannot create video writer")

    logger.info(f"Output will be saved to: {output_path}")

    # Processing loop
    frame_id = 0
    start_time = time.time()

    logger.info("Starting frame processing...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video reached")
                break

            frame_id += 1

            # Run tracking
            bboxes_tlwh, track_ids, avg_time = tracking_service.predict(
                frame_id=frame_id,
                frame=frame,
                box_type="tlwh"
            )

            # Calculate current FPS
            current_fps = 1.0 / max(1e-5, avg_time) if avg_time else fps

            # Draw tracking results
            vis_frame = draw_tracking_results(frame, bboxes_tlwh, track_ids, current_fps)

            # Write to output video
            video_writer.write(vis_frame)

            # Log progress every 30 frames
            if frame_id % 30 == 0:
                progress = (frame_id / total_frames) * 100 if total_frames > 0 else 0
                logger.info(
                    f"Frame {frame_id}/{total_frames} ({progress:.1f}%) | "
                    f"Tracks: {len(track_ids)} | FPS: {current_fps:.1f}"
                )

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")

    except Exception as e:
        logger.exception(f"Error during processing: {e}")
        raise

    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()

    # Final statistics
    end_time = time.time()
    total_time = end_time - start_time
    avg_fps = frame_id / total_time if total_time > 0 else 0

    logger.success(
        f"Processing completed!\n"
        f"  Total frames: {frame_id}\n"
        f"  Total time: {total_time:.2f}s\n"
        f"  Average FPS: {avg_fps:.2f}\n"
        f"  Output saved: {output_path}"
    )


if __name__ == "__main__":
    main()
