"""
Runtime statistics for DeepStream pipeline sources.

Tracks per-source FPS, detections, tracks, and frame counters for terminal logs
and on-video overlays.
"""

import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from loguru import logger


@dataclass(frozen=True)
class SourceStatsSnapshot:
    """Immutable snapshot of runtime statistics for one source."""

    source_id: int
    frame_id: int
    fps: float
    detections: int
    tracks: int
    processed_frames: int


class RuntimeStatsManager:
    """Tracks and logs per-source runtime statistics."""

    def __init__(self, source_ids: Iterable[int], log_interval_frames: int = 30) -> None:
        """Initialize counters for all configured source IDs."""
        self.log_interval_frames = log_interval_frames
        self.frame_counts: Dict[int, int] = {source_id: 0 for source_id in source_ids}
        self.fps_start_times: Dict[int, float] = {source_id: time.time() for source_id in source_ids}
        self.current_fps: Dict[int, float] = {source_id: 0.0 for source_id in source_ids}
        self.latest_snapshots: Dict[int, SourceStatsSnapshot] = {
            source_id: SourceStatsSnapshot(
                source_id=source_id,
                frame_id=0,
                fps=0.0,
                detections=0,
                tracks=0,
                processed_frames=0,
            )
            for source_id in source_ids
        }

    def update(self, source_id: int, frame_id: int, detections: int, tracks: int) -> SourceStatsSnapshot:
        """Update source statistics and return the latest snapshot."""
        self.frame_counts[source_id] = self.frame_counts.get(source_id, 0) + 1
        processed_frames = self.frame_counts[source_id]

        fps = self.current_fps.get(source_id, 0.0)
        if processed_frames % self.log_interval_frames == 0:
            now = time.time()
            elapsed = now - self.fps_start_times[source_id]
            fps = self.log_interval_frames / elapsed if elapsed > 0 else 0.0
            self.current_fps[source_id] = fps
            self.fps_start_times[source_id] = now

        snapshot = SourceStatsSnapshot(
            source_id=source_id,
            frame_id=frame_id,
            fps=fps,
            detections=detections,
            tracks=tracks,
            processed_frames=processed_frames,
        )
        self.latest_snapshots[source_id] = snapshot

        if processed_frames % self.log_interval_frames == 0:
            self._log_stats(snapshot)

        return snapshot

    def get_snapshot(self, source_id: int) -> Optional[SourceStatsSnapshot]:
        """Return latest stats for a source if available."""
        return self.latest_snapshots.get(source_id)

    def _log_stats(self, snapshot: SourceStatsSnapshot) -> None:
        """Log compact per-source and aggregate stats."""
        total_fps = sum(item.fps for item in self.latest_snapshots.values())
        total_detections = sum(item.detections for item in self.latest_snapshots.values())
        total_tracks = sum(item.tracks for item in self.latest_snapshots.values())

        logger.info(
            "[Stats] "
            f"source={snapshot.source_id} | "
            f"frame={snapshot.frame_id} | "
            f"fps={snapshot.fps:.1f} | "
            f"detections={snapshot.detections} | "
            f"tracks={snapshot.tracks} | "
            f"total_fps={total_fps:.1f} | "
            f"total_detections={total_detections} | "
            f"total_tracks={total_tracks}"
        )
