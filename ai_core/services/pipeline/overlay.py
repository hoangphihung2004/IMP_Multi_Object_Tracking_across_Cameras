"""
On-video runtime overlay rendering for DeepStream metadata.
"""

import pyds

from core.config import OsdConfig
from services.pipeline.runtime_stats import SourceStatsSnapshot


class RuntimeOverlayRenderer:
    """Renders runtime statistics as NvDsDisplayMeta text overlays."""

    def __init__(self, osd_config: OsdConfig) -> None:
        """Store OSD font configuration for overlay rendering."""
        self.osd_config = osd_config

    def render(self, batch_meta, frame_meta, stats: SourceStatsSnapshot) -> None:
        """Attach one runtime stats text overlay to a frame."""
        if not self.osd_config.enabled:
            return

        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1

        text_params = display_meta.text_params[0]
        text_params.display_text = self._format_text(stats)
        text_params.x_offset = 20
        text_params.y_offset = 30
        text_params.font_params.font_name = self.osd_config.font_name
        text_params.font_params.font_size = max(12, self.osd_config.font_size)
        text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        text_params.set_bg_clr = 1
        text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.65)

        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

    @staticmethod
    def _format_text(stats: SourceStatsSnapshot) -> str:
        """Format compact runtime stats for video overlay."""
        return (
            f"CAM {stats.source_id} | "
            f"FPS {stats.fps:.1f} | "
            f"DET {stats.detections} | "
            f"TRACKS {stats.tracks} | "
            f"FRAME {stats.frame_id}"
        )
