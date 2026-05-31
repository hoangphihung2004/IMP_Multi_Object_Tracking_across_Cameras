"""
Aura Analytics - Main Entry Point.

Local DeepStream tracking pipeline for Windows/WSL2 Docker.
"""

import signal
import sys
from typing import Dict

from loguru import logger

try:
    from dotenv import load_dotenv

    load_dotenv("config/.env")
    load_dotenv()
except ImportError:
    pass

from core.config import ConfigManager
from core.model_converter import check_and_convert_models
from services.analytics.tracking_service import TrackingService
from services.pipeline.manager import PipelineManager
from services.pipeline.probe import AnalyticsProbe


class AuraAnalyticsApp:
    """Ứng dụng chính cho local DeepStream tracking runtime."""

    def __init__(self) -> None:
        """Load config và chuẩn bị pipeline manager."""
        self.pipeline_manager = PipelineManager()
        self.services: Dict[str, object] = {}

        self.system_cfg = ConfigManager.get_system_config()
        self.yolox_cfg = ConfigManager.get_yolox_config()
        self.ocsort_cfg = ConfigManager.get_ocsort_config()
        self.sources_cfg = ConfigManager.get_sources_config()
        self.pipeline_cfg = ConfigManager.get_pipeline_config()

        ConfigManager.validate_runtime_config(self.sources_cfg, self.pipeline_cfg)
        ConfigManager.log_loaded_config(
            system_cfg=self.system_cfg,
            yolox_cfg=self.yolox_cfg,
            ocsort_cfg=self.ocsort_cfg,
            sources_cfg=self.sources_cfg,
            pipeline_cfg=self.pipeline_cfg,
        )

    def _setup_services(self) -> None:
        """Khởi tạo tracking service riêng cho từng source."""
        logger.info("[App] Initializing tracking services...")
        self.services["tracking"] = {
            source.id: TrackingService(config=self.ocsort_cfg)
            for source in self.sources_cfg.sources
        }
        logger.success(f"[App] Initialized {len(self.services['tracking'])} tracker instances.")

    def _setup_pipeline(self) -> None:
        """Dựng DeepStream pipeline và attach tracking probe."""
        probe = AnalyticsProbe(
            tracking_services=self.services["tracking"],
            tracking_config=self.ocsort_cfg,
            osd_config=self.pipeline_cfg.osd,
        )

        self.pipeline_manager.build_pipeline(
            sources_cfg=self.sources_cfg,
            pipeline_cfg=self.pipeline_cfg,
            yolox_cfg=self.yolox_cfg,
        )
        self.pipeline_manager.attach_probe(probe)

    def run(self) -> None:
        """Main execution entry point."""
        logger.info("=== Starting Local DeepStream Tracking Application ===")

        try:
            check_and_convert_models(
                yolox_cfg=self.yolox_cfg,
                system_cfg=self.system_cfg,
            )
            self._setup_services()
            self._setup_pipeline()
            self.pipeline_manager.run()
        except KeyboardInterrupt:
            logger.warning("[App] Interrupted by user (SIGINT).")
        except Exception as exc:
            logger.exception(f"[App] Fatal error during execution: {exc}")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Graceful resource cleanup."""
        logger.info("[App] Shutdown sequence initiated...")
        self.pipeline_manager.stop()
        logger.success("=== Local DeepStream Tracking Shutdown Complete ===")


def signal_handler(sig, frame) -> None:
    """Bridge OS signals to the application shutdown."""
    logger.warning(f"[System] Signal {sig} received.")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app = AuraAnalyticsApp()
    app.run()
