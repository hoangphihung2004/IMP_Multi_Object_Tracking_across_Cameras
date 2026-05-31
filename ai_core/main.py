"""
Aura Analytics - Main Entry Point
High-performance AI detection and tracking pipeline for NVIDIA Jetson.
"""

import os
import sys
import signal
from typing import Tuple
from loguru import logger

# Load environment logic
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.config import ConfigManager
from core.model_converter import check_and_convert_models
from core.camera_validator import CameraValidator
from services.analytics.tracking_service import TrackingService
from services.analytics.counting_service import CountingService
from services.analytics.mapping_service import MappingService
from services.gateway.backend_gateway import ResultPublisher, SettingsSubscriber
from services.pipeline.manager import PipelineManager
from services.pipeline.probe import AnalyticsProbe


class AuraAnalyticsApp:
    """
    Main Application class to coordinate services and the DeepStream pipeline.
    """

    def __init__(self):
        self.pipeline_manager = PipelineManager()
        self.settings_sub = None
        self.services = {}
        
        # Load System & YOLOX configs early (used by model_converter + pipeline)
        self.system_cfg = ConfigManager.get_system_config()
        self.yolox_cfg = ConfigManager.get_yolox_config()
        
        # Configuration paths
        self.config_paths = {
            "tracking": "config/ocsort_config.json",
            "counting": "config/counting_config.json",
            "mapping": "config/mapping_config.json",
            "infer": os.getenv("CONFIG_INFER", "config/config_infer.txt")
        }

    def _setup_services(self):
        """Initialize all business logic services."""
        logger.info("[App] Initializing core services...")
        logger.info(f"[App] System: device={self.system_cfg.device}, fp16={self.system_cfg.fp16}")
        logger.info(f"[App] YOLOX:  engine={self.yolox_cfg.engine_path}, conf={self.yolox_cfg.conf_thresh}, nms={self.yolox_cfg.nms_thresh}")
        
        # 1. Load Configs via ConfigManager
        tracking_cfg = ConfigManager.get_ocsort_config(self.config_paths["tracking"])
        counting_cfg = ConfigManager.get_counting_config(self.config_paths["counting"])
        mapping_cfg = ConfigManager.get_mapping_config(self.config_paths["mapping"])

        # 2. Instantiate Services
        # Default 1080p, will sync with actual source resolution in probe
        init_size = (1920, 1080)
        
        self.services["tracking"] = TrackingService(config=tracking_cfg)
        self.services["counting"] = CountingService(counting_config=counting_cfg, current_frame_size=init_size)
        self.services["mapping"] = MappingService(config=mapping_cfg)
        
        self.services["publisher"] = ResultPublisher(
            endpoint=os.getenv("RESULTS_WS_URL"),
            bearer_token=os.getenv("API_BEARER_TOKEN")
        )
        
        # === Log all configs for debugging ===
        logger.info("=" * 60)
        logger.info("[Config] ===== LOADED CONFIGURATION =====")
        logger.info(f"[Config] config_infer: {self.config_paths['infer']}")
        logger.info(f"[Config] OCSort: track_thresh={tracking_cfg.track_thresh}, iou_thresh={tracking_cfg.iou_thresh}, "
                     f"max_age={tracking_cfg.max_age}, min_hits={tracking_cfg.min_hits}, "
                     f"delta_t={tracking_cfg.delta_t}, inertia={tracking_cfg.inertia}, "
                     f"use_byte={tracking_cfg.use_byte}, asso_func={tracking_cfg.asso_func}")
        logger.info(f"[Config] OCSort Post-filter: aspect_ratio_thresh={tracking_cfg.aspect_ratio_thresh}, "
                     f"min_box_area={tracking_cfg.min_box_area}")
        logger.info(f"[Config] Counting: line_start={counting_cfg.line_start}, line_end={counting_cfg.line_end}, "
                     f"inside_point={counting_cfg.inside_point}, margin={counting_cfg.crossing_margin}, "
                     f"frame={counting_cfg.frame_width}x{counting_cfg.frame_height}")
        logger.info(f"[Config] Mapping: image_size={mapping_cfg.image_size}, map_size={mapping_cfg.map_size}, "
                     f"correspondences={len(mapping_cfg.correspondences)}")
        logger.info(f"[Config] YOLOX: engine={self.yolox_cfg.engine_path}, input_size={self.yolox_cfg.input_size}, "
                     f"conf={self.yolox_cfg.conf_thresh}, nms={self.yolox_cfg.nms_thresh}")
        logger.info(f"[Config] System: device={self.system_cfg.device}, fp16={self.system_cfg.fp16}")
        logger.info(f"[Config] WS endpoint: {os.getenv('RESULTS_WS_URL', 'NOT SET')}")
        logger.info("=" * 60)
        
        logger.success("[App] Services initialized.")

    def _setup_pipeline(self):
        """Assemble the GStreamer pipeline and attach the analytics probe."""
        video_source = os.getenv("VIDEO_SOURCE") or os.getenv("INPUT_VIDEO_SOURCE", "/dev/video0")
        
        # 0. Validate camera/video source before building pipeline
        cam_info = CameraValidator.check(video_source)
        if not cam_info.is_valid:
            logger.critical(f"[App] Video source validation FAILED: {video_source}")
            for err in cam_info.errors:
                logger.error(f"  → {err}")
            raise RuntimeError(f"Invalid video source: {video_source}")
        
        # 1. Initialize Probe
        probe = AnalyticsProbe(
            tracking_service=self.services["tracking"],
            counting_service=self.services["counting"],
            mapping_service=self.services["mapping"],
            publisher=self.services["publisher"],
            warning_threshold=int(os.getenv("ALERT_WARNING_THRESH", 25)),
            critical_threshold=int(os.getenv("ALERT_CRITICAL_THRESH", 50))
        )

        # 2. Build Pipeline
        try:
            self.pipeline_manager.build_pipeline(
                source_uri=video_source,
                config_infer=self.config_paths["infer"],
                yolox_cfg=self.yolox_cfg,
            )
            self.pipeline_manager.attach_probe(probe)
        except Exception as e:
            logger.critical(f"[App] Failed to construct pipeline: {e}")
            raise

    def _setup_hot_reload(self):
        """Configure SettingsSubscriber for live updates from backend."""
        logger.info("[App] Starting SettingsSubscriber...")
        
        def on_mapping_update(data):
            logger.info("[App:HotReload] Applying new Mapping config...")
            try:
                # Parse correspondences from backend (list of {camera: [x,y], map: [x,y]})
                correspondences = data.get("correspondences", [])
                camera_pts = []
                map_pts = []
                for item in correspondences:
                    cam = item.get("camera")
                    mp = item.get("map")
                    if cam and mp:
                        camera_pts.append((float(cam[0]), float(cam[1])))
                        map_pts.append((float(mp[0]), float(mp[1])))

                # Basic normalization of backend data
                img_size = data.get("image_size", {})
                m_size = data.get("map_size", {})
                
                self.services["mapping"].update_mapping(
                    camera_points=camera_pts,
                    map_points=map_pts,
                    camera_size=(int(img_size.get("width", 1920)), int(img_size.get("height", 1080))),
                    map_size=(int(m_size.get("width", 1000)), int(m_size.get("height", 1000)))
                )
                logger.success("[App:HotReload] Mapping updated.")
            except Exception as e:
                logger.error(f"[App:HotReload] Mapping update failed: {e}")

        def on_counting_update(data):
            logger.info("[App:HotReload] Applying new Counting config...")
            try:
                self.services["counting"].update_config(
                    line_start=tuple(data.get("line_start")),
                    line_end=tuple(data.get("line_end")),
                    inside_point=tuple(data.get("inside_point")),
                    crossing_margin=data.get("crossing_margin", 10.0),
                    source_frame_size=(data.get("frame_width", 1920), data.get("frame_height", 1080)),
                    current_frame_size=(1920, 1080) # Sync with actual if possible
                )
                logger.success("[App:HotReload] Counting updated.")
            except Exception as e:
                logger.error(f"[App:HotReload] Counting update failed: {e}")

        self.settings_sub = SettingsSubscriber(
            on_mapping_update=on_mapping_update,
            on_counting_update=on_counting_update
        )
        self.settings_sub.start()

    def run(self):
        """Main execution entry point."""
        logger.info("=== Starting Aura Analytics Application ===")
        
        try:
            # 0. Auto-convert model if needed (PTH → ONNX → TensorRT)
            check_and_convert_models(
                yolox_cfg=self.yolox_cfg,
                system_cfg=self.system_cfg,
            )
            
            self._setup_services()
            self._setup_pipeline()
            self._setup_hot_reload()
            
            # Start the GStreamer Main Loop
            self.pipeline_manager.run()
            
        except KeyboardInterrupt:
            logger.warning("[App] Implementation interrupted by user (SIGINT).")
        except Exception as e:
            logger.exception(f"[App] Fatal error during execution: {e}")
        finally:
            self.shutdown()

    def shutdown(self):
        """Graceful resource cleanup."""
        logger.info("[App] Shutdown sequence initiated...")
        
        if self.settings_sub:
            self.settings_sub.stop()
            
        self.pipeline_manager.stop()
        
        logger.success("=== Aura Analytics Shutdown Complete ===")


def signal_handler(sig, frame):
    """Bridge OS signals to the Application shutdown."""
    logger.warning(f"[System] Signal {sig} received.")
    sys.exit(0)


if __name__ == "__main__":
    # Register OS signals for Docker/Process management
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app = AuraAnalyticsApp()
    app.run()
