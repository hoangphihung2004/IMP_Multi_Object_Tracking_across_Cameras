"""
Core Configuration Management Module.

Provides immutable Pydantic models to strictly type-check and validate JSON
configuration files used by the local DeepStream tracking runtime.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SystemConfig(BaseModel):
    """Common hardware and runtime settings."""

    device: str = Field(default="gpu")
    fp16: bool = Field(default=False)

    model_config = ConfigDict(extra="ignore")


class YOLOXConfig(BaseModel):
    """
    Configuration for YOLOX detection inference and post-processing.

    Values are mapped from yolox_config.json and DeepStream nvinfer runtime.
    """

    model_path: str = Field(default="./pretrained/ocsort_x_mot20.pth.tar", alias="ckpt")
    engine_path: str = Field(default="./pretrained/model_trt.engine", alias="trt_file")
    num_classes: int = Field(default=1)
    depth: float = Field(default=1.33)
    width: float = Field(default=1.25)

    input_size: int = Field(default=640, alias="tsize")
    conf_thresh: float = Field(default=0.1, alias="conf", description="Detection obj * cls threshold.")
    nms_thresh: float = Field(default=0.7, alias="nms", description="NMS overlap threshold.")

    rgb_means: Tuple[float, float, float] = Field(default=(0.485, 0.456, 0.406))
    rgb_std: Tuple[float, float, float] = Field(default=(0.229, 0.224, 0.225))

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class OCSortConfig(BaseModel):
    """Configuration for OCSort tracking logic."""

    track_thresh: float = Field(default=0.6, description="High confidence threshold for first association.")
    iou_thresh: float = Field(default=0.3, description="Minimum IoU for matching.")

    max_age: int = Field(default=30, description="Frames to keep lost track before deletion.")
    min_hits: int = Field(default=3, description="Frames required to validate track.")

    delta_t: int = Field(default=3, description="Frame window for velocity calculation.")
    inertia: float = Field(default=0.2, description="VDC weight in cost matrix.")
    use_byte: bool = Field(default=False, description="Enable BYTE second-round low-confidence association.")
    asso_func: str = Field(default="iou", description="Assignment cost function: iou/giou/ciou/diou.")

    aspect_ratio_thresh: float = Field(default=1.6, description="Filter boxes with extreme horizontal ratio.")
    min_box_area: float = Field(default=100.0, description="Minimum box area in pixels.")

    model_config = ConfigDict(extra="ignore")


class SourceConfig(BaseModel):
    """A single local video source and its output path."""

    id: int = Field(..., ge=0)
    uri: str = Field(..., min_length=1)
    output: str = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid")

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, value: str) -> str:
        """Only local video files are supported in Windows local runtime."""
        if value.startswith(("rtsp://", "/dev/video")):
            raise ValueError("Windows local runtime only supports video files; RTSP/USB camera inputs are not supported.")
        return value


class SourcesConfig(BaseModel):
    """Collection of local video sources."""

    sources: List[SourceConfig] = Field(..., min_length=1)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "SourcesConfig":
        """Ensure source IDs are unique."""
        ids = [source.id for source in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate source ids: {ids}")
        return self

    def validate_files_exist(self, base_dir: str = ".") -> None:
        """Fail fast if input videos do not exist and create output directories."""
        for source in self.sources:
            input_path = Path(base_dir) / source.uri
            if not input_path.is_file():
                raise FileNotFoundError(f"Video source does not exist: {source.uri}")
            output_path = Path(base_dir) / source.output
            output_path.parent.mkdir(parents=True, exist_ok=True)


class MuxerConfig(BaseModel):
    """Configuration for nvstreammux."""

    width: int = Field(default=1920, gt=0)
    height: int = Field(default=1080, gt=0)
    batch_size: int = Field(default=2, gt=0)
    batched_push_timeout: int = Field(default=40000, ge=0)

    model_config = ConfigDict(extra="forbid")


class DecodeConfig(BaseModel):
    """Configuration for input parser/decoder elements."""

    parser: str = "h264parse"
    decoder: str = "nvv4l2decoder"

    model_config = ConfigDict(extra="forbid")


class InferConfig(BaseModel):
    """Configuration for nvinfer."""

    config_file: str = "config/config_infer.txt"

    model_config = ConfigDict(extra="forbid")


class OsdConfig(BaseModel):
    """Configuration for track box/text metadata rendered by nvdsosd."""

    enabled: bool = True
    bbox_border_width: int = Field(default=3, ge=0)
    font_name: str = "Serif"
    font_size: int = Field(default=12, gt=0)

    model_config = ConfigDict(extra="forbid")


class OutputConfig(BaseModel):
    """Configuration for output encoder/muxer/filesink branch."""

    codec: Literal["h264"] = "h264"
    encoder: str = "nvv4l2h264enc"
    bitrate: int = Field(default=4000000, gt=0)
    container: Literal["mp4"] = "mp4"
    muxer: str = "qtmux"
    parser: str = "h264parse"

    model_config = ConfigDict(extra="forbid")


class PipelineConfig(BaseModel):
    """Configuration for DeepStream/GStreamer pipeline."""

    muxer: MuxerConfig = Field(default_factory=MuxerConfig)
    decode: DecodeConfig = Field(default_factory=DecodeConfig)
    infer: InferConfig = Field(default_factory=InferConfig)
    osd: OsdConfig = Field(default_factory=OsdConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    stop_policy: Literal["all_eos"] = "all_eos"

    model_config = ConfigDict(extra="forbid")


class ConfigManager:
    """Manager that loads JSON configuration into strictly typed models."""

    @staticmethod
    def load_json(filepath: str) -> Dict[str, Any]:
        """Load a JSON file and return its dictionary content."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)

    @classmethod
    def get_system_config(cls, filepath: str = "config/system_config.json") -> SystemConfig:
        """Load system configuration."""
        return SystemConfig(**cls.load_json(filepath))

    @classmethod
    def get_yolox_config(cls, filepath: str = "config/yolox_config.json") -> YOLOXConfig:
        """Load YOLOX configuration."""
        return YOLOXConfig(**cls.load_json(filepath))

    @classmethod
    def get_ocsort_config(cls, filepath: str = "config/ocsort_config.json") -> OCSortConfig:
        """Load OCSort configuration."""
        return OCSortConfig(**cls.load_json(filepath))

    @classmethod
    def get_sources_config(cls, filepath: str = "config/sources_config.json") -> SourcesConfig:
        """Load sources configuration and validate local input files."""
        config = SourcesConfig(**cls.load_json(filepath))
        config.validate_files_exist(base_dir=".")
        return config

    @classmethod
    def get_pipeline_config(cls, filepath: str = "config/pipeline_config.json") -> PipelineConfig:
        """Load DeepStream pipeline configuration."""
        return PipelineConfig(**cls.load_json(filepath))

    @staticmethod
    def validate_runtime_config(sources_cfg: SourcesConfig, pipeline_cfg: PipelineConfig) -> None:
        """Validate cross-file runtime config relationships."""
        source_count = len(sources_cfg.sources)
        batch_size = pipeline_cfg.muxer.batch_size
        if source_count != batch_size:
            raise ValueError(f"Source count ({source_count}) must match muxer.batch_size ({batch_size}).")

    @staticmethod
    def log_loaded_config(
        *,
        system_cfg: SystemConfig,
        yolox_cfg: YOLOXConfig,
        ocsort_cfg: OCSortConfig,
        sources_cfg: SourcesConfig,
        pipeline_cfg: PipelineConfig,
    ) -> None:
        """Log all loaded runtime configuration values for debugging."""
        logger.info("═" * 60)
        logger.info("[Config] ===== LOADED CONFIGURATION =====")
        logger.info("[Config] sources_config.json")
        for source in sources_cfg.sources:
            logger.info(f"[Config]   sources[{source.id}].uri    = {source.uri}")
            logger.info(f"[Config]   sources[{source.id}].output = {source.output}")
        logger.info("[Config] pipeline_config.json")
        logger.info(f"[Config]   muxer.width                 = {pipeline_cfg.muxer.width}")
        logger.info(f"[Config]   muxer.height                = {pipeline_cfg.muxer.height}")
        logger.info(f"[Config]   muxer.batch_size            = {pipeline_cfg.muxer.batch_size}")
        logger.info(f"[Config]   muxer.batched_push_timeout  = {pipeline_cfg.muxer.batched_push_timeout}")
        logger.info(f"[Config]   decode.parser               = {pipeline_cfg.decode.parser}")
        logger.info(f"[Config]   decode.decoder              = {pipeline_cfg.decode.decoder}")
        logger.info(f"[Config]   infer.config_file           = {pipeline_cfg.infer.config_file}")
        logger.info(f"[Config]   osd.enabled                 = {pipeline_cfg.osd.enabled}")
        logger.info(f"[Config]   osd.bbox_border_width       = {pipeline_cfg.osd.bbox_border_width}")
        logger.info(f"[Config]   output.encoder              = {pipeline_cfg.output.encoder}")
        logger.info(f"[Config]   output.bitrate              = {pipeline_cfg.output.bitrate}")
        logger.info(f"[Config]   output.muxer                = {pipeline_cfg.output.muxer}")
        logger.info("[Config] yolox_config.json")
        logger.info(f"[Config]   engine_path                 = {yolox_cfg.engine_path}")
        logger.info(f"[Config]   model_path                  = {yolox_cfg.model_path}")
        logger.info(f"[Config]   input_size                  = {yolox_cfg.input_size}")
        logger.info(f"[Config]   conf_thresh                 = {yolox_cfg.conf_thresh}")
        logger.info(f"[Config]   nms_thresh                  = {yolox_cfg.nms_thresh}")
        logger.info("[Config] ocsort_config.json")
        logger.info(f"[Config]   track_thresh                = {ocsort_cfg.track_thresh}")
        logger.info(f"[Config]   iou_thresh                  = {ocsort_cfg.iou_thresh}")
        logger.info(f"[Config]   max_age                     = {ocsort_cfg.max_age}")
        logger.info(f"[Config]   min_hits                    = {ocsort_cfg.min_hits}")
        logger.info(f"[Config]   asso_func                   = {ocsort_cfg.asso_func}")
        logger.info("[Config] system_config.json")
        logger.info(f"[Config]   device                      = {system_cfg.device}")
        logger.info(f"[Config]   fp16                        = {system_cfg.fp16}")
        logger.info("═" * 60)
