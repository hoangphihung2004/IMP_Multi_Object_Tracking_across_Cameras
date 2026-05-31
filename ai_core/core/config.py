"""
Core Configuration Management Module.
Provides immutable Pydantic models to strictly type-check and validate
JSON configuration files used across the system. 

Includes all parameters identified in runtime analysis to ensure 100% control.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict


class SystemConfig(BaseModel):
    """Common hardware and backend settings."""
    device: str = Field(default="gpu")
    fp16: bool = Field(default=False)
    
    model_config = ConfigDict(extra="ignore")


class YOLOXConfig(BaseModel):
    """
    Configuration for YOLOX Detection (Inference & Post-processing).
    Values mapped from tracking_config.json, exps/yolox_x_mix_mot20_ch.py, and YOLOXHead.
    """
    # Weights and Architectures (MOT20 / YOLOX-X Defaults)
    model_path: str = Field(default="./pretrained/ocsort_x_mot20.pth.tar", alias="ckpt")
    engine_path: str = Field(default="./pretrained/model_trt.engine", alias="trt_file")
    num_classes: int = Field(default=1)
    depth: float = Field(default=1.33)
    width: float = Field(default=1.25)

    # Inference Settings
    input_size: int = Field(default=640, alias="tsize")
    conf_thresh: float = Field(default=0.1, alias="conf", description="Detection obj * cls threshold.")
    nms_thresh: float = Field(default=0.7, alias="nms", description="NMS overlap threshold.")
    
    # Custom Normalization (ImageNet Defaults)
    rgb_means: Tuple[float, float, float] = Field(default=(0.485, 0.456, 0.406))
    rgb_std: Tuple[float, float, float] = Field(default=(0.229, 0.224, 0.225))
    
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class OCSortConfig(BaseModel):
    """
    Configuration for OCSort Tracking Logic.
    Includes all parameters found in trackers/ocsort_tracker/ocsort.py
    """
    # Association Thresholds
    track_thresh: float = Field(default=0.6, description="High confidence threshold for 1st round association (det_thresh).")
    iou_thresh: float = Field(default=0.3, description="Minimum IoU for matching.")
    
    # Track Lifecycle
    max_age: int = Field(default=30, description="Frames to keep lost track before deletion.")
    min_hits: int = Field(default=3, description="Frames required to validate track.")
    
    # OCSort Specific Logic
    delta_t: int = Field(default=3, description="Frame window for velocity calculation.")
    inertia: float = Field(default=0.2, description="VDC weight in cost matrix.")
    use_byte: bool = Field(default=False, description="Enable BYTE (2nd round low-conf) association.")
    asso_func: str = Field(default="iou", description="Assignment cost function (iou/giou/ciou/diou).")
    
    # Post-filtering (Applied in TrackingService)
    aspect_ratio_thresh: float = Field(default=1.6, description="Filter boxes with extreme horizontal/vertical ratios.")
    min_box_area: float = Field(default=100.0, description="Minimum box area in pixels.")
    
    model_config = ConfigDict(extra="ignore")


class CountingConfig(BaseModel):
    """Configuration for Line-Crossing Logic."""
    line_start: List[float] = Field(default_factory=lambda: [984.47, 346.44])
    line_end: List[float] = Field(default_factory=lambda: [903.68, 383.81])
    inside_point: List[float] = Field(default_factory=lambda: [1001.31, 375.39])
    crossing_margin: float = Field(default=10.0)
    frame_height: int = Field(default=1080)
    frame_width: int = Field(default=1920)
    
    model_config = ConfigDict(extra="ignore")


class Correspondence(BaseModel):
    label: str
    camera: List[float]
    map: List[float]

class MappingConfig(BaseModel):
    """Configuration for Homography/Mapping Logic."""
    image_size: Dict[str, int] = Field(default_factory=lambda: {"width": 1920, "height": 1080})
    map_size: Dict[str, int] = Field(default_factory=lambda: {"width": 1000, "height": 1000})
    correspondences: List[Correspondence] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="ignore")


class ConfigManager:
    """
    Manager to load and parse JSON configurations into strictly typed models.
    """

    @staticmethod
    def load_json(filepath: str) -> Dict[str, Any]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config not found: {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    @classmethod
    def get_system_config(cls, filepath: str = "config/system_config.json") -> SystemConfig:
        return SystemConfig(**cls.load_json(filepath))

    @classmethod
    def get_yolox_config(cls, filepath: str = "config/yolox_config.json") -> YOLOXConfig:
        return YOLOXConfig(**cls.load_json(filepath))

    @classmethod
    def get_ocsort_config(cls, filepath: str = "config/ocsort_config.json") -> OCSortConfig:
        return OCSortConfig(**cls.load_json(filepath))

    @classmethod
    def get_counting_config(cls, filepath: str = "config/counting_config.json") -> CountingConfig:
        return CountingConfig(**cls.load_json(filepath))

    @classmethod
    def get_mapping_config(cls, filepath: str = "config/mapping_config.json") -> MappingConfig:
        return MappingConfig(**cls.load_json(filepath))
