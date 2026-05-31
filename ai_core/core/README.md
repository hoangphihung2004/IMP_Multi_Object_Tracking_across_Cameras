# Core Module

This directory contains the foundational components and centralized management logic for the Jetson Hybrid AI Pipeline.

## Overview

The `core` module acts as the "brain" of the application, ensuring that all services (Detection, Tracking, Counting, and Mapping) operate with consistent, validated, and strictly typed data. Its primary responsibility is managing the system's global state and configuration.

## Key Components

### 1. Configuration Management (`config.py`)

The pipeline utilizes a **Pydantic-based** configuration system. This approach provides several key advantages:
- **Strict Typing**: Ensures all configuration parameters (e.g., thresholds, model paths, hardware flags) match their expected types (int, float, bool).
- **Auto-Validation**: Automatically validates JSON values upon loading, preventing runtime crashes caused by malformed configuration files.
- **Single Source of Truth**: Python-side configurations are used to dynamically generate C++ plugin configs (like `nvinfer`'s `config_infer.txt`), ensuring full control from a single point.

#### Available Config Models:
- **`SystemConfig`**: Hardware settings (GPU/CPU), TensorRT toggles, and worker thread counts.
- **`YOLOXConfig`**: Comprehensive parameters for the YOLOX detector, including NMS thresholds, confidence levels, and architecture-specific multipliers.
- **`OCSortConfig`**: Detailed tracking logic parameters such as `max_age`, `min_hits`, `iou_thresh`, and `inertia`.
- **`CountingConfig`**: Geometry data for line-crossing and zone-counting logic.
- **`MappingConfig`**: Homography matrices and camera-to-map coordinate correspondences.

## Usage

To load a configuration within any service, use the `ConfigManager` utility:

```python
from core.config import ConfigManager

# Load specific configurations
yolo_cfg = ConfigManager.get_yolox_config("config/yolox_config.json")
ocsort_cfg = ConfigManager.get_ocsort_config("config/ocsort_config.json")

print(f"Current detection threshold: {yolo_cfg.conf_thresh}")
```

## Directory Structure
- `config.py`: Implementation of Pydantic models and the `ConfigManager`.
- `README.md`: Module documentation (this file).
