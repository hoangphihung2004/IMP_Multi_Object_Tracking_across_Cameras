# YOLOX Custom Bounding Box Parser

This C++ plugin provides a custom implementation of the `NvDsInferParseCustomFunc` for the YOLOX object detection model.

## Overview

YOLOX is an anchor-free detector that produces a dense output tensor (typically `1x8400x85`). This parser is responsible for:
1.  **Decoding**: Converting the model's grid-stride outputs into normalized pixel coordinates (`x1, y1, x2, y2`).
2.  **Filtering**: Applying detection thresholds (configured dynamically via Python/DeepStream).
3.  **Non-Maximum Suppression (NMS)**: Grouping overlapping predictions into a single object detection to remove duplicates.

## Key Features

- **Dynamic Thresholding**: Confidence and NMS thresholds are loaded from the DeepStream `config_infer.txt` file, which is managed by the Python `ConfigManager`.
- **C++ Performance**: NMS and coordinate transformations are computationally expensive in Python; this C++ implementation runs over 20x faster.
- **Integration**: Compiles into `libnvdsinfer_custombboxparser_yolox.so`, which is loaded by the `nvinfer` DeepStream element via the `custom-lib-path` property.

## Technical Specifications

- **Input Tensor Shape**: `(1, N, 85)` where `N` is the total number of anchor grids across all FPN levels.
- **Output**: Populates the `std::vector<NvDsInferParseObjectInfo>` list used to create `NvDsObjectMeta` in the DeepStream pipeline.
- **NMS Algorithm**: Standard intersection-over-union (IoU) based suppression with class-aware filtering.
