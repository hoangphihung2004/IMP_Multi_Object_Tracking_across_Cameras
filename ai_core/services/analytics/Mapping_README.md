# MappingService: Camera-to-Map Projection

## 1. Responsibilities
- Project camera pixel coordinates to a 2D floor plan using a **Homography matrix**.
- Handle **runtime resolution scaling** between the labeling frame size and the live camera frame size.
- Support **hot-reload** of calibration points at runtime via WebSocket settings updates.

## 2. Input / Output

### Input: `project_bbox(bbox, current_frame_size, current_map_size)`
| Parameter | Type | Format | Source |
|---|---|---|---|
| `bbox` | `List[float]` | `[x1, y1, x2, y2]` | From `TrackingService.update()` |
| `current_frame_size` | `Tuple[int, int]` | `(1920, 1080)` | Live camera resolution |
| `current_map_size` | `Tuple[int, int]` | `(1000, 1000)` | Runtime map display size |

### Output
```python
[map_x, map_y]  # e.g. [456.7, 321.2] — position on the 2D floor plan
```

### Batch Input: `project_bboxes(bboxes, frame_size, map_size)`
Returns: `List[Tuple[float, float]]` — one map coordinate per bbox.

## 3. Core Algorithm

1. **Foot-Point Extraction**: `foot = ((x1+x2)/2, y2)` — bottom-center of the bounding box represents where the person's feet touch the ground.
2. **Frame Scaling**: Scale from runtime camera resolution to the resolution used during labeling: `x_scaled = x * cam_w / cur_w`.
3. **Homography Projection**: Apply `cv2.perspectiveTransform()` with the 3×3 Homography matrix computed via `cv2.findHomography(RANSAC, 5.0)`.
4. **Map Scaling**: Scale the projected point from labeling map size to runtime map display size.

## 4. Configuration (`mapping_config.json`)
```json
{
    "image_size": {"width": 1920, "height": 1080},
    "map_size": {"width": 1000, "height": 1000},
    "correspondences": [
        {"label": "Point1", "camera": [100, 200], "map": [50, 60]},
        {"label": "Point2", "camera": [300, 400], "map": [150, 160]},
        ...
    ]
}
```
> Minimum 4 correspondence points required for RANSAC.

## 5. Properties
| Property | Returns | Description |
|---|---|---|
| `is_ready` | `bool` | Whether the homography matrix is valid |
| `num_correspondences` | `int` | Number of calibration point pairs |
| `num_inliers` | `int` | RANSAC inliers used in estimation |

## 6. Hot-Reload
```python
mapping_service.update_mapping(
    camera_points=[[100, 200], [300, 400], [500, 600], [700, 800]],
    map_points=[[50, 60], [150, 160], [250, 260], [350, 360]],
    camera_size=(1920, 1080),
    map_size=(1000, 1000),
)
```
