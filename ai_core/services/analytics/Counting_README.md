# CountingService: Line-Crossing Counter

## 1. Responsibilities
- Count objects (people) entering and exiting across a user-defined counting line.
- Use a **state machine** per Track ID to ensure each crossing is counted exactly once.
- Support **hot-reload** of line geometry at runtime via WebSocket settings updates.

## 2. Input / Output

### Input: `update(bboxes, track_ids)`
| Parameter | Type | Format | Source |
|---|---|---|---|
| `bboxes` | `List[List[float]]` | `[[x1, y1, x2, y2], ...]` | From `TrackingService.update()` |
| `track_ids` | `List[int]` | `[5, 12, 3, ...]` | From `TrackingService.update()` |

### Output: `get_counts()`
```python
{"in": 15, "out": 8, "current": 7}
```

### Properties
| Property | Returns | Description |
|---|---|---|
| `count_in` | `int` | Total people entered |
| `count_out` | `int` | Total people exited |
| `current_people` | `int` | `max(0, in - out)` |

## 3. Core Algorithm: LineCounter

1. **Signed Distance**: For each object's foot-point `((x1+x2)/2, y2)`, compute the signed perpendicular distance to the counting line using the dot product with the line's normal vector.
2. **Margin Band**: If the distance falls within `±crossing_margin`, the object is in the "no-decision zone" and is ignored.
3. **State Transition**: Only when an object clearly moves from `outside → inside` (enter) or `inside → outside` (exit) is a count registered.

## 4. Configuration (`counting_config.json`)
```json
{
    "line_start": [984.47, 346.44],
    "line_end": [903.68, 383.81],
    "inside_point": [1001.31, 375.39],
    "crossing_margin": 10.0,
    "frame_width": 1920,
    "frame_height": 1080
}
```

## 5. Resolution Scaling (`_scale_geometry`)
Supports 3 modes:
- **Normalized** (`normalized=True`): Coordinates in `[0, 1]` range, scaled to `current_frame_size`.
- **Cross-Resolution**: Scales from `source_frame_size` to `current_frame_size` using ratio `dst/src`.
- **No Scaling**: If no frame sizes provided, coordinates are used as-is.

## 6. Hot-Reload
```python
counting_service.update_config(
    line_start=[100, 200],
    line_end=[300, 400],
    inside_point=[150, 350],
    crossing_margin=15.0,
    source_frame_size=(1920, 1080),
    current_frame_size=(1280, 720),
)
```

## 7. Debug Visualization
```python
frame = counting_service.draw_margin_region(frame)  # Draws magenta margin band
```
