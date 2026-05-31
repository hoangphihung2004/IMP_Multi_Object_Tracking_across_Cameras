# Core Module

Thư mục `core/` chứa các thành phần nền tảng cho runtime AI local.

## Thành phần chính

### `config.py`

Quản lý cấu hình bằng Pydantic models:

- `SystemConfig`: thiết lập device/fp16.
- `YOLOXConfig`: đường dẫn model/engine và threshold detection.
- `OCSortConfig`: tham số tracking.
- `SourcesConfig`: danh sách input video và output mp4.
- `PipelineConfig`: property DeepStream/GStreamer pipeline.

Mục tiêu của `config.py` là gom cấu hình vào một nơi rõ ràng, validate sớm và in log debug khi app khởi động.

### `model_converter.py`

Kiểm tra model và tự động convert theo chuỗi:

```text
.pth → .onnx → .engine
```

Nếu TensorRT engine đã tồn tại thì app dùng trực tiếp.

## Không còn trong runtime local

Các phần sau đã bị xóa khỏi runtime local:

- Camera validator cho USB/RTSP.
- Counting config.
- Mapping config.
- Backend/WebSocket schema.
