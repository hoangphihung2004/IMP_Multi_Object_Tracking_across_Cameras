# IMP Multi-Object Tracking Across Cameras

Dự án chạy AI tracking local bằng **DeepStream + YOLOX + OCSort C++**.

Phần đang triển khai trong nhánh này tập trung vào `ai_core`:

- Chạy trên Windows 11 qua WSL2 + Docker Desktop.
- Đọc 2 video trong `ai_core/videos/`.
- Chạy YOLOX TensorRT inference bằng DeepStream `nvinfer`.
- Chạy OCSort C++ tracking theo từng source.
- Ghi overlay bounding boxes, track IDs và runtime stats lên video.
- Xuất 2 video tracking vào `ai_core/output/`.
- Không dùng backend/WebSocket/counting/mapping trong runtime local.

Runtime local hiện dùng software codec fallback đã kiểm chứng trên Docker Desktop + WSL2:

- Decode: `avdec_h264`.
- Encode: `x264enc`.

Lý do: NVIDIA V4L2 codec (`nvv4l2decoder`, `nvv4l2h264enc`) có thể lỗi trong Docker/WSL2.

Xem hướng dẫn chi tiết tại:

- [`ai_core/README.md`](ai_core/README.md)
- [`ai_core/services/pipeline/Pipeline_README.md`](ai_core/services/pipeline/Pipeline_README.md)
- [`ai_core/core/README.md`](ai_core/core/README.md)
- [`docs/superpowers/specs/2026-05-31-windows-docker-migration-design.md`](docs/superpowers/specs/2026-05-31-windows-docker-migration-design.md)
- [`docs/superpowers/plans/2026-06-01-windows-docker-deepstream-local-tracking.md`](docs/superpowers/plans/2026-06-01-windows-docker-deepstream-local-tracking.md)
