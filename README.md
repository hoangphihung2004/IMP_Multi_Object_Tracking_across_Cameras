# IMP Multi-Object Tracking Across Cameras

Dự án chạy AI tracking local bằng **DeepStream + YOLOX + OCSort C++**.

Phần đang triển khai trong nhánh này tập trung vào `ai_core`:

- Chạy trên Windows 11 qua WSL2 + Docker Desktop.
- Đọc 2 video trong `ai_core/videos/`.
- Xuất 2 video tracking vào `ai_core/output/`.
- Không dùng backend/WebSocket/counting/mapping trong runtime local.

Xem hướng dẫn chi tiết tại:

- [`ai_core/README.md`](ai_core/README.md)
- [`docs/superpowers/specs/2026-05-31-windows-docker-migration-design.md`](docs/superpowers/specs/2026-05-31-windows-docker-migration-design.md)
- [`docs/superpowers/plans/2026-06-01-windows-docker-deepstream-local-tracking.md`](docs/superpowers/plans/2026-06-01-windows-docker-deepstream-local-tracking.md)
