# AI Core - Windows Local DeepStream Tracking

Tài liệu này hướng dẫn chạy `ai_core` local trên **Windows 11 + WSL2 + Docker Desktop**, giữ kiến trúc **DeepStream + YOLOX + OCSort C++**.

Runtime hiện tại chỉ làm các việc sau:

- Đọc 2 video test trong `videos/`.
- Chạy YOLOX detection bằng DeepStream `nvinfer`.
- Chạy OCSort C++ tracking theo từng source.
- Vẽ bounding box + track ID lên video.
- Xuất 2 file mp4 vào `output/`.

Các tính năng đã cắt khỏi runtime local:

- Counting.
- Mapping 2D.
- WebSocket/backend gateway.
- Janus/WebRTC streaming.
- USB camera/RTSP validator.

## Cấu trúc config chính

```text
config/
├── sources_config.json      # input videos + output files
├── pipeline_config.json     # DeepStream pipeline properties
├── yolox_config.json        # YOLOX model/inference config
├── ocsort_config.json       # OCSort tracking config
├── system_config.json       # device/fp16 settings
├── config_infer.txt         # DeepStream nvinfer config
└── labels.txt               # label file
```

`pipeline_config.json` là nơi kiểm soát các property quan trọng của DeepStream như `nvstreammux`, decoder, encoder và output. Không chỉnh các giá trị này trực tiếp trong code nếu đã có trong config.

## Chuẩn bị môi trường

Yêu cầu:

- Windows 11.
- WSL2.
- Docker Desktop bật WSL2 backend.
- NVIDIA GPU driver hỗ trợ Docker GPU passthrough.

Kiểm tra GPU trên host:

```bash
nvidia-smi
```

## Chạy bằng Docker

Từ thư mục `ai_core`:

```bash
cp .env.example .env
docker compose build
docker compose up
```

Output sau khi chạy:

```text
output/out_0.mp4
output/out_1.mp4
```

## Kiểm tra GPU trong container

```bash
docker compose run --rm ai_core nvidia-smi
```

Kỳ vọng: thấy GPU NVIDIA của máy Windows.

## Debug config

Khi app khởi động, log sẽ in toàn bộ config đã nạp:

- sources input/output.
- pipeline properties.
- YOLOX config.
- OCSort config.
- system config.

Nếu config thiếu hoặc sai, app sẽ dừng sớm thay vì chạy với giá trị mơ hồ.

## Ghi chú về WSL2 codec

Pipeline ưu tiên NVIDIA elements:

- `nvv4l2decoder`
- `nvv4l2h264enc`

Nếu WSL2 không hỗ trợ codec phần cứng đúng như môi trường Jetson/Linux, cần kiểm tra bằng:

```bash
gst-inspect-1.0 nvv4l2decoder
gst-inspect-1.0 nvv4l2h264enc
```

Nếu cần fallback, sửa trong `config/pipeline_config.json`, không hardcode trong `services/pipeline/manager.py`.
