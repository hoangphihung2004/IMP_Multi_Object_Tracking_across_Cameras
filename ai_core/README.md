# AI Core - Windows Local DeepStream Tracking

Tài liệu này hướng dẫn chạy `ai_core` local trên **Windows 11 + WSL2 + Docker Desktop**, giữ kiến trúc **DeepStream + YOLOX + OCSort C++**.

Runtime hiện tại chỉ làm các việc sau:

- Đọc 2 video test trong `videos/`.
- Chạy YOLOX detection bằng DeepStream `nvinfer` với TensorRT engine.
- Chạy OCSort C++ tracking theo từng source.
- Vẽ bounding box, track ID và runtime stats overlay lên video.
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
├── pipeline_config.json     # DeepStream/GStreamer pipeline properties
├── yolox_config.json        # YOLOX model/inference config
├── ocsort_config.json       # OCSort tracking config
├── system_config.json       # device/fp16 settings
├── config_infer.txt         # DeepStream nvinfer config
└── labels.txt               # label file
```

`pipeline_config.json` là nơi kiểm soát các property quan trọng của pipeline như `nvstreammux`, decoder, encoder, bitrate và output. Không chỉnh các giá trị này trực tiếp trong code nếu đã có trong config.

## Runtime pipeline hiện tại

Do Docker Desktop + WSL2 không ổn định với NVIDIA V4L2 codec (`nvv4l2decoder`, `nvv4l2h264enc`), runtime local hiện dùng software codec fallback đã được kiểm chứng:

```text
Input branch:
filesrc → qtdemux → h264parse → avdec_h264 → nvvideoconvert → nvstreammux

Core branch:
nvstreammux → nvinfer → AnalyticsProbe/OCSort → nvstreamdemux

Output branch:
nvstreamdemux → queue → nvvideoconvert → nvdsosd → nvvideoconvert
             → capsfilter video/x-raw,format=I420 → x264enc → h264parse → qtmux → filesink
```

Config tương ứng:

```json
"decode": {
  "parser": "h264parse",
  "decoder": "avdec_h264",
  "converter": "nvvideoconvert"
},
"output": {
  "encoder": "x264enc"
}
```

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
# Nếu .env đã tồn tại thì giữ nguyên file hiện có.
docker compose build
docker compose up
```

Output sau khi chạy:

```text
output/out_0.mp4
output/out_1.mp4
```

`output/`, `videos/`, `pretrained/` và các file model generated (`*.onnx`, `*.engine`) không được commit vào git.

## Chạy dev trong container hiện có

Khi cần chạy tay trong container dev:

```bash
docker compose run --name ai_core_manual --entrypoint bash ai_core
cd /workspace/ai_core
python3 main.py
```

Nếu container đã tồn tại:

```bash
docker start -ai ai_core_manual
cd /workspace/ai_core
python3 main.py
```

Source code được bind mount:

```text
Host ai_core/ ↔ /workspace/ai_core
```

Vì vậy sửa Python/config trên host chỉ cần chạy lại `python3 main.py`, không cần build Docker. Chỉ build lại image khi thay đổi Dockerfile hoặc dependency hệ thống/Python.

## Kiểm tra GPU trong container

```bash
docker compose run --rm ai_core nvidia-smi
```

Kỳ vọng: thấy GPU NVIDIA của máy Windows.

## Model conversion

Khi app khởi động, `core/model_converter.py` kiểm tra theo thứ tự:

1. Nếu có `pretrained/model_trt.engine` thì dùng trực tiếp.
2. Nếu có ONNX thì build TensorRT engine bằng `trtexec`.
3. Nếu chỉ có PTH thì convert `.pth → .onnx → .engine`.

Các path chính lấy từ `.env` và config:

```env
MODEL_ENGINE_PATH=pretrained/model_trt.engine
MODEL_ONNX_PATH=pretrained/model_trt.onnx
MODEL_PTH_PATH=pretrained/ocsort_x_mot20.pth.tar
TRT_FP16=1
TRT_WORKSPACE_MB=4096
```

## Debug runtime config và stats

Khi app khởi động, log sẽ in:

- Config đã nạp từ JSON/env.
- Giá trị model conversion thật sau khi resolve env/config.
- GStreamer properties thật sau khi `set_property`.
- Runtime stats từng source.

Log stats mẫu:

```text
[Stats] source=0 | frame=629 | fps=25.4 | detections=5 | tracks=1 | total_fps=50.8 | total_detections=7 | total_tracks=3
```

Video output có overlay dạng:

```text
CAM 0 | FPS 25.4 | DET 5 | TRACKS 1 | FRAME 629
```

## Ghi chú về WSL2 codec

Pipeline từng thử NVIDIA codec:

- `nvv4l2decoder`
- `nvv4l2h264enc`

Nhưng trên Docker Desktop + WSL2 có thể gặp lỗi:

```text
S_EXT_CTRLS for CUDA_GPU_ID failed
ENC_CTX Error in initializing nvenc context
```

Vì vậy config mặc định của runtime local dùng:

- Decode: `avdec_h264`
- Encode: `x264enc`

Các package GStreamer fallback đã được đưa vào Dockerfile:

```text
gstreamer1.0-libav
gstreamer1.0-plugins-ugly
libavcodec58 libavutil56 libavformat58
libmpg123-0 libvpx7 libx264-163 libx265-199
```
