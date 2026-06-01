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

## 1. Yêu cầu trước khi chạy

### 1.1. Môi trường máy host

Yêu cầu:

- Windows 11.
- WSL2 đã bật.
- Docker Desktop đã cài và bật WSL2 backend.
- NVIDIA GPU driver hỗ trợ Docker GPU passthrough.
- Đang đứng trong thư mục `ai_core` khi chạy lệnh Docker.

Đường dẫn dự án mẫu:

```powershell
cd "C:\Users\manhm\Desktop\IMP_Multi_Object_Tracking_across_Cameras\.claude\worktrees\feature-AI-windows-docker-migration\ai_core"
```

Kiểm tra GPU trên host:

```powershell
nvidia-smi
```

Nếu lệnh này không thấy GPU NVIDIA thì Docker container cũng chưa thể dùng GPU.

### 1.2. File/thư mục bắt buộc trong `ai_core`

Trước khi chạy cần có:

```text
ai_core/
├── .env                         # env runtime thật, không commit
├── .env.example                 # template env
├── docker-compose.yml
├── Dockerfile
├── main.py
├── config/
│   ├── config_infer.txt
│   ├── sources_config.json
│   ├── pipeline_config.json
│   ├── yolox_config.json
│   ├── ocsort_config.json
│   ├── system_config.json
│   └── labels.txt
├── pretrained/
│   └── ocsort_x_mot20.pth.tar   # checkpoint gốc, hoặc có sẵn model_trt.engine
├── videos/
│   ├── 1-1.mp4
│   └── 2-1.mp4
└── output/                      # nơi ghi out_0.mp4, out_1.mp4
```

Các file/thư mục lớn không commit vào git:

```text
.env
videos/
output/
pretrained/
*.onnx
*.engine
*.so
```

### 1.3. File `.env`

Nếu chưa có `.env`, tạo từ template:

```powershell
copy .env.example .env
```

Nội dung chính:

```env
CONFIG_INFER=config/config_infer.txt
MODEL_ENGINE_PATH=pretrained/model_trt.engine
MODEL_ONNX_PATH=pretrained/model_trt.onnx
MODEL_PTH_PATH=pretrained/ocsort_x_mot20.pth.tar
TRT_FP16=1
TRT_WORKSPACE_MB=4096
LOG_LEVEL=DEBUG
```

Chỉ dùng `.env` ở root `ai_core`. Không tạo thêm `config/.env` để tránh lệch config.

## 2. Cấu trúc config chính

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

## 3. Runtime pipeline hiện tại

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

## 4. Cách chạy tự động bằng Docker Compose

Cách này dùng `docker compose up`, phù hợp khi muốn chạy toàn bộ pipeline giống runtime chính thức.

### 4.1. Kiểm tra không có container cũ đang chạy

```powershell
docker ps -a
```

Nếu có container cũ tên `dat_lab_deepstream_local` đang chạy, dừng nó:

```powershell
docker stop dat_lab_deepstream_local
```

Nếu muốn xoá container cũ để chạy sạch:

```powershell
docker rm dat_lab_deepstream_local
```

### 4.2. Build image

Chạy từ thư mục `ai_core`:

```powershell
docker compose build
```

Không dùng `--no-cache` trừ khi thật sự cần, vì image DeepStream rất nặng.

Image build ra:

```text
manhmitcf/dat_lab_management_aicore:windows-local
```

### 4.3. Kiểm tra GPU trong container

```powershell
docker compose run --rm ai_core nvidia-smi
```

Kỳ vọng thấy GPU NVIDIA của máy Windows.

### 4.4. Chạy pipeline tự động

```powershell
docker compose up
```

Compose sẽ:

1. Tạo container `dat_lab_deepstream_local`.
2. Chạy `docker_entrypoint.sh`.
3. Compile lại C++ plugins trong container.
4. Chạy `python3 main.py`.
5. Ghi output vào `output/`.

Khi chạy thành công sẽ thấy log:

```text
[ModelConverter] ✅ TensorRT engine found ... Ready!
[Pipeline] Pipeline linked successfully for local MP4 output.
[Pipeline] Starting pipeline...
[Stats] source=0 | frame=... | fps=... | detections=... | tracks=...
[Stats] source=1 | frame=... | fps=... | detections=... | tracks=...
nvstreammux: Successfully handled EOS for source_id=0
nvstreammux: Successfully handled EOS for source_id=1
[Bus] End-of-Stream (EOS) reached.
[Pipeline] Pipeline stopped.
```

### 4.5. Kiểm tra output

Mở PowerShell khác:

```powershell
Get-ChildItem .\output\*.mp4 | Select-Object Name,Length,LastWriteTime
```

Kỳ vọng:

```text
out_0.mp4    > 0 bytes
out_1.mp4    > 0 bytes
```

Mở video để kiểm tra:

```text
output/out_0.mp4
output/out_1.mp4
```

Trong video sẽ có:

- Bounding boxes.
- Track ID.
- Runtime overlay: `CAM`, `FPS`, `DET`, `TRACKS`, `FRAME`.

## 5. Cách chạy bằng terminal trong Docker

Cách này phù hợp để debug thủ công, cài package thử, chạy lại nhanh mà không build Docker liên tục.

### 5.1. Tạo container manual lần đầu

Từ thư mục `ai_core`:

```powershell
docker compose run --name ai_core_manual --entrypoint bash ai_core
```

Sau khi vào container, prompt sẽ giống:

```text
root@<container_id>:/workspace/ai_core#
```

### 5.2. Nếu container `ai_core_manual` đã tồn tại

Kiểm tra:

```powershell
docker ps -a
```

Nếu container đang dừng:

```powershell
docker start -ai ai_core_manual
```

Nếu container đang chạy và muốn mở thêm terminal:

```powershell
docker exec -it ai_core_manual bash
```

### 5.3. Chạy app trong container manual

Trong container:

```bash
cd /workspace/ai_core
rm -f output/out_0.mp4 output/out_1.mp4
python3 main.py
```

Nếu muốn chạy một lệnh từ PowerShell, không cần vào shell:

```powershell
docker exec -it ai_core_manual bash -lc "cd /workspace/ai_core && rm -f output/out_0.mp4 output/out_1.mp4 && python3 main.py"
```

### 5.4. Kiểm tra package/codec trong container

Trong container:

```bash
python3 - <<'PY'
import numba, torchvision
print('numba', numba.__version__)
print('torchvision', torchvision.__version__)
PY
```

Kiểm tra GStreamer elements:

```bash
for e in avdec_h264 x264enc h264parse qtmux nvvideoconvert nvdsosd nvinfer nvstreammux nvstreamdemux; do
  gst-inspect-1.0 $e >/dev/null && echo OK:$e || echo MISSING:$e
done
```

Tất cả phải là `OK`.

### 5.5. Test decode riêng

Dùng để kiểm tra input video có decode được không:

```bash
gst-launch-1.0 -q filesrc location=videos/1-1.mp4 ! qtdemux ! h264parse ! avdec_h264 ! fakesink sync=false
```

Nếu lệnh chạy xong không lỗi là decode OK.

### 5.6. Khi nào cần build Docker lại?

Không cần build lại khi chỉ sửa:

- Python code.
- JSON config.
- `config_infer.txt`.
- README/docs.

Vì source code được bind mount:

```text
Host ai_core/ ↔ /workspace/ai_core
```

Cần build lại khi sửa:

- `Dockerfile`.
- `requirements.txt`.
- apt/system packages.
- Python packages cần cài vào image.

## 6. Model conversion

Khi app khởi động, `core/model_converter.py` kiểm tra theo thứ tự:

1. Nếu có `pretrained/model_trt.engine` thì dùng trực tiếp.
2. Nếu có ONNX thì build TensorRT engine bằng `trtexec`.
3. Nếu chỉ có PTH thì convert `.pth → .onnx → .engine`.

Nếu engine chưa có, lần chạy đầu có thể mất lâu vì phải convert model và build TensorRT engine.

Log build TensorRT thành công sẽ có dạng:

```text
&&&& PASSED TensorRT.trtexec
[ModelConverter] ✅ ONNX → TensorRT Engine (model_trt.engine) succeeded.
```

## 7. Debug runtime config và stats

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

## 8. Lỗi thường gặp

### 8.1. `ModuleNotFoundError: No module named 'numba'`

Image/container thiếu `numba`.

Trong container manual có thể cài tạm:

```bash
pip3 install --retries 10 --timeout 120 numba
```

Dockerfile/requirements hiện đã bổ sung `numba`.

### 8.2. `ModuleNotFoundError: No module named 'torchvision'`

Image thiếu `torchvision`.

Dockerfile hiện cài:

```dockerfile
RUN pip3 install --retries 10 --timeout 120 --no-deps torchvision
```

Không đưa `torchvision` vào `requirements.txt` để tránh pip kéo lại `torch`/CUDA packages rất lớn.

### 8.3. `Unknown key 'cluster-mode'`

`cluster-mode` phải nằm trong `[property]` của `config/config_infer.txt`, không đặt dưới `[class-attrs-all]`.

Đúng:

```ini
[property]
cluster-mode=4

[class-attrs-all]
pre-cluster-threshold=0.1
nms-iou-threshold=0.7
```

### 8.4. `Backend has maxBatchSize 1 whereas 2 has been requested`

TensorRT explicit batch engine báo max batch size là 1. Giữ trong `config/config_infer.txt`:

```ini
batch-size=1
```

`nvstreammux` vẫn có thể batch 2 nguồn, nhưng `nvinfer` config hiện dùng batch-size 1 để khớp engine đã build.

### 8.5. `S_EXT_CTRLS for CUDA_GPU_ID failed`

Thường xảy ra với `nvv4l2decoder` trong Docker Desktop + WSL2. Runtime hiện đã chuyển sang `avdec_h264`.

### 8.6. `ENC_CTX Error in initializing nvenc context`

Thường xảy ra với `nvv4l2h264enc` trong Docker Desktop + WSL2. Runtime hiện đã chuyển sang `x264enc`.

## 9. Ghi chú về `docker_entrypoint.sh`

`docker-compose.yml` đang dùng:

```yaml
entrypoint: ["bash", "/workspace/ai_core/docker_entrypoint.sh"]
command: ["python3", "main.py"]
```

Entrypoint này compile lại C++ plugins khi container start:

- OCSort C++ API.
- YOLOX custom parser.
- YOLOX preprocess plugin.

Với chế độ dev bind mount, việc compile lại giúp `.so` luôn khớp source C++ hiện tại.

## 10. Dọn container manual

Nếu muốn xoá container manual:

```powershell
docker rm -f ai_core_manual
```

Tạo lại:

```powershell
docker compose run --name ai_core_manual --entrypoint bash ai_core
```
