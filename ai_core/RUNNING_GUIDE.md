# 🚀 Hướng Dẫn Chạy AI Core - YOLOX + OC-SORT Tracking

> **Cập nhật**: 2026-06-14  
> **Branch**: `aicore-dev`  
> **Tác giả**: Anh Mạnh

---

## 📋 Tổng Quan

Hệ thống tracking sử dụng:
- **Detection Model**: YOLOX-X (MOT20 pretrained)
- **Tracking Algorithm**: OC-SORT (Observation-Centric SORT)
- **Optimization**: TensorRT FP16 (RTX 3050)
- **Output**: Video với bounding boxes và track IDs

---

## 🏗️ Kiến Trúc Pipeline

```
Input Video (MP4)
    ↓
Frame-by-Frame Processing
    ↓
YOLOX Detection (TensorRT)
    ↓
OC-SORT Tracking (Kalman Filter)
    ↓
Visualization (OpenCV)
    ↓
Output Video (MP4)
```

---

## 📦 1. Setup Docker Container

> **Lưu ý**: Thay `<path-to-project>` bằng đường dẫn thực tế đến thư mục project trên máy bạn.

### 1.1. Build Docker Image

```bash
cd <path-to-project>/ai_core
docker build -t yolox-ocsort:latest .
```

**Ví dụ**:
```bash
# Windows
cd C:\Users\manhm\Desktop\DAT_Lab_Management\ai_core

# Linux/Mac
cd /home/user/projects/DAT_Lab_Management/ai_core

# Sau đó chạy:
docker build -t yolox-ocsort:latest .
```

**Lưu ý**: File `.dockerignore` đã loại trừ:
- `pretrained/` (model weights)
- `videos/` (input videos)
- `.git/`, `__pycache__/`, `*.pyc`

### 1.2. Run Docker với GPU Support

```bash
docker run -it --gpus all -v "<path-to-project>/ai_core:/workspace/ai_core" yolox-ocsort:latest bash
```

**Ví dụ**:
```bash
# Windows
docker run -it --gpus all -v "C:\Users\manhm\Desktop\DAT_Lab_Management\ai_core:/workspace/ai_core" yolox-ocsort:latest bash

# Linux/Mac
docker run -it --gpus all -v "/home/user/projects/DAT_Lab_Management/ai_core:/workspace/ai_core" yolox-ocsort:latest bash
```

**Volume Mapping**:
| Host Path | Container Path | Mục đích |
|-----------|---------------|----------|
| `<path-to-project>/ai_core/` | `/workspace/ai_core/` | ✅ **Toàn bộ project** (code + pretrained + videos + output) |

**Cấu trúc trong container** (root = `ai_core/`):
```
ai_core/                      (root directory)
├── main.py
├── config/tracking_config.json
├── pretrained/
│   ├── ocsort_x_mot20.pth.tar
│   └── model_trt.pth
├── videos/demo.mp4
├── output/result.mp4
└── services/...
```

> **⚠️ Quan trọng**: 
> - Tất cả path trong code đều **relative từ `ai_core/`** làm root
> - Ví dụ: `pretrained/model.pth`, `videos/demo.mp4`, `config/tracking_config.json`
> - Không cần `/workspace/` hay `../` trong code!
> - **Working directory phải là `ai_core/`** khi chạy mọi lệnh

### 1.3. Set Working Directory

**Trong Docker**:
```bash
cd /workspace/ai_core
pwd  # Verify: /workspace/ai_core
```

**Trên máy local** (không Docker):
```bash
# Windows
cd C:\Users\manhm\Desktop\DAT_Lab_Management\ai_core

# Linux/Mac
cd /home/user/projects/DAT_Lab_Management/ai_core

# Verify
pwd  # Should show path ending with /ai_core
```

> **⚠️ Tất cả lệnh Python/trtexec phải chạy từ thư mục `ai_core/`!**

### 1.4. Verify GPU Inside Docker

```bash
nvidia-smi
```

**Expected Output**:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 596.49                 Driver Version: 596.49         CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3050 ...  WDDM  |   00000000:01:00.0  On |                  N/A |
| N/A   50C    P0              15W /  55W  |    1234MiB /   6144MiB |     12%      Default |
+-----------------------------------------+------------------------+----------------------+
```

---

## ⚙️ 2. Model Conversion Pipeline

> **Lưu ý**: Tất cả lệnh chạy với `ai_core/` làm working directory root.

### 2.1. PyTorch → ONNX (Export)

```bash
python deploy/scripts/export_onnx.py --output-name pretrained/yolox_x.onnx -f exps/yolox_x_mix_mot20_ch.py -c pretrained/ocsort_x_mot20.pth.tar
```

**Output**: `pretrained/yolox_x.onnx` (~350MB)

### 2.2. ONNX → TensorRT Engine (Optimization)

```bash
trtexec --onnx=pretrained/yolox_x.onnx --saveEngine=pretrained/yolox_x.engine --workspace=3096 --fp16
```

**Parameters**:
- `--workspace=3096`: 3GB memory workspace
- `--fp16`: Half precision (faster inference)
- **Build time**: ~37 phút (RTX 3050)

**Output**: `pretrained/yolox_x.engine` (~180MB)

### 2.3. TensorRT Engine → Wrapped .pth (Deploy)

```bash
python deploy/scripts/trt.py -f exps/yolox_x_mix_mot20_ch.py -c pretrained/ocsort_x_mot20.pth.tar --trt pretrained/yolox_x.engine
```

**Output**: `pretrained/model_trt.pth` (TensorRT wrapper)

---

## 🎬 3. Chạy Tracking

> **Working directory**: `ai_core/` (root của project)

### 3.1. Config File

**File**: `config/tracking_config.json`

```json
{
  "device": "gpu",
  "conf": 0.1,              // Detection confidence threshold
  "nms": 0.7,               // Non-Maximum Suppression threshold
  "tsize": 640,             // Input size (640x640)
  "fp16": false,            // FP16 inference (handled by TensorRT)
  "fuse": false,            // Model layer fusion (không dùng với TRT)
  "trt": true,              // ✅ Enable TensorRT
  "ckpt": "pretrained/ocsort_x_mot20.pth.tar",      // ✅ Relative từ ai_core/
  "trt_file": "pretrained/model_trt.pth",           // ✅ Relative từ ai_core/
  "track_thresh": 0.4,      // Tracking confidence threshold (giảm để giữ ID ổn định)
  "iou_thresh": 0.2,        // IoU threshold for matching (giảm để dễ match)
  "use_byte": false,        // ByteTrack-style matching
  "aspect_ratio_thresh": 1.6,  // Loại bỏ bbox quá cao/hẹp
  "min_box_area": 100       // Loại bỏ bbox quá nhỏ
}
```

> **Lưu ý**: Tất cả path đều **relative từ `ai_core/`** làm root directory.

### 3.2. Chạy Main Script

```bash
python main.py
```

**Input**: `videos/demo.mp4` (relative từ `ai_core/`)  
**Output**: `output/result.mp4` (relative từ `ai_core/`)

### 3.3. Code Flow (`main.py`)

```python
# 1. Load config
tracking_cfg = OCSortConfig("config/tracking_config.json")

# 2. Initialize tracking service (YOLOX + OC-SORT)
tracking_service = TrackingService(args=tracking_cfg)

# 3. Process video frame-by-frame
for frame in video:
    bboxes_tlwh, track_ids, avg_time = tracking_service.predict(
        frame_id=frame_id,
        frame=frame,
        box_type="tlwh"  # top-left width-height format
    )
    
    # 4. Draw tracking results
    vis_frame = draw_tracking_results(frame, bboxes_tlwh, track_ids, fps_info)
    
    # 5. Write to output video
    video_writer.write(vis_frame)
```

---

## 🎨 4. Visualization

### 4.1. Bounding Box Drawing

**Function**: `draw_tracking_results()` (main.py)

```python
# Mỗi tracked object có:
- Bounding box: cv2.rectangle() với màu riêng cho mỗi track_id
- Track ID label: "ID:123" góc trên-trái bbox
- FPS info: "FPS: 45.2 | Tracks: 8" góc trên-trái frame
```

### 4.2. Color Palette

**Import**: `from utils.visualize import sample_colors`

Colors tự động cycle theo `track_id % len(sample_colors)`.

---

## 📊 5. Performance Metrics

### 5.1. Expected Performance (RTX 3050)

| Stage | FPS | Notes |
|-------|-----|-------|
| **YOLOX Detection (TensorRT FP16)** | ~60-80 FPS | Optimized inference |
| **OC-SORT Tracking** | ~100+ FPS | Lightweight Kalman Filter |
| **Visualization** | ~50-60 FPS | OpenCV drawing overhead |
| **Overall Pipeline** | **~45-55 FPS** | Bottleneck: video encoding |

### 5.2. Log Output Example

```
2026-06-14 10:30:15 | INFO     | Starting tracking application...
2026-06-14 10:30:16 | SUCCESS  | Tracking service initialized.
2026-06-14 10:30:16 | INFO     | Opening video: ./videos/demo.mp4
2026-06-14 10:30:16 | INFO     | Video info: 1920x1080, 30.00 FPS, 900 frames
2026-06-14 10:30:17 | INFO     | Output will be saved to: ./output/result.mp4
2026-06-14 10:30:17 | INFO     | Starting frame processing...
2026-06-14 10:30:18 | INFO     | Frame 30/900 (3.3%) | Tracks: 5 | FPS: 48.2
2026-06-14 10:30:20 | INFO     | Frame 60/900 (6.7%) | Tracks: 7 | FPS: 51.3
...
2026-06-14 10:30:45 | SUCCESS  | Processing completed!
                                   Total frames: 900
                                   Total time: 18.50s
                                   Average FPS: 48.65
                                   Output saved: ./output/result.mp4
```

---

## 🔧 6. Troubleshooting

### 6.1. CUDA Driver Not Detected

**Error**:
```
CUDA driver version is insufficient for CUDA runtime version
```

**Solution**:
```bash
# Restart Docker với GPU flag
docker run -it --gpus all ...
```

### 6.2. Model File Not Found

**Error**:
```
FileNotFoundError: pretrained/ocsort_x_mot20.pth.tar
```

**Solution**:
```bash
# Đảm bảo đang ở đúng working directory (ai_core/)
pwd  # Should output: /workspace/ai_core (in Docker) or <your-path>/ai_core (local)

# Verify file tồn tại
ls pretrained/
# Kết quả mong đợi: ocsort_x_mot20.pth.tar, model_trt.pth
```

### 6.3. Missing Dependencies

**Error**:
```
ModuleNotFoundError: No module named 'seaborn'
```

**Solution**:
```bash
pip install seaborn
```

**Hoặc rebuild Docker** sau khi thêm vào `requirements.txt`.

### 6.4. TensorRT Model Shape Mismatch

**Error**:
```
RuntimeError: Expected tensor with 3 dims, got 4
```

**Fix**: Đã xử lý trong `tracking_service.py` (lines 76-77):
```python
if outputs.dim() == 4 and outputs.shape[1] == 1:
    outputs = outputs.squeeze(1)  # (1,1,N,C) → (1,N,C)
```

---

## 📁 7. File Structure

```
ai_core/
├── config/
│   ├── tracking_config.json       # ✅ Main config
│   └── data_config.py             # OCSortConfig dataclass
├── services/
│   └── tracking_service.py        # YOLOX + OC-SORT implementation
├── trackers/
│   └── ocsort_tracker/
│       └── ocsort.py              # OC-SORT algorithm
├── exps/
│   └── yolox_x_mix_mot20_ch.py   # Model architecture
├── deploy/
│   └── scripts/
│       ├── export_onnx.py         # PyTorch → ONNX
│       └── trt.py                 # TensorRT wrapper
├── utils/
│   └── visualize.py               # Color palette
├── pretrained/                    # ⚠️ Not in Docker image
│   ├── ocsort_x_mot20.pth.tar    # Original PyTorch weights
│   ├── yolox_x.onnx              # ONNX model
│   ├── yolox_x.engine            # TensorRT engine
│   └── model_trt.pth             # TensorRT wrapper
├── videos/                        # ⚠️ Not in Docker image
│   └── demo.mp4                  # Input video
├── output/                        # ⚠️ Not in Docker image
│   └── result.mp4                # Output video
├── main.py                        # ✅ Entry point
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

---

## 🔄 8. So Sánh với Branch `aicore/mapping_service`

### 8.1. Giống Nhau (Core Tracking)

| Component | Status |
|-----------|--------|
| `tracking_config.json` | ✅ **GIỐNG 100%** |
| YOLOX Detection | ✅ Giống (conf=0.1, nms=0.7) |
| OC-SORT Tracking | ✅ Giống (track_thresh=0.6, iou_thresh=0.3) |
| TrackingService class | ✅ Giống |

### 8.2. Khác Nhau (Services đã xóa)

Branch `aicore-dev` đã **cleanup**:
- ❌ `counting_service.py` (Line counting)
- ❌ `mapping_service.py` (Coordinate mapping with 1233 points)
- ❌ `pipeline_service.py` (Pipeline orchestration)
- ❌ `backend_gateway.py` (API communication)
- ❌ `TrackingUltralytics` class (YOLOv8 integration)
- ❌ ByteTrack/BoT-SORT configs

**Kết quả**: ~2000 lines code đã xóa, chỉ giữ lại YOLOX + OC-SORT core.

---

## 💡 9. Tips & Best Practices

### 9.1. Tuning Detection

**Tăng precision** (ít false positives):
```json
{
  "conf": 0.3,    // Tăng từ 0.1 → 0.3
  "nms": 0.5      // Giảm từ 0.7 → 0.5
}
```

**Tăng recall** (catch nhiều object):
```json
{
  "conf": 0.05,   // Giảm xuống 0.05
  "nms": 0.9      // Tăng lên 0.9
}
```

### 9.2. Tuning Tracking

**Tracking ổn định hơn** (giữ ID lâu):
```json
{
  "track_thresh": 0.4,   // Giảm từ 0.6
  "iou_thresh": 0.5      // Tăng từ 0.3
}
```

**Tracking responsive hơn** (switch ID nhanh):
```json
{
  "track_thresh": 0.8,   // Tăng lên 0.8
  "iou_thresh": 0.2      // Giảm xuống 0.2
}
```

### 9.3. Memory Optimization

```bash
# Giảm workspace nếu thiếu VRAM
trtexec --workspace=2048 ...  # 2GB thay vì 3GB

# Dùng batch size nhỏ hơn (mặc định=1, đã tối ưu)
```

---

## 📞 10. Support & Maintenance

**Maintainer**: Anh Mạnh  
**Last Updated**: 2026-06-14  
**Branch**: `aicore-dev`  
**Commit**: `1ff90b2` (chore: remove unused config files)

**Known Issues**:
- [ ] Video output codec compatibility (mp4v → h264)
- [ ] Docker image size optimization (~8GB)
- [ ] Multi-GPU support

**Future Work**:
- [ ] Add command-line argument support (`run_tracking.py`)
- [ ] Real-time camera input support
- [ ] Export tracking results to JSON/CSV

---

## 📚 References

- [YOLOX Paper](https://arxiv.org/abs/2107.08430)
- [OC-SORT Paper](https://arxiv.org/abs/2203.14360)
- [TensorRT Documentation](https://developer.nvidia.com/tensorrt)
- [MOT20 Dataset](https://motchallenge.net/data/MOT20/)

---

**✅ Hệ thống đã sẵn sàng chạy!** 🚀
