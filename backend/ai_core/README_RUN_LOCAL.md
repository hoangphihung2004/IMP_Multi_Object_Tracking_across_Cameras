# Hướng dẫn chạy local tracking

Tài liệu này hướng dẫn chạy lại dự án ở local bằng Docker để tạo video output tracking. Luồng hiện tại chỉ chạy thuật toán local, không dùng backend API, WebSocket, counting, hoặc mapping.

## 1. Mục tiêu

Input:

```text
videos/demo.mp4
```

Output:

```text
videos/result_demo.mp4
```

Luồng xử lý:

```text
video input
→ YOLOX + TensorRT detection
→ OC-SORT tracking
→ vẽ bounding box + track ID
→ video output
```

## 2. Yêu cầu trước khi chạy

Cần có:

- Docker Desktop đang chạy.
- GPU NVIDIA và driver hợp lệ.
- Docker hỗ trợ GPU qua `--gpus all`.
- File video input:

```text
backend/ai_core/videos/demo.mp4
```

- File TensorRT model:

```text
backend/ai_core/pretrained/model_trt.pth
```

- File config tracking:

```text
backend/ai_core/config/tracking_config.json
```

Config mặc định đang dùng TensorRT:

```json
"device": "gpu",
"trt": true,
"trt_file": "./pretrained/model_trt.pth"
```

## 3. Build Docker image

Mở PowerShell tại thư mục `backend/ai_core`:

```powershell
cd D:\IMP_Multi_Object_Tracking_across_Cameras\backend\ai_core
docker build -t ai-core:latest .
```

Lần build đầu có thể lâu vì Dockerfile dùng image TensorRT của NVIDIA:

```dockerfile
nvcr.io/nvidia/tensorrt:21.09-py3
```

## 4. Chạy container

Từ thư mục `backend/ai_core`:

```powershell
docker run -it --gpus all --name ai_core_container -v ${PWD}:/workspace/ai_core ai-core:latest bash
```

Nếu gặp lỗi container name đã tồn tại:

```powershell
docker rm ai_core_container
```

Rồi chạy lại lệnh `docker run`.

## 5. Chạy tracking trong container

Trong shell của container:

```bash
cd /workspace/ai_core
python3 demo/demo.py --input videos/demo.mp4 --output videos/result_demo.mp4
```

Nếu chạy xong, terminal sẽ in dạng:

```text
Input: videos/demo.mp4
Output: videos/result_demo.mp4
Frames processed: <number>
Elapsed seconds: <number>
```

## 6. Kiểm tra output

Trong container:

```bash
ls -lh videos/result_demo.mp4
```

File phải tồn tại và dung lượng lớn hơn `0`.

Output nằm ở máy host tại:

```text
D:\IMP_Multi_Object_Tracking_across_Cameras\backend\ai_core\videos\result_demo.mp4
```

## 7. Chạy với video khác

Copy video vào thư mục:

```text
backend/ai_core/videos/
```

Ví dụ file mới là `my_video.mp4`, chạy:

```bash
python3 demo/demo.py --input videos/my_video.mp4 --output videos/my_video_result.mp4
```

## 8. Kiểm tra Docker có thấy GPU không

Chạy trên PowerShell host:

```powershell
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

Nếu lệnh này lỗi, cần sửa Docker Desktop, WSL2 backend, hoặc NVIDIA driver trước khi chạy dự án.

## 9. Lỗi thường gặp

### `TensorRT model not found`

Thiếu file:

```text
pretrained/model_trt.pth
```

Kiểm tra trong container:

```bash
ls -lh pretrained/model_trt.pth
```

### `Failed to open input video`

Sai path hoặc thiếu video input.

Kiểm tra:

```bash
ls -lh videos/demo.mp4
```

### `Failed to open output video writer`

OpenCV không tạo được file output. Thử xóa file cũ và chạy lại:

```bash
rm -f videos/result_demo.mp4
python3 demo/demo.py --input videos/demo.mp4 --output videos/result_demo.mp4
```

### `docker: could not select device driver ... gpu`

Docker chưa dùng được NVIDIA GPU. Kiểm tra bằng:

```powershell
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## 10. Lệnh nhanh

Từ PowerShell host:

```powershell
cd D:\IMP_Multi_Object_Tracking_across_Cameras\backend\ai_core
docker build -t ai-core:latest .
docker run --rm --gpus all -v ${PWD}:/workspace/ai_core ai-core:latest python3 demo/demo.py --input videos/demo.mp4 --output videos/result_demo.mp4
```
