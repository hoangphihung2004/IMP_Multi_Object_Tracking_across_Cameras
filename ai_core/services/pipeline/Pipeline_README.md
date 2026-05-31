# PipelineManager: Local Two-Source DeepStream Pipeline

Tài liệu này mô tả pipeline runtime local sau khi cắt backend, counting, mapping và Janus/WebRTC.

## 1. Trách nhiệm

`PipelineManager` chịu trách nhiệm:

- Tạo 2 input branches từ video files.
- Gom 2 stream vào `nvstreammux` với batch size đọc từ config.
- Chạy YOLOX qua `nvinfer`.
- Tách stream bằng `nvstreamdemux`.
- Ghi mỗi stream ra một file mp4 riêng.
- Quản lý lifecycle: build, attach probe, run, stop.

## 2. Pipeline Structure

```mermaid
graph TD
    Src0[filesrc: videos/1-1.mp4] --> Demux0[qtdemux]
    Demux0 --> Parse0[h264parse]
    Parse0 --> Dec0[nvv4l2decoder]

    Src1[filesrc: videos/2-1.mp4] --> Demux1[qtdemux]
    Demux1 --> Parse1[h264parse]
    Parse1 --> Dec1[nvv4l2decoder]

    Dec0 --> Mux[nvstreammux batch=2]
    Dec1 --> Mux
    Mux --> Infer[nvinfer: YOLOX]
    Infer --> Probe[AnalyticsProbe: OCSort per source]
    Probe --> StreamDemux[nvstreamdemux]

    StreamDemux --> Q0[queue]
    Q0 --> Conv0[nvvideoconvert]
    Conv0 --> OSD0[nvdsosd]
    OSD0 --> Enc0[nvv4l2h264enc]
    Enc0 --> OutParse0[h264parse]
    OutParse0 --> Mux0[qtmux]
    Mux0 --> Sink0[filesink: output/out_0.mp4]

    StreamDemux --> Q1[queue]
    Q1 --> Conv1[nvvideoconvert]
    Conv1 --> OSD1[nvdsosd]
    OSD1 --> Enc1[nvv4l2h264enc]
    Enc1 --> OutParse1[h264parse]
    OutParse1 --> Mux1[qtmux]
    Mux1 --> Sink1[filesink: output/out_1.mp4]
```

## 3. Config-driven properties

Pipeline properties are controlled by:

- `config/sources_config.json`
- `config/pipeline_config.json`
- `config/yolox_config.json`

Không hardcode input/output path, width/height, batch size hoặc bitrate trong `manager.py`.

## 4. Runtime outputs

Kết quả nằm ở:

```text
output/out_0.mp4
output/out_1.mp4
```

## 5. Không còn dùng

Pipeline local không còn:

- Janus/WebRTC branch.
- UDP/RTP streaming.
- Backend WebSocket.
- Counting line OSD.
- Mapping overlay.
