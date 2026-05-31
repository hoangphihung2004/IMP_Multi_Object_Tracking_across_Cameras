# C++ DeepStream Plugins

This directory contains high-performance C++ and CUDA components designed to offload heavy computation from the Python service layer to the Jetson hardware.

## Architecture

To achieve real-time performance (>50 FPS), critical tasks that involve pixel-level manipulation or large-scale array decoding are implemented as compiled C++ shared libraries (.so). 

These plugins integrate directly into the NVIDIA DeepStream SDK pipeline.

## Directory Structure

- **`preprocess/`**: Contains the custom CUDA kernel for ImageNet normalization. This ensures that input frames match the exact pre-processing requirements of the YOLOX model.
- **`yolox_parser/`**: Contains the custom bounding box parser for DeepStream's `nvinfer` element. It decodes the raw model output tensors into standard DeepStream `NvDsObjectMeta`.

## Key Benefits

1.  **Zero-Latency Normalization**: By performing ImageNet normalization (`mean/std`) directly on the GPU using CUDA, we avoid expensive CPU-GPU memory transfers and Python-level loop overhead.
2.  **Hardware-Accelerated Parsing**: Decoding thousands of bounding box proposals and performing Non-Maximum Suppression (NMS) in C++ ensures the host CPU remains free for business logic services.
3.  **Accuracy Parity**: These plugins use the exact same mathematical constants as the original PyTorch/Python implementation, ensuring 100% accuracy matching.

## Building the Plugins

Each subdirectory contains its own `CMakeLists.txt`. To build:

```bash
cd cpp_plugins/<sub_dir>
mkdir build && cd build
cmake ..
make
```

The resulting `.so` file can then be referenced in the DeepStream `config_infer.txt` or loaded dynamically by the Python `PipelineManager`.
