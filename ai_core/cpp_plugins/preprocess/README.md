# CUDA ImageNet Preprocessing Plugin

This plugin implements a high-performance **CUDA kernel** to perform ImageNet normalization on batched RGBA input frames from NVIDIA DeepStream.

## Why Custom Preprocessing?

The standard DeepStream `nvinfer` element only supports a single pixel-scaling factor and a single mean subtraction vector. However, YOLOX requires a more complex normalization:
1.  Scaling pixels to [0, 1] range.
2.  Subtracting channel-wise means (`0.485, 0.456, 0.406`).
3.  Dividing by channel-wise standard deviations (`0.229, 0.224, 0.225`).

By implementing this in **CUDA**, the operation is performed on the GPU at almost zero cost, ensuring the input to the TensorRT model is mathematically identical to the Python training environment.

## Implementation Details

- **Kernel**: `normalize_imagenet_kernel` converts batched RGBA (Interleaved HWC) to RGB (Planar CHW) float32.
- **Precision**: Uses `fp32` for maximal accuracy.
- **Optimization**: Uses `__constant__` memory for normalization statistics to achieve peak memory throughput.

## Build and Integration

Refer to the root `cpp_plugins/README.md` for build instructions. Once compiled into `libyolox_preprocess.so`, it can be integrated into a custom GStreamer element or used as a standalone pre-processing library for the Jetson pipeline.
