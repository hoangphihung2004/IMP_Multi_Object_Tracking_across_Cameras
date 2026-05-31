/*
 * Custom CUDA kernel for exact ImageNet normalization matching Python YOLOX.
 *
 * This kernel converts batched RGBA images (HWC format) from DeepStream 
 * into normalized CHW Float32 tensors used directly by TensorRT nvinfer.
 */

#include "yolox_preprocess.hpp"

// Standard ImageNet values used during YOLOX training
__constant__ float c_mean[3] = {0.485f, 0.456f, 0.406f};
__constant__ float c_std[3]  = {0.229f, 0.224f, 0.225f};

/**
 * CUDA Kernel: RGBA (HWC) -> RGB (CHW) Float32 Normalized
 */
__global__ void normalize_imagenet_kernel(
    float* out_tensor, 
    const unsigned char* in_rgba, 
    int batch_size, 
    int width, 
    int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || b >= batch_size) {
        return;
    }

    // Input index (RGBA interleaved HWC)
    // 4 elements per pixel
    int in_idx = b * (height * width * 4) + y * (width * 4) + x * 4;

    // DeepStream pad is usually RGBA
    float r = in_rgba[in_idx] / 255.0f;
    float g = in_rgba[in_idx + 1] / 255.0f;
    float b_val = in_rgba[in_idx + 2] / 255.0f;
    // float a = in_rgba[in_idx + 3] / 255.0f; // Alpha ignored

    // Normalize
    r = (r - c_mean[0]) / c_std[0];
    g = (g - c_mean[1]) / c_std[1];
    b_val = (b_val - c_mean[2]) / c_std[2];

    // Output index (CHW planar)
    int channel_stride = height * width;
    int batch_stride = 3 * channel_stride;
    
    int out_r_idx = b * batch_stride + 0 * channel_stride + y * width + x;
    int out_g_idx = b * batch_stride + 1 * channel_stride + y * width + x;
    int out_b_idx = b * batch_stride + 2 * channel_stride + y * width + x;

    out_tensor[out_r_idx] = r;
    out_tensor[out_g_idx] = g;
    out_tensor[out_b_idx] = b_val;
}

extern "C" {

cudaError_t normalize_imagenet_rgba(
    float* d_out_tensor, 
    const unsigned char* d_in_rgba, 
    int batch_size, 
    int width, 
    int height, 
    cudaStream_t stream)
{
    // Block dimensions for 2D image processing
    dim3 block(16, 16, 1);
    
    // Grid dimensions mapping to image size and batch
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y,
        batch_size
    );

    normalize_imagenet_kernel<<<grid, block, 0, stream>>>(
        d_out_tensor, 
        d_in_rgba, 
        batch_size, 
        width, 
        height
    );

    return cudaGetLastError();
}

} // extern "C"
