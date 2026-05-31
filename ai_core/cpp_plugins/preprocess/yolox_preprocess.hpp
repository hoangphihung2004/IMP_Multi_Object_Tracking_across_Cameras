/*
 * Custom CUDA Preprocess Header for YOLOX
 * 
 * Defines the C-API interface for the ImageNet Normalization kernel.
 * This can be linked directly into a custom GStreamer element or 
 * loaded as a custom library by nvdspreprocess.
 */

#ifndef YOLOX_PREPROCESS_HPP
#define YOLOX_PREPROCESS_HPP

#include <cuda_runtime.h>

extern "C" {

/**
 * @brief Normalizes a batched RGBA image buffer using ImageNet statistics.
 * 
 * DeepStream typically provides frames in RGBA format in device memory.
 * This kernel converts RGBA -> RGB -> Float, scales to 0-1, and subtracts 
 * mean and divides by standard deviation for exactly matching the Python 
 * PyTorch transform pipeline.
 *
 * @param d_out_tensor Pointer to pre-allocated float tensor memory on GPU, shape (N, C, H, W)
 * @param d_in_rgba Pointer to batched RGBA interleaved frame memory on GPU
 * @param batch_size Number of frames in the batch (N)
 * @param width Network input width (W)
 * @param height Network input height (H)
 * @param stream CUDA stream to execute on
 * @return cudaError_t Success or error code
 */
cudaError_t normalize_imagenet_rgba(
    float* d_out_tensor, 
    const unsigned char* d_in_rgba, 
    int batch_size, 
    int width, 
    int height, 
    cudaStream_t stream
);

}

#endif // YOLOX_PREPROCESS_HPP
