/*
 * DeepStream Custom Bounding Box Parser for YOLOX
 * Parses the (1, 8400, 6) raw output tensor from YOLOX.
 * 
 * CRITICAL: The ONNX model is exported with decode_in_inference=False,
 * meaning the output contains RAW offsets, NOT decoded pixel coordinates.
 * This parser must perform the grid+stride decode step:
 *   decoded_cx = (raw_x + grid_x) * stride
 *   decoded_cy = (raw_y + grid_y) * stride
 *   decoded_w  = exp(raw_w) * stride
 *   decoded_h  = exp(raw_h) * stride
 *
 * Grid layout for 640x640 input:
 *   Stride  8: 80x80 = 6400 anchors
 *   Stride 16: 40x40 = 1600 anchors
 *   Stride 32: 20x20 = 400  anchors
 *   Total:              8400 anchors
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "nvdsinfer_custom_impl.h"

#define EXPORT_CUSTOM_PARSER extern "C" __attribute__((visibility("default")))

namespace {

    #ifndef YOLOX_NMS_THRESH
    #define YOLOX_NMS_THRESH 0.7
    #endif

    struct Bbox {
        float x1, y1, x2, y2, score;
        int class_id;
    };

    // Grid + Stride info for YOLOX decode
    struct GridAndStride {
        int grid_x;
        int grid_y;
        int stride;
    };

    // Pre-generate grid and stride for all 8400 anchors
    void generate_grids_and_strides(int input_w, int input_h,
                                     const std::vector<int>& strides,
                                     std::vector<GridAndStride>& grid_strides) {
        grid_strides.clear();
        for (int stride : strides) {
            int grid_w = input_w / stride;
            int grid_h = input_h / stride;
            for (int gy = 0; gy < grid_h; ++gy) {
                for (int gx = 0; gx < grid_w; ++gx) {
                    grid_strides.push_back({gx, gy, stride});
                }
            }
        }
    }

    float calculateIoU(const Bbox& a, const Bbox& b) {
        float inter_x1 = std::max(a.x1, b.x1);
        float inter_y1 = std::max(a.y1, b.y1);
        float inter_x2 = std::min(a.x2, b.x2);
        float inter_y2 = std::min(a.y2, b.y2);

        float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
        if (inter_area == 0.0f) return 0.0f;

        float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
        float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);

        return inter_area / (area_a + area_b - inter_area);
    }

    void nms(std::vector<Bbox>& boxes, float nms_thresh) {
        std::sort(boxes.begin(), boxes.end(), [](const Bbox& a, const Bbox& b) {
            return a.score > b.score;
        });

        std::vector<bool> keep(boxes.size(), true);
        for (size_t i = 0; i < boxes.size(); ++i) {
            if (!keep[i]) continue;
            for (size_t j = i + 1; j < boxes.size(); ++j) {
                if (keep[j] && boxes[i].class_id == boxes[j].class_id) {
                    if (calculateIoU(boxes[i], boxes[j]) > nms_thresh) {
                        keep[j] = false;
                    }
                }
            }
        }

        size_t keep_idx = 0;
        for (size_t i = 0; i < boxes.size(); ++i) {
            if (keep[i]) {
                boxes[keep_idx++] = boxes[i];
            }
        }
        boxes.resize(keep_idx);
    }

} // namespace

EXPORT_CUSTOM_PARSER bool NvDsInferParseCustomYOLOX(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList) 
{
    if (outputLayersInfo.empty()) {
        std::cerr << "[YOLOX Parser] Error: Missing output layer." << std::endl;
        return false;
    }

    const NvDsInferLayerInfo& outputLayer = outputLayersInfo[0];
    const float* output_data = (const float*)outputLayer.buffer;
    
    int num_proposals = outputLayer.inferDims.d[0];  // 8400
    int num_elements = outputLayer.inferDims.d[1];    // 6 (for 1 class)

    if (num_elements < 6) {
        std::cerr << "[YOLOX Parser] Error: Unexpected output dimensions: "
                  << num_proposals << "x" << num_elements << std::endl;
        return false;
    }

    int num_classes = num_elements - 5;

    // Generate grid+stride mapping for YOLOX decode
    // YOLOX uses strides [8, 16, 32] with input_size from networkInfo
    std::vector<int> strides = {8, 16, 32};
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_strides(networkInfo.width, networkInfo.height, strides, grid_strides);

    if ((int)grid_strides.size() != num_proposals) {
        std::cerr << "[YOLOX Parser] Warning: grid_strides size (" << grid_strides.size()
                  << ") != num_proposals (" << num_proposals << ")" << std::endl;
    }

    float global_det_thresh = detectionParams.perClassPreclusterThreshold[0]; 
    std::vector<Bbox> bboxes;

    // Decode predictions with grid+stride
    for (int i = 0; i < num_proposals && i < (int)grid_strides.size(); ++i) {
        const float* row = output_data + i * num_elements;
        
        float obj_conf = row[4];
        if (obj_conf < global_det_thresh) continue;

        // Find max class confidence
        int max_class_id = 0;
        float max_class_conf = row[5];
        for (int c = 1; c < num_classes; ++c) {
            if (row[5 + c] > max_class_conf) {
                max_class_conf = row[5 + c];
                max_class_id = c;
            }
        }

        float final_score = obj_conf * max_class_conf;

        float class_thresh = detectionParams.perClassPreclusterThreshold[max_class_id];
        if (final_score > class_thresh) {
            // === YOLOX Grid+Stride Decode ===
            // Raw output: [raw_cx, raw_cy, raw_w, raw_h, obj_conf, cls_conf]
            // Decoded:
            //   cx = (raw_cx + grid_x) * stride
            //   cy = (raw_cy + grid_y) * stride
            //   w  = exp(raw_w) * stride
            //   h  = exp(raw_h) * stride
            const GridAndStride& gs = grid_strides[i];
            float cx = (row[0] + gs.grid_x) * gs.stride;
            float cy = (row[1] + gs.grid_y) * gs.stride;
            float w  = std::exp(row[2]) * gs.stride;
            float h  = std::exp(row[3]) * gs.stride;

            // Convert center format to top-left / bottom-right
            float x1 = cx - w / 2.0f;
            float y1 = cy - h / 2.0f;
            float x2 = cx + w / 2.0f;
            float y2 = cy + h / 2.0f;

            // Clamp to network dimensions
            x1 = std::max(0.0f, std::min(x1, (float)networkInfo.width));
            y1 = std::max(0.0f, std::min(y1, (float)networkInfo.height));
            x2 = std::max(0.0f, std::min(x2, (float)networkInfo.width));
            y2 = std::max(0.0f, std::min(y2, (float)networkInfo.height));

            bboxes.push_back({x1, y1, x2, y2, final_score, max_class_id});
        }
    }

    // Apply NMS
    nms(bboxes, YOLOX_NMS_THRESH);

    // Convert to DeepStream Objects
    for (const auto& box : bboxes) {
        NvDsInferParseObjectInfo obj;
        obj.classId = box.class_id;
        obj.left = box.x1;
        obj.top = box.y1;
        obj.width = box.x2 - box.x1;
        obj.height = box.y2 - box.y1;
        obj.detectionConfidence = box.score;
        objectList.push_back(obj);
    }

    return true;
}

// Ensure the parser implementation is compliant
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYOLOX);
