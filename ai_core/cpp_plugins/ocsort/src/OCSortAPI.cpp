/*
 * OCSort C-API Implementation.
 * Bridges between Python ctypes (C structs) and the C++ OCSort engine (Eigen).
 *
 * This file implements the three functions declared in OCSortAPI.h:
 *   - ocsort_create:  Instantiates an OCSort object on the heap.
 *   - ocsort_update:  Converts Detection_C[] → Eigen::MatrixXf, runs update(),
 *                     and writes results back into Track_C[].
 *   - ocsort_delete:  Frees the OCSort instance.
 */

#include "../include/OCSortAPI.h"
#include "../include/OCSort.hpp"

#include <string>
#include <vector>

static const char* ASSO_FUNC_NAMES[] = {"iou", "giou", "ciou", "diou"};

extern "C" {

void* ocsort_create(OCSortConfig_C config) {
    // Map asso_func_idx to string name
    std::string asso_name = "iou";
    if (config.asso_func_idx >= 0 && config.asso_func_idx < 4) {
        asso_name = ASSO_FUNC_NAMES[config.asso_func_idx];
    }

    // Allocate OCSort instance on the heap and return opaque handle
    auto* tracker = new ocsort::OCSort(
        config.track_thresh,
        config.max_age,
        config.min_hits,
        config.iou_thresh,
        config.delta_t,
        asso_name,
        config.inertia,
        config.use_byte
    );

    return static_cast<void*>(tracker);
}

void ocsort_update(
    void* handle,
    Detection_C* dets,
    int num_dets,
    Track_C* out_tracks,
    int* out_num_tracks)
{
    auto* tracker = static_cast<ocsort::OCSort*>(handle);

    // Build Eigen matrix from C detections: [x1, y1, x2, y2, score, class_id]
    Eigen::MatrixXf det_matrix;
    if (num_dets > 0 && dets != nullptr) {
        det_matrix.resize(num_dets, 6);
        for (int i = 0; i < num_dets; ++i) {
            det_matrix(i, 0) = dets[i].x1;
            det_matrix(i, 1) = dets[i].y1;
            det_matrix(i, 2) = dets[i].x2;
            det_matrix(i, 3) = dets[i].y2;
            det_matrix(i, 4) = dets[i].score;
            det_matrix(i, 5) = static_cast<float>(dets[i].class_id);
        }
    } else {
        // Empty detection matrix (0 rows) to still age Kalman tracks
        det_matrix.resize(0, 6);
    }

    // Run OCSort update
    std::vector<Eigen::RowVectorXf> results = tracker->update(det_matrix);

    // Write results into pre-allocated Track_C buffer
    int n = static_cast<int>(results.size());
    *out_num_tracks = n;

    for (int i = 0; i < n; ++i) {
        // OCSort output format: [x1, y1, x2, y2, track_id, class_id, conf]
        out_tracks[i].x1       = results[i](0);
        out_tracks[i].y1       = results[i](1);
        out_tracks[i].x2       = results[i](2);
        out_tracks[i].y2       = results[i](3);
        out_tracks[i].track_id = static_cast<int>(results[i](4));
        out_tracks[i].class_id = static_cast<int>(results[i](5));
        out_tracks[i].score    = results[i](6);
    }
}

void ocsort_delete(void* handle) {
    auto* tracker = static_cast<ocsort::OCSort*>(handle);
    delete tracker;
}

} // extern "C"
