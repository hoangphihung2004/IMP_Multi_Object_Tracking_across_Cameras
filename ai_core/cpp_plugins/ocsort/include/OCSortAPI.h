/*
 * OCSort C-API Wrapper
 * Provides a stable C interface for the Python OCSortConfig to pass
 * parameters and detection data to the C++ OCSort engine.
 */

#ifndef OCSORT_API_H
#define OCSORT_API_H

#ifdef __cplusplus
extern "C" {
#endif

// Struct matching OCSortConfig in Python for direct binary passing
typedef struct {
    float track_thresh;
    float iou_thresh;
    int max_age;
    int min_hits;
    int delta_t;
    float inertia;
    bool use_byte;
    // Note: asso_func is passed as a string index or enum
    int asso_func_idx; 
} OCSortConfig_C;

// Struct for Bounding Box Detection
typedef struct {
    float x1, y1, x2, y2, score;
    int class_id;
} Detection_C;

// Struct for Tracked Result
typedef struct {
    float x1, y1, x2, y2;
    int track_id;
    int class_id;
    float score;
} Track_C;

/**
 * Initialize a new OCSort instance with full configuration control.
 * @return Handle to the OCSort instance.
 */
void* ocsort_create(OCSortConfig_C config);

/**
 * Update the tracker with new detections.
 * @param handle OCSort handle.
 * @param dets Array of Detection_C.
 * @param num_dets Number of detections.
 * @param out_tracks Array to write results into (pre-allocated by Python).
 * @param out_num_tracks Pointer to int to write current track count.
 */
void ocsort_update(void* handle, Detection_C* dets, int num_dets, Track_C* out_tracks, int* out_num_tracks);

/**
 * Cleanup and delete the OCSort instance.
 */
void ocsort_delete(void* handle);

#ifdef __cplusplus
}
#endif

#endif // OCSORT_API_H
