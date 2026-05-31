## 1. Core Responsibilities

The OCSort C++ plugin is the **"Tracking Brain"** of the system. Its mission is to:
- **Maintain Identity**: Ensure that Person A remains ID #5 even if they are temporarily blocked by a tree or another person.
- **Smooth Trajectories**: Use Kalman Filtering to predict where an object will be in the next frame, reducing "jitter" in bounding box coordinates.
- **Filter Noise**: Reject low-confidence detections or "ghost" boxes that don't follow a consistent physical motion path.

## 2. Data Input/Output Specification (C-API)

To maintain absolute performance, data is passed as **flat binary memory buffers** via the C-API.

### **Input (From Python to C++)**
- **Configuration**: A `struct` containing thresholds (track_thresh, iou_thresh), ages, and algorithmic flags.
- **Detections**: A contiguous array of `Detection_C` structs. Each detection is:
  `[x1, y1, x2, y2, score, class_id]`
  - *Example*: `[100.5, 200.0, 150.2, 310.8, 0.95, 0]` (A person with 95% confidence).

### **Output (From C++ to Python)**
- **Tracks**: A contiguous array of `Track_C` structs written back into Python's pre-allocated memory. Each track is:
  `[x1, y1, x2, y2, track_id, class_id, score]`
  - *Example*: `[101.2, 201.5, 152.0, 312.0, 5, 0, 0.95]` (Person ID #5, coordinates slightly smoothed by Kalman Filter).

---

## 3. Algorithmic Overview

OCSort is a tracking-by-detection algorithm that improves upon SORT by introducing **Observation-Centric** mechanisms. It addresses three major flaws in standard Kalman-filter-based trackers:
1.  **High Sensitivity to Frame Rate**: Standard SORT fails if the object moves significantly between frames.
2.  **Inaccurate Re-identification**: Standard SORT loses IDs during short-term occlusions.
3.  **Velocity Bias**: Standard SORT's velocity estimation is often noisy.

### **Key Mechanisms in This Implementation:**
- **Observation-Centric Kalman Filter (OCKF)**: Updates the filter state only when a reliable detection is available, preventing the "drift" caused by purely predictive steps.
- **Velocity Directional Consistency (VDC)**: Adds a momentum-based cost to the association matrix, penalizing trackers that change direction too abruptly.
- **Inertia Scaling**: Weights the IoU cost with the object's previous velocity to favor smooth motion.

## 2. Critical Bug Fixes (Jetson Edition)

The original C++ port of OCSort contained 3 critical logic errors that were fixed in this version to match the original Python behavior and ensure 100% accuracy:

1.  **Comma Operator Bug (`KalmanFilter.cpp`)**: Fixed a syntax error in the covariance update `P = (I-KH)P` where a comma was used instead of an assignment, preventing the filter from ever updating its uncertainty.
2.  **Virtual Trajectory Frozen `dy`**: Fixed a logic error where the vertical velocity (`dy`) was never updated during long-term occlusions, causing tracks to drift vertically.
3.  **W/H Covariance Unfreeze**: Enabled dynamic covariance scaling for Width and Height to handle multi-scale object detection (zooming in/out).

## 3. The C-API Bridge (`OCSortAPI.h/cpp`)

To provide **Full Control** from Python, we use a C-compatible interface. This allows us to pass a single binary `struct` containing all 10 runtime parameters:

```cpp
void* ocsort_create(OCSortConfig_C config);
```
When Python calls this, it "hydrates" the C++ engine with the values from your `ocsort_config.json`. No hardcoded values are used.

## 4. File Breakdown

- **`OCSort.cpp`**: The primary orchestrator. It handles the `update()` loop, association rounds (including BYTE logic), and track creation/deletion.
- **`KalmanBoxTracker.cpp`**: Implements the state machine for a single object. Each tracked person has one `KalmanBoxTracker` instance.
- **`KalmanFilter.cpp`**: A custom implementation of the 7-state Kalman Filter (x, y, area, ratio, vx, vy, va).
- **`Association.cpp`**: Implements the Hungarian algorithm (via LapJV) and IoU/GIoU cost calculation.
- **`lapjv.cpp`**: A highly optimized Jonker-Volgenant linear assignment solver for the C++ backend.

## 5. Memory Management

- **Heap Allocation**: The `OCSort` instance is created on the heap via `ocsort_create()`.
- **Safety**: Python manages the lifecycle. When the Python `TrackingService` object is garbage collected or stopped, it calls `ocsort_delete()` to free all C++ memory, preventing leaks.
- **Batched I/O**: Detections and results are passed as flat C arrays to minimize overhead during the Python-C++ boundary crossings.

## 6. Building

Refer to the root `cpp_plugins/README.md` for standard CMake build instructions. Ensure **Eigen 3.4.0** is installed, as it is the core math library for all matrix transformations in this engine.
