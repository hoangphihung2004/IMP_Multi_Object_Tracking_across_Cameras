"""
Tracking Service Implementation.
Wraps the high-performance C++ OCSort engine using ctypes.
Ensures 100% parameter control from OCSortConfig.
"""

import ctypes
import os
import numpy as np
from typing import Tuple, List, Optional
from services.analytics.base import BaseTrackingService
from core.config import OCSortConfig


# --- C-API Structures ---

class OCSortConfig_C(ctypes.Structure):
    _fields_ = [
        ("track_thresh", ctypes.c_float),
        ("iou_thresh", ctypes.c_float),
        ("max_age", ctypes.c_int),
        ("min_hits", ctypes.c_int),
        ("delta_t", ctypes.c_int),
        ("inertia", ctypes.c_float),
        ("use_byte", ctypes.c_bool),
        ("asso_func_idx", ctypes.c_int),
    ]

class Detection_C(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x2", ctypes.c_float),
        ("y2", ctypes.c_float),
        ("score", ctypes.c_float),
        ("class_id", ctypes.c_int),
    ]

class Track_C(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x2", ctypes.c_float),
        ("y2", ctypes.c_float),
        ("track_id", ctypes.c_int),
        ("class_id", ctypes.c_int),
        ("score", ctypes.c_float),
    ]


class TrackingService(BaseTrackingService):
    """
    High-performance Tracking Service using C++ OCSort backend.
    """

    def __init__(self, config: OCSortConfig, lib_path: Optional[str] = None) -> None:
        """
        Initialize the C++ OCSort engine via ctypes.
        """
        self.config = config
        
        # Determine library path (defaulting to the cpp_plugins build folder)
        if lib_path is None:
            # Adjust this path based on your actual build directory on Jetson
            lib_path = os.path.join(os.getcwd(), "cpp_plugins", "lib", "libocsort_api.so")
            if not os.path.exists(lib_path):
                # Fallback for Windows or common OS names
                lib_path = lib_path.replace(".so", ".dll") if os.name == "nt" else lib_path

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"OCSort shared library not found at: {lib_path}")

        # Load the library
        self.lib = ctypes.CDLL(lib_path)

        # Map C-API functions
        self.lib.ocsort_create.argtypes = [OCSortConfig_C]
        self.lib.ocsort_create.restype = ctypes.c_void_p

        self.lib.ocsort_update.argtypes = [
            ctypes.c_void_p, 
            ctypes.POINTER(Detection_C), 
            ctypes.c_int, 
            ctypes.POINTER(Track_C), 
            ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.ocsort_update.restype = None

        self.lib.ocsort_delete.argtypes = [ctypes.c_void_p]
        self.lib.ocsort_delete.restype = None

        # Create the OCSort instance with full config control
        c_config = OCSortConfig_C(
            track_thresh=config.track_thresh,
            iou_thresh=config.iou_thresh,
            max_age=config.max_age,
            min_hits=config.min_hits,
            delta_t=config.delta_t,
            inertia=config.inertia,
            use_byte=config.use_byte,
            asso_func_idx=1 if config.asso_func == "giou" else 0
        )
        self.tracker_handle = self.lib.ocsort_create(c_config)
        
        # Pre-allocate result buffers (max 100 tracks per frame)
        self.max_tracks = 100
        self.tracks_buffer = (Track_C * self.max_tracks)()
        self.num_tracks = ctypes.c_int(0)

    def update(
        self, 
        detections: np.ndarray, 
        img_info: Tuple[int, int], 
        img_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Pass detections to the C++ engine and return track results.
        
        Args:
            detections (np.ndarray): Array of [x1, y1, x2, y2, score, class_id]
        """
        num_dets = len(detections)
        if num_dets == 0:
            # Still call update with 0 dets to age tracks in Kalman Filter
            c_dets = None
        else:
            # Convert numpy array to C-style detection array
            c_dets = (Detection_C * num_dets)()
            for i in range(num_dets):
                c_dets[i].x1 = float(detections[i, 0])
                c_dets[i].y1 = float(detections[i, 1])
                c_dets[i].x2 = float(detections[i, 2])
                c_dets[i].y2 = float(detections[i, 3])
                c_dets[i].score = float(detections[i, 4])
                c_dets[i].class_id = int(detections[i, 5])

        # Execute C++ Update
        self.lib.ocsort_update(
            self.tracker_handle, 
            c_dets, 
            num_dets, 
            self.tracks_buffer, 
            ctypes.byref(self.num_tracks)
        )

        n_results = self.num_tracks.value
        if n_results == 0:
            return np.empty((0, 7))

        # Convert C results back to NumPy array [x1, y1, x2, y2, track_id, class_id, score]
        results = np.zeros((n_results, 7), dtype=np.float32)
        for i in range(n_results):
            results[i, 0] = self.tracks_buffer[i].x1
            results[i, 1] = self.tracks_buffer[i].y1
            results[i, 2] = self.tracks_buffer[i].x2
            results[i, 3] = self.tracks_buffer[i].y2
            results[i, 4] = self.tracks_buffer[i].track_id
            results[i, 5] = self.tracks_buffer[i].class_id
            results[i, 6] = self.tracks_buffer[i].score

        # Return raw tracked results — geometric filtering (aspect_ratio, min_box_area)
        # is applied by the CALLER (probe.py), exactly like legacy Python code where
        # tracking_service.predict() filters AFTER tracker.update() returns.
        return results

    def __del__(self):
        """Ensure the C++ object is deleted when the Python object is collected."""
        if hasattr(self, 'lib') and hasattr(self, 'tracker_handle'):
            self.lib.ocsort_delete(self.tracker_handle)