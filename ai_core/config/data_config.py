import json
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class OCSortConfig:

    json_path: Optional[str] = None

    device: str = "gpu"

    # detection
    conf: float = 0.1
    nms: float = 0.7
    tsize: int = 640
    ckpt: Optional[str] = None

    # inference options
    fp16: bool = False
    fuse: bool = False
    trt: bool = False
    trt_file: Optional[str] = None

    # tracking
    track_thresh: float = 0.6
    iou_thresh: float = 0.3
    use_byte: bool = False
    max_age: int = 30
    min_hits: int = 3
    delta_t: int = 3
    asso_func: str = "iou"
    inertia: float = 0.2
    aspect_ratio_thresh: float = 1.6
    min_box_area: float = 10

    def __post_init__(self):

        if self.json_path is None:
            return

        with open(self.json_path, "r") as f:
            config = json.load(f)

        for key, value in config.items():

            key = key.replace("-", "_")

            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Unknown config key: {key}")
