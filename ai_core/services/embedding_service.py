import torch
import numpy as np
import cv2
import torchreid
import threading

class EmbeddingService:
    """Service to extract ReID embeddings using OSNet."""

    def __init__(
            self,
            model_name='osnet_x1_0',
            model_path='',
            device='cuda' if torch.cuda.is_available() else 'cpu',
            input_size=(256, 128)
    ):
        self.device = device
        self.lock = threading.Lock()
        self.input_height, self.input_width = input_size

        # Load OSNet model via torchreid feature extractor
        self.extractor = torchreid.utils.FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            device=device
        )
        print(f"EmbeddingService initialized with {model_name} on {device}")

    def _prepare_crop(self, crop: np.ndarray) -> np.ndarray:
        """Resize a person crop to OSNet input size."""
        if crop.size == 0:
            return np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)

        return cv2.resize(crop, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)

    def get_crops(self, frame: np.ndarray, bboxes: np.ndarray) -> list:
        """Crop bounding boxes from the frame."""
        crops = []
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox[:4])
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            # Avoid empty crops
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                crops.append(crop)
            else:
                # Handle edge case where bbox is out of bounds or invalid
                crops.append(np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8))
        return crops

    def extract(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """
        Crop images based on bboxes and extract embeddings.
        Returns: numpy array of shape (N, 512) containing embeddings.
        """
        if len(bboxes) == 0:
            return np.array([])

        crops = self.get_crops(frame, bboxes)
        return self.extract_crops(crops)

    def extract_crops(self, crops: list) -> np.ndarray:
        """
        Extract embeddings from already-cropped person images.
        Crops are normalized to OSNet input size before inference.
        """
        if len(crops) == 0:
            return np.array([])

        crops = [self._prepare_crop(crop) for crop in crops]

        # Extractor takes a list of numpy arrays (RGB images)
        # Assuming cv2 frame is BGR, we convert to RGB
        crops_rgb = [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops]

        with self.lock:
            embeddings = self.extractor(crops_rgb)

        # Convert tensor to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        # Normalize embeddings for cosine distance
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        embeddings = embeddings / norms
        return embeddings
