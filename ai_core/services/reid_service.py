from dataclasses import dataclass, field
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return vector
    return vector / norm


@dataclass
class TrackletSample:
    crop: np.ndarray
    confidence: float
    quality: float
    frame_id: int


@dataclass
class EdgeTracklet:
    camera_id: str
    local_id: int
    first_frame: int
    last_frame: int
    global_id: Optional[int] = None
    samples: List[TrackletSample] = field(default_factory=list)
    frames_seen: int = 0
    last_packet_frame: int = 0


@dataclass
class TrackletPacket:
    camera_id: str
    local_id: int
    frame_id: int
    vector: np.ndarray
    samples: int


class ReIDService:
    """
    In-process MTMCT ReID service following an Edge-to-Central architecture:

    Edge side:
    - Receives local tracks from YOLOX + OC-SORT.
    - Applies anti-occlusion/quality gate.
    - Builds clean tracklet windows.
    - Extracts OSNet embeddings and fuses them using confidence weights.

    Central side:
    - Receives tracklet packets.
    - Applies optional spatio-temporal candidate filtering.
    - Runs cosine-similarity + Hungarian association.
    - Maintains a multi-view gallery per global ID.
    """

    def __init__(
            self,
            tracklet_window_size=10,
            min_window_samples=6,
            min_tracklet_frames=30,
            window_stride=12,
            min_detection_conf=0.50,
            max_occlusion_iou=0.30,
            edge_margin=4,
            min_box_area=1200,
            match_threshold=0.56,
            match_margin=0.01,
            gallery_size=5,
            ema_similarity=0.85,
            new_view_similarity=0.75,
            ema_alpha=0.90,
            association_batch_size=8,
            max_pending_packets=16,
            same_camera_exclusion_frames=0,
            debug=True
    ):
        self.lock = threading.Lock()

        self.tracklet_window_size = tracklet_window_size
        self.min_window_samples = min_window_samples
        self.min_tracklet_frames = min_tracklet_frames
        self.window_stride = window_stride
        self.min_detection_conf = min_detection_conf
        self.max_occlusion_iou = max_occlusion_iou
        self.edge_margin = edge_margin
        self.min_box_area = min_box_area

        self.match_threshold = match_threshold
        self.match_margin = match_margin
        self.gallery_size = gallery_size
        self.ema_similarity = ema_similarity
        self.new_view_similarity = new_view_similarity
        self.ema_alpha = ema_alpha
        self.association_batch_size = association_batch_size
        self.max_pending_packets = max_pending_packets
        self.same_camera_exclusion_frames = same_camera_exclusion_frames
        self.debug = debug

        self.next_global_id = 1
        self.edge_tracklets: Dict[Tuple[str, int], EdgeTracklet] = {}
        self.local_to_global: Dict[Tuple[str, int], int] = {}
        self.pending_packets: List[TrackletPacket] = []
        self.global_gallery = {}

    def update(
            self,
            camera_id: str,
            frame_id: int,
            frame: np.ndarray,
            local_track_ids: list,
            bboxes_xyxy: list,
            scores: list,
            embedder
    ) -> list:
        """
        Update edge tracklets and central associations.
        Returns global IDs aligned with local_track_ids. A value may be None
        until a tracklet window has enough clean samples and central matching runs.
        """
        with self.lock:
            outputs = []
            if len(local_track_ids) == 0:
                return outputs

            if not scores:
                scores = [1.0] * len(local_track_ids)

            qualities = self._compute_quality_scores(frame, bboxes_xyxy, scores)
            for local_id, bbox, score, quality in zip(local_track_ids, bboxes_xyxy, scores, qualities):
                local_id = int(local_id)
                key = (camera_id, local_id)
                tracklet = self.edge_tracklets.get(key)
                if tracklet is None:
                    tracklet = EdgeTracklet(
                        camera_id=camera_id,
                        local_id=local_id,
                        first_frame=frame_id,
                        last_frame=frame_id
                    )
                    self.edge_tracklets[key] = tracklet

                tracklet.frames_seen += 1
                tracklet.last_frame = frame_id

                if quality > 0.0:
                    crop = self._crop(frame, bbox)
                    if crop is not None:
                        self._add_sample(tracklet, TrackletSample(crop, float(score), quality, frame_id))

                if self._should_emit_packet(tracklet, frame_id):
                    packet = self._build_packet(tracklet, embedder, frame_id)
                    if packet is not None:
                        self.pending_packets.append(packet)
                        tracklet.last_packet_frame = frame_id

                outputs.append(self.local_to_global.get(key))

            if len(self.pending_packets) >= self.association_batch_size:
                self._associate_pending_packets()
            elif len(self.pending_packets) >= self.max_pending_packets:
                self._associate_pending_packets()

            return [self.local_to_global.get((camera_id, int(local_id))) for local_id in local_track_ids]

    def flush(self):
        """Force association of remaining pending packets."""
        with self.lock:
            self._associate_pending_packets()

    def _compute_quality_scores(self, frame: np.ndarray, bboxes: list, scores: list) -> List[float]:
        frame_h, frame_w = frame.shape[:2]
        qualities = []
        for index, (bbox, score) in enumerate(zip(bboxes, scores)):
            x1, y1, x2, y2 = bbox
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            area = width * height
            aspect = width / max(height, 1.0)
            max_iou = 0.0

            for other_index, other_bbox in enumerate(bboxes):
                if other_index == index:
                    continue
                max_iou = max(max_iou, self._iou(bbox, other_bbox))

            near_edge = (
                x1 <= self.edge_margin or
                y1 <= self.edge_margin or
                x2 >= frame_w - self.edge_margin or
                y2 >= frame_h - self.edge_margin
            )

            if score < self.min_detection_conf:
                qualities.append(0.0)
                continue
            if area < self.min_box_area:
                qualities.append(0.0)
                continue
            if aspect > 0.95 or aspect < 0.12:
                qualities.append(0.0)
                continue
            if max_iou > self.max_occlusion_iou:
                qualities.append(0.0)
                continue
            if near_edge:
                qualities.append(0.0)
                continue

            area_score = min(1.0, area / 16000.0)
            aspect_score = 1.0 - min(1.0, abs(aspect - 0.42) / 0.42)
            occlusion_score = 1.0 - min(1.0, max_iou / self.max_occlusion_iou)
            qualities.append(0.50 * float(score) + 0.20 * area_score + 0.15 * aspect_score + 0.15 * occlusion_score)

        return qualities

    def _add_sample(self, tracklet: EdgeTracklet, sample: TrackletSample):
        tracklet.samples.append(sample)
        tracklet.samples.sort(key=lambda item: item.quality, reverse=True)
        if len(tracklet.samples) > self.tracklet_window_size:
            tracklet.samples = tracklet.samples[:self.tracklet_window_size]

    def _should_emit_packet(self, tracklet: EdgeTracklet, frame_id: int) -> bool:
        if tracklet.frames_seen < self.min_tracklet_frames:
            return False
        if len(tracklet.samples) < self.min_window_samples:
            return False
        if tracklet.last_packet_frame == 0:
            return True
        return frame_id - tracklet.last_packet_frame >= self.window_stride

    def _build_packet(self, tracklet: EdgeTracklet, embedder, frame_id: int) -> Optional[TrackletPacket]:
        selected = sorted(tracklet.samples, key=lambda item: item.quality, reverse=True)[:self.tracklet_window_size]
        crops = [sample.crop for sample in selected]
        confidences = np.array([sample.confidence for sample in selected], dtype=np.float32)

        embeddings = embedder.extract_crops(crops)
        if len(embeddings) == 0:
            return None

        weights = confidences / max(float(np.sum(confidences)), 1e-12)
        fused = _l2_normalize(np.sum(embeddings * weights[:, None], axis=0))
        tracklet.samples.clear()

        return TrackletPacket(
            camera_id=tracklet.camera_id,
            local_id=tracklet.local_id,
            frame_id=frame_id,
            vector=fused,
            samples=len(selected)
        )

    def _associate_pending_packets(self):
        if not self.pending_packets:
            return

        packets = self.pending_packets
        self.pending_packets = []

        if not self.global_gallery:
            first_packet = packets.pop(0)
            gid = self._create_global(first_packet)
            self._bind(first_packet, gid)
            if self.debug:
                print(
                    f"[ReID] NEW   {first_packet.camera_id} L{first_packet.local_id} -> G{gid} "
                    f"best=0.000 samples={first_packet.samples}"
                )
            if not packets:
                return

        packets = self._refresh_already_bound_packets(packets)
        if not packets:
            return

        global_ids = list(self.global_gallery.keys())
        cost_matrix = np.ones((len(packets), len(global_ids)), dtype=np.float32)
        score_matrix = np.zeros((len(packets), len(global_ids)), dtype=np.float32)

        for row, packet in enumerate(packets):
            for col, gid in enumerate(global_ids):
                if not self._is_valid_candidate(packet, gid):
                    continue
                score = self._gallery_similarity(packet.vector, self.global_gallery[gid]["vectors"])
                score_matrix[row, col] = score
                cost_matrix[row, col] = 1.0 - score

        matched_rows = set()
        matched_cols = set()
        if len(packets) > 0 and len(global_ids) > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for row, col in zip(row_ind, col_ind):
                score = float(score_matrix[row, col])
                if score < self.match_threshold:
                    continue
                second = self._second_best_score(score_matrix[row], col)
                if score - second < self.match_margin:
                    continue

                packet = packets[row]
                gid = global_ids[col]
                self._bind(packet, gid)
                self._update_gallery(gid, packet)
                matched_rows.add(row)
                matched_cols.add(col)
                if self.debug:
                    print(
                        f"[ReID] MATCH {packet.camera_id} L{packet.local_id} -> G{gid} "
                        f"score={score:.3f} second={second:.3f} samples={packet.samples}"
                    )

        for row, packet in enumerate(packets):
            if row in matched_rows:
                continue
            retry = self._find_best_gallery(packet)
            if retry is not None:
                gid, score, second = retry
                self._bind(packet, gid)
                self._update_gallery(gid, packet)
                if self.debug:
                    print(
                        f"[ReID] MATCH {packet.camera_id} L{packet.local_id} -> G{gid} "
                        f"score={score:.3f} second={second:.3f} samples={packet.samples}"
                    )
                continue

            best_score = float(np.max(score_matrix[row])) if score_matrix.shape[1] else 0.0
            gid = self._create_global(packet)
            self._bind(packet, gid)
            if self.debug:
                print(
                    f"[ReID] NEW   {packet.camera_id} L{packet.local_id} -> G{gid} "
                    f"best={best_score:.3f} samples={packet.samples}"
                )

    def _refresh_already_bound_packets(self, packets: List[TrackletPacket]) -> List[TrackletPacket]:
        unbound_packets = []
        for packet in packets:
            key = (packet.camera_id, packet.local_id)
            gid = self.local_to_global.get(key)
            if gid is None:
                unbound_packets.append(packet)
                continue

            if gid not in self.global_gallery:
                unbound_packets.append(packet)
                continue

            self._update_gallery(gid, packet)
            if self.debug:
                print(
                    f"[ReID] KEEP  {packet.camera_id} L{packet.local_id} -> G{gid} "
                    f"samples={packet.samples}"
                )

        return unbound_packets

    def _passes_spatio_temporal_filter(self, packet: TrackletPacket, gid: int) -> bool:
        # No explicit topology is configured yet, so search all global IDs.
        # This method is the hook for camera-neighbor and time-window gating.
        return True

    def _is_valid_candidate(self, packet: TrackletPacket, gid: int) -> bool:
        if not self._passes_spatio_temporal_filter(packet, gid):
            return False
        return not self._has_same_camera_active_conflict(packet, gid)

    def _has_same_camera_active_conflict(self, packet: TrackletPacket, gid: int) -> bool:
        """
        A single global identity cannot explain two active local tracks in the
        same camera. Without this guard, a loose ReID threshold can collapse all
        same-frame detections into the first created global ID.
        """
        packet_key = (packet.camera_id, packet.local_id)
        for key, mapped_gid in self.local_to_global.items():
            if mapped_gid != gid or key == packet_key:
                continue

            camera_id, _ = key
            if camera_id != packet.camera_id:
                continue

            tracklet = self.edge_tracklets.get(key)
            if tracklet is None:
                continue

            if packet.frame_id - tracklet.last_frame <= self.same_camera_exclusion_frames:
                return True

        return False

    def _find_best_gallery(self, packet: TrackletPacket):
        if not self.global_gallery:
            return None

        scores = []
        for gid, data in self.global_gallery.items():
            if not self._is_valid_candidate(packet, gid):
                continue
            scores.append((gid, self._gallery_similarity(packet.vector, data["vectors"])))

        if not scores:
            return None

        scores.sort(key=lambda item: item[1], reverse=True)
        best_gid, best_score = scores[0]
        second = scores[1][1] if len(scores) > 1 else -1.0
        if best_score < self.match_threshold:
            return None
        if best_score - second < self.match_margin:
            return None
        return best_gid, best_score, second

    def _gallery_similarity(self, query: np.ndarray, gallery_vectors: List[np.ndarray]) -> float:
        vectors = np.array(gallery_vectors)
        similarities = np.dot(vectors, query)
        return float(np.max(similarities))

    def _second_best_score(self, scores: np.ndarray, best_col: int) -> float:
        if len(scores) <= 1:
            return -1.0
        masked = scores.copy()
        masked[best_col] = -1.0
        return float(np.max(masked))

    def _create_global(self, packet: TrackletPacket) -> int:
        gid = self.next_global_id
        self.next_global_id += 1
        self.global_gallery[gid] = {
            "vectors": [packet.vector],
            "timestamps": [packet.frame_id],
            "cameras": {packet.camera_id},
            "last_seen": packet.frame_id,
        }
        return gid

    def _bind(self, packet: TrackletPacket, gid: int):
        key = (packet.camera_id, packet.local_id)
        self.local_to_global[key] = gid
        tracklet = self.edge_tracklets.get(key)
        if tracklet is not None:
            tracklet.global_id = gid

    def _update_gallery(self, gid: int, packet: TrackletPacket):
        gallery = self.global_gallery[gid]
        vectors = gallery["vectors"]
        similarities = [float(np.dot(vector, packet.vector)) for vector in vectors]
        best_index = int(np.argmax(similarities))
        best_similarity = similarities[best_index]

        gallery["cameras"].add(packet.camera_id)
        gallery["last_seen"] = packet.frame_id

        if best_similarity >= self.ema_similarity:
            vectors[best_index] = _l2_normalize(
                self.ema_alpha * vectors[best_index] + (1.0 - self.ema_alpha) * packet.vector
            )
            gallery["timestamps"][best_index] = packet.frame_id
            return

        if best_similarity <= self.new_view_similarity:
            if len(vectors) >= self.gallery_size:
                vectors.pop(0)
                gallery["timestamps"].pop(0)
            vectors.append(packet.vector)
            gallery["timestamps"].append(packet.frame_id)
            return

        vectors[best_index] = _l2_normalize(0.95 * vectors[best_index] + 0.05 * packet.vector)
        gallery["timestamps"][best_index] = packet.frame_id

    def save_database(self, path: str):
        """Persist current in-memory vector database for inspection/reuse."""
        with self.lock:
            self._associate_pending_packets()
            payload = {}
            metadata = []
            for gid, data in self.global_gallery.items():
                key = f"global_{gid}"
                payload[key] = np.array(data["vectors"], dtype=np.float32)
                metadata.append({
                    "global_id": gid,
                    "num_vectors": len(data["vectors"]),
                    "cameras": sorted(data["cameras"]),
                    "last_seen": data["last_seen"],
                })
            payload["metadata"] = np.array(metadata, dtype=object)
            np.savez_compressed(path, **payload)

    def _crop(self, frame: np.ndarray, bbox) -> Optional[np.ndarray]:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox[:4])
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def _iou(self, box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0.0:
            return 0.0
        return inter_area / union
