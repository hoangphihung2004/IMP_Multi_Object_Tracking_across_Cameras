"""
Camera / Video Source Validator.
Pre-flight check: device connected? basic params OK?
"""

import os
import subprocess
import re
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger


@dataclass
class CameraInfo:
    """Kết quả kiểm tra camera."""
    source_uri: str
    is_valid: bool = False
    width: int = 0
    height: int = 0
    fps: float = 0.0
    pixel_format: str = ""
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "✅ OK" if self.is_valid else "❌ FAIL"
        lines = [f"  Source: {self.source_uri}  [{status}]"]
        if self.width and self.height:
            lines.append(f"  Resolution: {self.width}x{self.height} @ {self.fps:.0f}fps")
        if self.pixel_format:
            lines.append(f"  Format: {self.pixel_format}")
        for err in self.errors:
            lines.append(f"  ❌ {err}")
        return "\n".join(lines)


class CameraValidator:
    """
    Kiểm tra đầu vào video trước khi chạy pipeline.

    Usage:
        info = CameraValidator.check("/dev/video0")
        if not info.is_valid:
            sys.exit(1)
    """

    @staticmethod
    def check(source_uri: str) -> CameraInfo:
        """Kiểm tra source có kết nối và sẵn sàng không."""
        info = CameraInfo(source_uri=source_uri)

        if source_uri.startswith("/dev/video"):
            CameraValidator._check_usb(info)
        elif source_uri.startswith("rtsp://"):
            CameraValidator._check_rtsp(info)
        elif os.path.isfile(source_uri):
            info.is_valid = True
            size_mb = os.path.getsize(source_uri) / (1024 * 1024)
            logger.info(f"[Camera] Video file OK ({size_mb:.1f} MB)")
        else:
            info.errors.append(f"Source not found: {source_uri}")

        logger.info(info.summary())
        return info

    # ── USB Camera ───────────────────────────────────────────
    @staticmethod
    def _check_usb(info: CameraInfo) -> None:
        """Kiểm tra USB camera: tồn tại + đọc được + thông số cơ bản."""
        if not os.path.exists(info.source_uri):
            info.errors.append(f"Device not found: {info.source_uri}")
            info.errors.append("Chạy 'ls /dev/video*' để xem danh sách camera.")
            return

        if not os.access(info.source_uri, os.R_OK):
            info.errors.append(f"Permission denied: {info.source_uri}")
            info.errors.append(f"Thử: sudo chmod 666 {info.source_uri}")
            return

        # Query v4l2-ctl nếu có
        try:
            result = subprocess.run(
                ["v4l2-ctl", f"--device={info.source_uri}", "--all"],
                capture_output=True, text=True, timeout=5,
            )
            output = result.stdout

            if "Video Capture" not in output:
                info.errors.append("Device không hỗ trợ video capture.")
                return

            # Parse resolution hiện tại
            wh = re.search(r"Width/Height\s*:\s*(\d+)/(\d+)", output)
            if wh:
                info.width = int(wh.group(1))
                info.height = int(wh.group(2))

            # Parse pixel format
            fmt = re.search(r"Pixel Format\s*:\s*'(\w+)'", output)
            if fmt:
                info.pixel_format = fmt.group(1)

        except FileNotFoundError:
            logger.debug("[Camera] v4l2-ctl not found, skipping detail check.")
        except subprocess.TimeoutExpired:
            logger.warning("[Camera] v4l2-ctl timeout.")

        info.is_valid = True

    # ── RTSP Stream ──────────────────────────────────────────
    @staticmethod
    def _check_rtsp(info: CameraInfo) -> None:
        """Kiểm tra RTSP: ping host."""
        host_match = re.search(r"rtsp://(?:[^@]+@)?([^:/]+)", info.source_uri)
        if not host_match:
            info.errors.append(f"Không parse được host từ: {info.source_uri}")
            return

        host = host_match.group(1)
        try:
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "2", host],
                capture_output=True, timeout=5,
            )
            if result.returncode != 0:
                info.errors.append(f"Host không phản hồi: {host}")
                return
        except Exception:
            info.errors.append(f"Không ping được: {host}")
            return

        info.is_valid = True
        logger.info(f"[Camera] RTSP host {host} reachable.")
