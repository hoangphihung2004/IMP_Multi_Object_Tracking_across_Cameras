"""
Result Publisher & Settings Subscriber Service.
Handles two-way WebSocket communication with the Django backend:
  1. Publishing frame results (tracks, counts, map positions) to the server.
  2. Subscribing to live setting updates (mapping/counting config changes).

Logic is preserved 100% from the legacy ai_core backend_gateway.py.
"""

import os
import json
import time
import threading
from typing import Any, Callable, Dict, Optional

import websocket
from loguru import logger

from services.gateway.base import BaseResultPublisher
from schemas.schemas import FrameData


class ResultPublisher(BaseResultPublisher):
    """
    Publishes per-frame pipeline results to the Django backend via WebSocket.

    Input:  FrameData (Pydantic model containing tracks, counts, map positions).
    Output: JSON payload sent over WebSocket to the server.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        bearer_token: Optional[str] = None,
        reconnect: bool = True,
    ) -> None:
        """
        Initialize the WebSocket publisher.

        Args:
            endpoint: WebSocket URL (e.g., ws://server:8000/ws/results/).
                      Falls back to RESULTS_WS_URL environment variable.
            bearer_token: Authentication token. Falls back to API_BEARER_TOKEN env var.
            reconnect: Whether to auto-reconnect on send failure.
        """
        self.endpoint = endpoint or os.getenv("RESULTS_WS_URL")
        self.bearer_token = bearer_token or os.getenv("API_BEARER_TOKEN")
        self.reconnect = reconnect
        self.ws: Optional[websocket.WebSocket] = None

    def _connect(self) -> None:
        """Establish WebSocket connection with optional auth header."""
        if not self.endpoint:
            raise ValueError("RESULTS_WS_URL is not configured.")

        headers = []
        if self.bearer_token:
            headers.append(f"Authorization: Bearer {self.bearer_token}")

        logger.info(f"[Publisher] Connecting -> {self.endpoint}")
        self.ws = websocket.create_connection(self.endpoint, header=headers)

    def send_frame(self, frame_data: FrameData) -> None:
        """
        Serialize and send a FrameData payload over WebSocket.

        Args:
            frame_data: Pydantic model with all frame-level results.
        """
        payload = frame_data.model_dump_json()

        if self.ws is None:
            self._connect()

        try:
            self.ws.send(payload)
        except Exception as e:
            logger.warning(f"[Publisher] Send failed: {e}")
            if self.reconnect:
                logger.info("[Publisher] Reconnecting...")
                self._connect()
                self.ws.send(payload)
            else:
                raise

    def close(self) -> None:
        """Gracefully close the WebSocket connection."""
        if self.ws:
            self.ws.close()
            self.ws = None
            logger.info("[Publisher] Connection closed.")


class SettingsSubscriber:
    """
    Listens for live configuration updates (Mapping & Counting)
    from the Django backend via WebSocket.

    When a new config is received, it calls the registered callback
    so the PipelineManager can hot-reload services without restarting.
    """

    def __init__(
        self,
        mapping_ws_url: Optional[str] = None,
        counting_ws_url: Optional[str] = None,
        bearer_token: Optional[str] = None,
        on_mapping_update: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_counting_update: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        """
        Initialize the settings subscriber.

        Args:
            mapping_ws_url: WebSocket URL for mapping config updates.
            counting_ws_url: WebSocket URL for counting config updates.
            bearer_token: Authentication token.
            on_mapping_update: Callback invoked with new mapping data dict.
            on_counting_update: Callback invoked with new counting data dict.
        """
        self.mapping_url = mapping_ws_url or os.getenv("MAPPING_SETTINGS_WS_URL")
        self.counting_url = counting_ws_url or os.getenv("COUNTING_SETTINGS_WS_URL")
        self.bearer_token = bearer_token or os.getenv("API_BEARER_TOKEN")

        self.on_mapping_update = on_mapping_update
        self.on_counting_update = on_counting_update

        self.mapping_ws: Optional[websocket.WebSocketApp] = None
        self.counting_ws: Optional[websocket.WebSocketApp] = None
        self._stop_event = threading.Event()

    def _get_headers(self) -> list:
        """Build authorization headers."""
        headers = []
        if self.bearer_token:
            headers.append(f"Authorization: Bearer {self.bearer_token}")
        return headers

    def _on_mapping_message(self, ws, message: str) -> None:
        """Handle incoming mapping configuration update."""
        logger.info("[SettingsSubscriber] Received MAPPING update.")
        try:
            data = json.loads(message)
            if self.on_mapping_update:
                self.on_mapping_update(data)
        except Exception as e:
            logger.error(f"[SettingsSubscriber] Failed to parse mapping message: {e}")

    def _on_counting_message(self, ws, message: str) -> None:
        """Handle incoming counting configuration update."""
        logger.info("[SettingsSubscriber] Received COUNTING update.")
        try:
            data = json.loads(message)
            if self.on_counting_update:
                self.on_counting_update(data)
        except Exception as e:
            logger.error(f"[SettingsSubscriber] Failed to parse counting message: {e}")

    def _on_error(self, ws, error) -> None:
        """Log WebSocket errors."""
        logger.error(f"[SettingsSubscriber] WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """Log WebSocket close events."""
        logger.info(f"[SettingsSubscriber] Closed: {close_status_code} - {close_msg}")

    def _run_mapping_ws(self) -> None:
        """Reconnection loop for the mapping settings WebSocket."""
        while not self._stop_event.is_set():
            if not self.mapping_url:
                break
            logger.info(f"[SettingsSubscriber] Connecting -> {self.mapping_url}")
            self.mapping_ws = websocket.WebSocketApp(
                self.mapping_url,
                header=self._get_headers(),
                on_message=self._on_mapping_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self.mapping_ws.run_forever(ping_interval=30, ping_timeout=10)

            if not self._stop_event.is_set():
                logger.info("[SettingsSubscriber] Reconnecting mapping WS in 3 seconds...")
                time.sleep(3)

    def _run_counting_ws(self) -> None:
        """Reconnection loop for the counting settings WebSocket."""
        while not self._stop_event.is_set():
            if not self.counting_url:
                break
            logger.info(f"[SettingsSubscriber] Connecting -> {self.counting_url}")
            self.counting_ws = websocket.WebSocketApp(
                self.counting_url,
                header=self._get_headers(),
                on_message=self._on_counting_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self.counting_ws.run_forever(ping_interval=30, ping_timeout=10)

            if not self._stop_event.is_set():
                logger.info("[SettingsSubscriber] Reconnecting counting WS in 3 seconds...")
                time.sleep(3)

    def start(self) -> None:
        """Start background threads for mapping and counting WebSocket listeners."""
        self._stop_event.clear()
        if self.mapping_url:
            threading.Thread(target=self._run_mapping_ws, daemon=True).start()
        if self.counting_url:
            threading.Thread(target=self._run_counting_ws, daemon=True).start()

    def stop(self) -> None:
        """Stop all WebSocket listener threads."""
        self._stop_event.set()
        # Force-close WebSocket connections to unblock run_forever() immediately
        if self.mapping_ws:
            self.mapping_ws.close()
        if self.counting_ws:
            self.counting_ws.close()
        logger.info("[SettingsSubscriber] Stopped all subscribers.")
