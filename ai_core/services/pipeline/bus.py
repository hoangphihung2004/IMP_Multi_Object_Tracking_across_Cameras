"""
GStreamer Bus Manager.
Handles message loop from the GStreamer pipeline bus (EOS, Error, etc.).
"""

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
from loguru import logger


class BusManager:
    """
    Manager for the GStreamer pipeline bus.
    Listens for messages and handles them accordingly.
    """

    def __init__(self, pipeline: Gst.Pipeline, loop: GLib.MainLoop):
        self.pipeline = pipeline
        self.loop = loop
        self.bus = pipeline.get_bus()

    def setup(self) -> None:
        """
        Add watch to the bus to listen for messages.
        """
        self.bus.add_signal_watch()
        self.bus.connect("message", self._on_message)
        logger.debug("[Bus] Watch added to pipeline bus.")

    def _on_message(self, bus: Gst.Bus, message: Gst.Message) -> None:
        """
        Internal message handler.
        """
        msg_type = message.type

        if msg_type == Gst.MessageType.EOS:
            logger.info("[Bus] End-of-Stream (EOS) reached.")
            self.loop.quit()

        elif msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"[Bus] Pipeline Error: {err.message}")
            logger.debug(f"[Bus] Debug info: {debug}")
            self.loop.quit()

        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                logger.trace(
                    f"[Bus] Pipeline state changed from {Gst.Element.state_get_name(old_state)} "
                    f"to {Gst.Element.state_get_name(new_state)}."
                )

        elif msg_type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            logger.warning(f"[Bus] Pipeline Warning: {warn.message}")
            if debug:
                logger.debug(f"[Bus] Debug info: {debug}")

    def teardown(self) -> None:
        """
        Remove signal watch.
        """
        self.bus.remove_signal_watch()
        logger.debug("[Bus] Watch removed.")
