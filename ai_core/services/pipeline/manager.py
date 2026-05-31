"""
Pipeline Manager for DeepStream.
Orchestrates the assembly, execution, and lifecycle of the GStreamer pipeline.
"""

import os
import sys
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
from loguru import logger

from services.pipeline.elements import ElementFactory
from services.pipeline.bus import BusManager
from services.pipeline.probe import AnalyticsProbe


class PipelineManager:
    """
    Orchestrator for the DeepStream Pipeline.
    """

    def __init__(self):
        Gst.init(None)
        self.loop = GLib.MainLoop()
        self.pipeline = Gst.Pipeline.new("jetson-analytics-pipeline")
        self.bus_manager = BusManager(self.pipeline, self.loop)
        
        self.elements = {}
        self.is_running = False

    def build_pipeline(self, source_uri: str, config_infer: str, yolox_cfg=None):
        """
        Build the DeepStream pipeline graph.
        
        Args:
            source_uri: Path to video file or RTSP URL.
            config_infer: Path to nvinfer config file.
            yolox_cfg: YOLOXConfig Pydantic model for overriding inference params.

        Structure: 
        filesrc/v4l2src -> h264parse -> nvv4l2decoder -> nvstreammux -> 
        nvinfer (YOLOX) -> nvvideoconvert -> nvdsosd -> tee -> [sink | janus]
        """
        logger.info(f"[Pipeline] Building pipeline for source: {source_uri}")
        if yolox_cfg:
            logger.info(f"[Pipeline] YOLOX engine: {yolox_cfg.engine_path}")
            logger.info(f"[Pipeline] YOLOX conf={yolox_cfg.conf_thresh}, nms={yolox_cfg.nms_thresh}")

        # 1. Create Elements
        try:
            # Source handling
            if source_uri.startswith("rtsp://") or source_uri.startswith("http://") or source_uri.endswith(".mp4"):
                self.elements["source"] = ElementFactory.create("filesrc", "file-source")
                self.elements["source"].set_property("location", source_uri)
                self.elements["qtdemux"] = ElementFactory.create("qtdemux", "demux")
                self.elements["h264parse"] = ElementFactory.create("h264parse", "h264-parser")
                self.elements["decoder"] = ElementFactory.create("nvv4l2decoder", "decoder")
            else:
                # USB Camera (Logitech C270 / MJPG webcam)
                # Pipeline: v4l2src → caps(MJPG 1080p@30) → queue → jpegparse → nvv4l2decoder(mjpeg=1) → nvstreammux
                self.elements["source"] = ElementFactory.create("v4l2src", "usb-source")
                self.elements["source"].set_property("device", source_uri)
                self.elements["source"].set_property("io-mode", 2)
                self.elements["source"].set_property("do-timestamp", True)
                
                self.elements["capsfilter"] = ElementFactory.create("capsfilter", "mjpg-caps")
                self.elements["capsfilter"].set_property(
                    "caps", Gst.Caps.from_string(
                        "image/jpeg,width=1920,height=1080,framerate=30/1"
                    )
                )
                
                self.elements["cam_queue"] = ElementFactory.create_and_configure(
                    "queue", "camera-queue",
                    {"max-size-buffers": 1, "leaky": 2}
                )
                self.elements["jpegparse"] = ElementFactory.create("jpegparse", "jpeg-parser")
                self.elements["decoder"] = ElementFactory.create_and_configure(
                    "nvv4l2decoder", "jpeg-decoder",
                    {"mjpeg": 1}
                )
                self.elements["hwconvert"] = ElementFactory.create("nvvideoconvert", "hw-convert")

            # DeepStream Core
            self.elements["muxer"] = ElementFactory.create_and_configure(
                "nvstreammux", "stream-muxer",
                {"width": 1920, "height": 1080, "batch-size": 1, "batched-push-timeout": 40000}
            )
            
            # nvinfer: base config from file, then override with YOLOXConfig
            pgie_props = {"config-file-path": config_infer}
            self.elements["pgie"] = ElementFactory.create_and_configure(
                "nvinfer", "primary-inference", pgie_props
            )
            
            # Override nvinfer with YOLOXConfig values (engine path, thresholds)
            if yolox_cfg:
                pgie = self.elements["pgie"]
                pgie.set_property("model-engine-file", yolox_cfg.engine_path)
                logger.info(f"[Pipeline] nvinfer model-engine-file → {yolox_cfg.engine_path}")
            
            # Visualization & Sink
            self.elements["nvvidconv"] = ElementFactory.create("nvvideoconvert", "nvvideo-converter")
            self.elements["nvosd"] = ElementFactory.create("nvdsosd", "on-screen-display")
            self.elements["tee"] = ElementFactory.create("tee", "stream-tee")
            
            # --- Branch 1: Local Display/Fakesink ---
            self.elements["queue_sink"] = ElementFactory.create("queue", "queue-sink")
            self.elements["sink"] = ElementFactory.create_and_configure(
                "fakesink", "sink", {"sync": True}
            )

            # --- Branch 2: Janus Streaming (720p H.264) ---
            self.elements["janus_queue"] = ElementFactory.create("queue", "janus-queue")
            self.elements["nvvidconv_scale"] = ElementFactory.create("nvvideoconvert", "janus-scaler")
            self.elements["caps_scale"] = ElementFactory.create_and_configure(
                "capsfilter", "janus-caps",
                {"caps": Gst.Caps.from_string("video/x-raw(memory:NVMM), width=1280, height=720")}
            )
            self.elements["encoder"] = ElementFactory.create_and_configure(
                "nvv4l2h264enc", "janus-h264-enc",
                {
                    "bitrate": int(os.getenv("VIDEO_BITRATE", "1200000")),
                    "preset-level": 1,
                    "insert-sps-pps": True,
                }
            )
            self.elements["h264parse_janus"] = ElementFactory.create("h264parse", "janus-h264-parser")
            self.elements["rtppay"] = ElementFactory.create("rtph264pay", "janus-rtp-pay")
            self.elements["udpsink"] = ElementFactory.create_and_configure(
                "udpsink", "janus-udp-sink",
                {
                    "host": os.getenv("JANUS_SERVER_IP", "127.0.0.1"),
                    "port": int(os.getenv("JANUS_UDP_PORT", "8004")),
                    "async": False,
                    "sync": True
                }
            )

            # 2. Add elements to pipeline
            for el in self.elements.values():
                self.pipeline.add(el)

            # 3. Link elements (Sequential)
            # Source -> Muxer
            if "qtdemux" in self.elements:
                self.elements["source"].link(self.elements["qtdemux"])
                self.elements["qtdemux"].connect("pad-added", self._on_pad_added)
                self.elements["h264parse"].link(self.elements["decoder"])
            else:
                # USB Camera (MJPG): source → caps → queue → jpegparse → decoder → hwconvert → muxer
                self._link(self.elements["source"], self.elements["capsfilter"])
                self._link(self.elements["capsfilter"], self.elements["cam_queue"])
                self._link(self.elements["cam_queue"], self.elements["jpegparse"])
                self._link(self.elements["jpegparse"], self.elements["decoder"])
                self._link(self.elements["decoder"], self.elements["hwconvert"])

            # Common StreamMux Link
            sink_pad = self.elements["muxer"].get_request_pad("sink_0")
            src_pad = self.elements["hwconvert" if "hwconvert" in self.elements else "decoder"].get_static_pad("src")
            src_pad.link(sink_pad)

            # Core: muxer -> pgie -> nvvidconv -> nvosd -> tee
            self._link(self.elements["muxer"], self.elements["pgie"])
            self._link(self.elements["pgie"], self.elements["nvvidconv"])
            self._link(self.elements["nvvidconv"], self.elements["nvosd"])
            self._link(self.elements["nvosd"], self.elements["tee"])

            # --- Link Branch 1: Fakesink ---
            tee_pad_sink = self.elements["tee"].get_request_pad("src_%u")
            queue_sink_pad = self.elements["queue_sink"].get_static_pad("sink")
            tee_pad_sink.link(queue_sink_pad)
            self._link(self.elements["queue_sink"], self.elements["sink"])

            # --- Link Janus Streaming Branch (720p) ---
            tee_pad_janus = self.elements["tee"].get_request_pad("src_%u")
            janus_queue_pad = self.elements["janus_queue"].get_static_pad("sink")
            if tee_pad_janus.link(janus_queue_pad) != Gst.PadLinkReturn.OK:
                logger.error("[Pipeline] Failed to link tee to janus queue")
                sys.exit(1)
            
            logger.debug("[Pipeline] Forked stream for Janus branch.")
            self._link(self.elements["janus_queue"], self.elements["nvvidconv_scale"])
            self._link(self.elements["nvvidconv_scale"], self.elements["caps_scale"])
            self._link(self.elements["caps_scale"], self.elements["encoder"])
            self._link(self.elements["encoder"], self.elements["h264parse_janus"])
            self._link(self.elements["h264parse_janus"], self.elements["rtppay"])
            self._link(self.elements["rtppay"], self.elements["udpsink"])

            logger.success("[Pipeline] Pipeline linked successfully with Janus branch.")

        except Exception as e:
            logger.error(f"[Pipeline] Failed to build pipeline: {e}")
            raise

    def _link(self, el1: Gst.Element, el2: Gst.Element):
        """Helper to link two elements with error checking."""
        if not el1.link(el2):
            logger.error(f"[Pipeline] Failed to link {el1.get_name()} to {el2.get_name()}")
            sys.exit(1)

    def attach_probe(self, probe_handler: AnalyticsProbe):
        """
        Attach the AnalyticsProbe to the nvinfer source pad.
        """
        pgie_src_pad = self.elements["pgie"].get_static_pad("src")
        if not pgie_src_pad:
            logger.error("[Pipeline] Unable to get nvinfer src pad")
            return
        
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, probe_handler.probe_callback, None)
        logger.info("[Pipeline] AnalyticsProbe attached to nvinfer src pad.")

    def run(self):
        """
        Start the pipeline execution.
        """
        self.bus_manager.setup()
        
        logger.info("[Pipeline] Starting pipeline...")
        self.pipeline.set_state(Gst.State.PLAYING)
        self.is_running = True
        
        try:
            self.loop.run()
        except KeyboardInterrupt:
            logger.warning("[Pipeline] Interrupted by user.")
        finally:
            self.stop()

    def stop(self):
        """
        Gracefully stop the pipeline.
        """
        if not self.is_running:
            return
            
        logger.info("[Pipeline] Stopping pipeline...")
        self.pipeline.set_state(Gst.State.NULL)
        self.bus_manager.teardown()
        self.is_running = False
        logger.success("[Pipeline] Pipeline stopped.")

    def _on_pad_added(self, src, pad):
        """Dynamic link handler for demuxer."""
        sink_pad = self.elements["h264parse"].get_static_pad("sink")
        pad.link(sink_pad)
