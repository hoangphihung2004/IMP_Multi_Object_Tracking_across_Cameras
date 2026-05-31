"""
Pipeline Manager for DeepStream.

Builds and owns the local multi-source DeepStream pipeline lifecycle.
"""

from typing import Dict

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst
from loguru import logger

from core.config import PipelineConfig, SourcesConfig, YOLOXConfig
from services.pipeline.bus import BusManager
from services.pipeline.elements import ElementFactory
from services.pipeline.probe import AnalyticsProbe


class PipelineManager:
    """Orchestrator for the local DeepStream tracking pipeline."""

    def __init__(self) -> None:
        Gst.init(None)
        self.loop = GLib.MainLoop()
        self.pipeline = Gst.Pipeline.new("windows-local-tracking-pipeline")
        self.bus_manager = BusManager(self.pipeline, self.loop)
        self.elements: Dict[str, Gst.Element] = {}
        self.is_running = False

    def build_pipeline(
        self,
        sources_cfg: SourcesConfig,
        pipeline_cfg: PipelineConfig,
        yolox_cfg: YOLOXConfig,
    ) -> None:
        """Build a multi-file DeepStream pipeline that writes one MP4 per source."""
        logger.info(f"[Pipeline] Building pipeline for {len(sources_cfg.sources)} sources")
        logger.info(f"[Pipeline] YOLOX engine: {yolox_cfg.engine_path}")
        logger.info(f"[Pipeline] YOLOX conf={yolox_cfg.conf_thresh}, nms={yolox_cfg.nms_thresh}")

        try:
            self._create_input_elements(sources_cfg, pipeline_cfg)
            self._create_core_elements(pipeline_cfg, yolox_cfg)
            self._create_output_elements(sources_cfg, pipeline_cfg)
            self._add_all_elements()
            self._link_input_branches(sources_cfg)
            self._link_core_branch()
            self._link_output_branches(sources_cfg)
            logger.success("[Pipeline] Pipeline linked successfully for local MP4 output.")
        except Exception as exc:
            logger.error(f"[Pipeline] Failed to build pipeline: {exc}")
            raise

    def _create_input_elements(self, sources_cfg: SourcesConfig, pipeline_cfg: PipelineConfig) -> None:
        """Create filesrc/qtdemux/parser/decoder elements for every source."""
        for source in sources_cfg.sources:
            prefix = f"source_{source.id}"
            self.elements[f"{prefix}_filesrc"] = ElementFactory.create(
                "filesrc",
                f"file-source-{source.id}",
            )
            self.elements[f"{prefix}_filesrc"].set_property("location", source.uri)
            self.elements[f"{prefix}_demux"] = ElementFactory.create("qtdemux", f"demux-{source.id}")
            self.elements[f"{prefix}_parse"] = ElementFactory.create(
                pipeline_cfg.decode.parser,
                f"parser-{source.id}",
            )
            self.elements[f"{prefix}_decoder"] = ElementFactory.create(
                pipeline_cfg.decode.decoder,
                f"decoder-{source.id}",
            )

    def _create_core_elements(self, pipeline_cfg: PipelineConfig, yolox_cfg: YOLOXConfig) -> None:
        """Create nvstreammux, nvinfer, and nvstreamdemux elements."""
        self.elements["muxer"] = ElementFactory.create_and_configure(
            "nvstreammux",
            "stream-muxer",
            {
                "width": pipeline_cfg.muxer.width,
                "height": pipeline_cfg.muxer.height,
                "batch-size": pipeline_cfg.muxer.batch_size,
                "batched-push-timeout": pipeline_cfg.muxer.batched_push_timeout,
            },
        )
        self.elements["pgie"] = ElementFactory.create_and_configure(
            "nvinfer",
            "primary-inference",
            {"config-file-path": pipeline_cfg.infer.config_file},
        )
        self.elements["pgie"].set_property("model-engine-file", yolox_cfg.engine_path)
        self.elements["demux"] = ElementFactory.create("nvstreamdemux", "stream-demuxer")

    def _create_output_elements(self, sources_cfg: SourcesConfig, pipeline_cfg: PipelineConfig) -> None:
        """Create one output branch per source."""
        for source in sources_cfg.sources:
            prefix = f"out_{source.id}"
            self.elements[f"{prefix}_queue"] = ElementFactory.create("queue", f"output-queue-{source.id}")
            self.elements[f"{prefix}_conv"] = ElementFactory.create(
                "nvvideoconvert",
                f"output-conv-{source.id}",
            )
            self.elements[f"{prefix}_osd"] = ElementFactory.create("nvdsosd", f"output-osd-{source.id}")
            self.elements[f"{prefix}_encoder"] = ElementFactory.create_and_configure(
                pipeline_cfg.output.encoder,
                f"output-encoder-{source.id}",
                {"bitrate": pipeline_cfg.output.bitrate},
            )
            self.elements[f"{prefix}_parse"] = ElementFactory.create(
                pipeline_cfg.output.parser,
                f"output-parser-{source.id}",
            )
            self.elements[f"{prefix}_mux"] = ElementFactory.create(
                pipeline_cfg.output.muxer,
                f"output-mux-{source.id}",
            )
            self.elements[f"{prefix}_sink"] = ElementFactory.create_and_configure(
                "filesink",
                f"file-sink-{source.id}",
                {"location": source.output, "sync": False, "async": False},
            )

    def _add_all_elements(self) -> None:
        """Add every created element to the pipeline."""
        for element in self.elements.values():
            self.pipeline.add(element)

    def _link_input_branches(self, sources_cfg: SourcesConfig) -> None:
        """Link every input branch into nvstreammux."""
        muxer = self.elements["muxer"]
        for source in sources_cfg.sources:
            prefix = f"source_{source.id}"
            filesrc = self.elements[f"{prefix}_filesrc"]
            demux = self.elements[f"{prefix}_demux"]
            parser = self.elements[f"{prefix}_parse"]
            decoder = self.elements[f"{prefix}_decoder"]

            self._link(filesrc, demux)
            demux.connect("pad-added", self._on_demux_pad_added, parser)
            self._link(parser, decoder)

            sink_pad = muxer.get_request_pad(f"sink_{source.id}")
            src_pad = decoder.get_static_pad("src")
            if not sink_pad or not src_pad:
                raise RuntimeError(f"Unable to get muxer/decoder pad for source {source.id}")
            if src_pad.link(sink_pad) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link decoder source {source.id} to muxer")

    def _link_core_branch(self) -> None:
        """Link muxer, nvinfer, and demux."""
        self._link(self.elements["muxer"], self.elements["pgie"])
        self._link(self.elements["pgie"], self.elements["demux"])

    def _link_output_branches(self, sources_cfg: SourcesConfig) -> None:
        """Link nvstreamdemux outputs to per-source MP4 writers."""
        demux = self.elements["demux"]
        for source in sources_cfg.sources:
            prefix = f"out_{source.id}"
            demux_src_pad = demux.get_request_pad(f"src_{source.id}")
            queue_sink_pad = self.elements[f"{prefix}_queue"].get_static_pad("sink")
            if not demux_src_pad or not queue_sink_pad:
                raise RuntimeError(f"Unable to get demux/queue pad for source {source.id}")
            if demux_src_pad.link(queue_sink_pad) != Gst.PadLinkReturn.OK:
                raise RuntimeError(f"Failed to link demux source {source.id} to output queue")

            self._link(self.elements[f"{prefix}_queue"], self.elements[f"{prefix}_conv"])
            self._link(self.elements[f"{prefix}_conv"], self.elements[f"{prefix}_osd"])
            self._link(self.elements[f"{prefix}_osd"], self.elements[f"{prefix}_encoder"])
            self._link(self.elements[f"{prefix}_encoder"], self.elements[f"{prefix}_parse"])
            self._link(self.elements[f"{prefix}_parse"], self.elements[f"{prefix}_mux"])
            self._link(self.elements[f"{prefix}_mux"], self.elements[f"{prefix}_sink"])

    def _link(self, source: Gst.Element, target: Gst.Element) -> None:
        """Link two elements and raise on failure."""
        if not source.link(target):
            raise RuntimeError(f"Failed to link {source.get_name()} to {target.get_name()}")

    def attach_probe(self, probe_handler: AnalyticsProbe) -> None:
        """Attach AnalyticsProbe to the nvinfer source pad."""
        pgie_src_pad = self.elements["pgie"].get_static_pad("src")
        if not pgie_src_pad:
            raise RuntimeError("Unable to get nvinfer src pad")
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, probe_handler.probe_callback, None)
        logger.info("[Pipeline] AnalyticsProbe attached to nvinfer src pad.")

    def run(self) -> None:
        """Start the pipeline execution loop."""
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

    def stop(self) -> None:
        """Gracefully stop the pipeline."""
        if not self.is_running:
            return

        logger.info("[Pipeline] Stopping pipeline...")
        self.pipeline.set_state(Gst.State.NULL)
        self.bus_manager.teardown()
        self.is_running = False
        logger.success("[Pipeline] Pipeline stopped.")

    def _on_demux_pad_added(self, src, pad, parser: Gst.Element) -> None:
        """Dynamic link handler for qtdemux to parser."""
        sink_pad = parser.get_static_pad("sink")
        if sink_pad.is_linked():
            return
        result = pad.link(sink_pad)
        if result != Gst.PadLinkReturn.OK:
            logger.error(f"[Pipeline] Failed to link demux pad to {parser.get_name()}: {result}")
