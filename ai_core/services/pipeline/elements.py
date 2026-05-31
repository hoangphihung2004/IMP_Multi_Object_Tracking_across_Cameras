"""
GStreamer Element Factory.
Centralized utility to create and configure GStreamer/DeepStream elements
with strict error handling and property validation.
"""

from typing import Any, Dict, Optional
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst
from loguru import logger


class ElementFactory:
    """
    Factory class to create GStreamer elements and set their properties.
    Ensures that elements exist and are correctly initialized.
    """

    @staticmethod
    def create(factory_name: str, element_name: str) -> Gst.Element:
        """
        Create a GStreamer element by its factory name.

        Args:
            factory_name: Name of the GStreamer plugin (e.g., 'nvinfer', 'nvstreammux').
            element_name: Unique name for this specific instance in the pipeline.

        Returns:
            Gst.Element: The created element.

        Raises:
            RuntimeError: If the element could not be created (e.g., plugin missing).
        """
        element = Gst.ElementFactory.make(factory_name, element_name)
        if not element:
            logger.error(f"[Factory] Failed to create element '{factory_name}' named '{element_name}'.")
            logger.error(f"[Factory] Check if the DeepStream/GStreamer plugin '{factory_name}' is installed.")
            raise RuntimeError(f"Could not create GStreamer element: {factory_name}")
        
        logger.debug(f"[Factory] Created element '{factory_name}' -> '{element_name}'")
        return element

    @staticmethod
    def set_properties(element: Gst.Element, properties: Dict[str, Any]) -> None:
        """
        Batch set properties on a GStreamer element.

        Args:
            element: The Gst.Element to configure.
            properties: Dictionary of {property_name: value}.
        """
        for key, value in properties.items():
            try:
                element.set_property(key, value)
                logger.trace(f"[Factory] {element.get_name()}: Set '{key}' = {value}")
            except Exception as e:
                logger.error(f"[Factory] Failed to set property '{key}' on '{element.get_name()}': {e}")
                raise

    @classmethod
    def create_and_configure(
        cls, 
        factory_name: str, 
        element_name: str, 
        properties: Optional[Dict[str, Any]] = None
    ) -> Gst.Element:
        """
        Utility to create an element and immediately set its properties.
        """
        element = cls.create(factory_name, element_name)
        if properties:
            cls.set_properties(element, properties)
        return element
