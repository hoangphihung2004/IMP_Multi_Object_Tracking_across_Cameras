from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional, Literal, Union
from datetime import datetime, timezone

# Define Alert Levels
AlertType = Literal['none', 'info', 'warning', 'critical']

class ObjectDetection(BaseModel):
    """
    Detailed information for each detected object within a video frame.
    """
    model_config = ConfigDict(populate_by_name=True)

    track_id: int = Field(..., description="Unique tracking ID")

    # Pixel-based bounding box
    bbox: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x1, y1, x2, y2] in pixels"
    )

    # 2D map projected coordinates
    coordinates_2D: Optional[List[float]] = Field(
        None,
        min_length=2,
        max_length=2,
        description="[x, y] center projected onto 2D map"
    )

class FrameData(BaseModel):
    """
    Standardized JSON structure sent from Edge Device for persistence.
    """
    # UTC timestamp of the inference process
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="UTC inference timestamp")

    # Line crossing statistics (Realtime count)
    count_in: int = Field(0, description="Cumulative count of entries")
    count_out: int = Field(0, description="Cumulative count of exits")
    occupancy: int = Field(0, description="Current occupancy (count_in - count_out, clamped to >=0 if provided that way)")

    # Alert level
    alert: AlertType = Field('none', description="Alert level (none, info, warning, critical)")

    fps: Optional[float] = Field(None, description="Processing FPS for the current frame")

    # Flag to signal the Backend to persist data in PostgreSQL for Heatmap/Analytics
    save_to_db: bool = Field(True, description="true = save to DB, false = do nothing")

    # Original frame dimensions
    height_frame: Optional[int] = Field(None, description="Height of the original frame in pixels")
    width_frame: Optional[int] = Field(None, description="Width of the original frame in pixels")

    # 2D map dimensions
    height_2D: Optional[int] = Field(None, description="Height of the 2D map in pixels")
    width_2D: Optional[int] = Field(None, description="Width of the 2D map in pixels")

    # List of detected objects
    objects: List[ObjectDetection] = Field(default_factory=list)

# Schemas for Calibration/Mapping API

class Size(BaseModel):
    width: Union[int, float]
    height: Union[int, float]

class Correspondence(BaseModel):
    label: str
    camera: List[Union[int, float]]
    map: List[Union[int, float]]

    @field_validator('camera', 'map')
    def validate_coords(cls, v):
        if len(v) != 2:
            raise ValueError("Coordinates must have exactly 2 elements [x, y]")
        return v

class MappingRequest(BaseModel):
    """
    Request body for the calibration endpoint.
    """
    image_size: Size
    map_size: Size
    correspondences: List[Correspondence] = Field(..., min_length=4, description="At least 4 correspondences are required for homography.")

class CountingRequest(BaseModel):
    """
    Request body for the counting line settings.
    """
    line_start: List[Union[int, float]] = Field(..., min_length=2, max_length=2)
    line_end: List[Union[int, float]] = Field(..., min_length=2, max_length=2)
    inside_point: List[Union[int, float]] = Field(..., min_length=2, max_length=2)
    crossing_margin: Union[int, float]
    frame_height: int
    frame_width: int

class CalibrationSuccessResponse(BaseModel):
    """
    Standard success response for the calibration endpoint.
    """
    status: Literal["ok"] = "ok"
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ErrorDetail(BaseModel):
    """
    Detailed error information for API responses.
    """
    code: str
    message: str

class ErrorResponse(BaseModel):
    """
    Standard error response schema.
    """
    error: ErrorDetail