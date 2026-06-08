"""Shared map feature data types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LaneType(str, Enum):
    """Supported lane feature classes."""

    LANE_LINE = "lane_line"


@dataclass(frozen=True)
class LaneBoundaryFeature:
    """Lane boundary polyline in world coordinates."""

    geometry: list[list[float]]
    feature_type: LaneType
    confidence: float
    point_count: int
    source: str = "geometric"

