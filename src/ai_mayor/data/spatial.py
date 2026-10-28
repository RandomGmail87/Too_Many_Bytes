"""Spatial mapping utilities for Chicago Community Areas.

This module provides spatial alignment to map geographic coordinates (latitude,
longitude) to Chicago's 77 official Community Areas without requiring heavy
GIS dependencies (such as GDAL or geopandas).

It uses pure-Python ray-casting point-in-polygon test against the official City of
Chicago boundary polygons (dataset igwz-8jzy) or pre-indexed community bounding boxes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Official Chicago 77 Community Area Names (City of Chicago Data Portal)
COMMUNITY_AREA_NAMES: Dict[int, str] = {
    1: "Rogers Park",
    2: "West Ridge",
    3: "Uptown",
    4: "Lincoln Square",
    5: "North Center",
    6: "Lake View",
    7: "Lincoln Park",
    8: "Near North Side",
    9: "Edison Park",
    10: "Norwood Park",
    11: "Jefferson Park",
    12: "Forest Glen",
    13: "North Park",
    14: "Albany Park",
    15: "Portage Park",
    16: "Irving Park",
    17: "Dunning",
    18: "Montclare",
    19: "Belmont Cragin",
    20: "Hermosa",
    21: "Avondale",
    22: "Logan Square",
    23: "Humboldt Park",
    24: "West Town",
    25: "Austin",
    26: "West Garfield Park",
    27: "East Garfield Park",
    28: "Near West Side",
    29: "North Lawndale",
    30: "South Lawndale",
    31: "Lower West Side",
    32: "Loop",
    33: "Near South Side",
    34: "Armour Square",
    35: "Douglas",
    36: "Oakland",
    37: "Fuller Park",
    38: "Grand Boulevard",
    39: "Kenwood",
    40: "Washington Park",
    41: "Hyde Park",
    42: "Woodlawn",
    43: "South Shore",
    44: "Chatham",
    45: "Avalon Park",
    46: "South Chicago",
    47: "Burnside",
    48: "Calumet Heights",
    49: "Roseland",
    50: "Pullman",
    51: "South Deering",
    52: "East Side",
    53: "West Pullman",
    54: "Riverdale",
    55: "Hegewisch",
    56: "Garfield Ridge",
    57: "Archer Heights",
    58: "Brighton Park",
    59: "McKinley Park",
    60: "Bridgeport",
    61: "New City",
    62: "West Elsdon",
    63: "Gage Park",
    64: "Clearing",
    65: "West Lawn",
    66: "Chicago Lawn",
    67: "West Englewood",
    68: "Englewood",
    69: "Greater Grand Crossing",
    70: "Ashburn",
    71: "Auburn Gresham",
    72: "Beverly",
    73: "Washington Heights",
    74: "Mount Greenwood",
    75: "Morgan Park",
    76: "O'Hare",
    77: "Edgewater",
}


def point_in_polygon(x: float, y: float, polygon: List[List[float]]) -> bool:
    """Ray-casting algorithm to test whether point (x, y) is inside a 2D polygon.

    Args:
        x: Longitude of test point.
        y: Latitude of test point.
        polygon: List of [lon, lat] vertex pairs forming the polygon perimeter.

    Returns:
        bool: True if point is strictly inside or on edge of polygon.
    """
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0][0], polygon[0][1]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n][0], polygon[i % n][1]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


class SpatialCommunityMapper:
    """Maps spatial coordinates (latitude, longitude) to Chicago Community Area IDs."""

    def __init__(self, geojson_path: Optional[Path | str] = None) -> None:
        """Initialize spatial mapper.

        Args:
            geojson_path: Optional path to Boundaries - Community Areas GeoJSON.
        """
        self.polygons: Dict[int, List[List[List[float]]]] = {}
        self.bboxes: Dict[int, Tuple[float, float, float, float]] = {}

        if geojson_path is not None:
            self.load_geojson(geojson_path)

    def load_geojson(self, path: Path | str) -> None:
        """Load and index polygons from GeoJSON file.

        Args:
            path: Path to Boundaries - Community Areas GeoJSON.
        """
        target = Path(path)
        if not target.exists():
            return

        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)

        for feature in data.get("features", []):
            props = feature.get("properties", {})
            ca_num = props.get("area_num_1") or props.get("area_numbe")
            if ca_num is None:
                continue
            ca_id = int(ca_num)

            geom = feature.get("geometry", {})
            g_type = geom.get("type")
            coords = geom.get("coordinates", [])

            rings: List[List[List[float]]] = []
            if g_type == "Polygon":
                rings.append(coords[0])
            elif g_type == "MultiPolygon":
                for poly in coords:
                    rings.append(poly[0])

            self.polygons[ca_id] = rings

            # Compute bounding box (min_x, min_y, max_x, max_y)
            all_x = [pt[0] for ring in rings for pt in ring]
            all_y = [pt[1] for ring in rings for pt in ring]
            if all_x and all_y:
                self.bboxes[ca_id] = (min(all_x), min(all_y), max(all_x), max(all_y))

    def find_community_area(self, lat: Optional[float], lon: Optional[float]) -> Optional[int]:
        """Find the matching Community Area ID for given latitude and longitude.

        Args:
            lat: Latitude (Y coordinate).
            lon: Longitude (X coordinate).

        Returns:
            Optional[int]: Community area ID (1 to 77) or None if unmapped.
        """
        if lat is None or lon is None:
            return None

        # Quick Chicago metropolitan bounding box check
        # Latitude roughly 41.6 to 42.1; Longitude roughly -87.95 to -87.50
        if not (41.60 <= lat <= 42.10 and -88.00 <= lon <= -87.40):
            return None

        # Test against loaded polygons
        for ca_id, rings in self.polygons.items():
            # Fast bounding box rejection
            bbox = self.bboxes.get(ca_id)
            if bbox:
                min_x, min_y, max_x, max_y = bbox
                if not (min_x <= lon <= max_x and min_y <= lat <= max_y):
                    continue

            for ring in rings:
                if point_in_polygon(lon, lat, ring):
                    return ca_id

        return None

