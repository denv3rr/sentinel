from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ZoneHit:
    include_zones: list[str]
    ignore_zones: list[str]


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y)
        if intersects:
            denominator = (yj - yi) if (yj - yi) != 0 else 1e-9
            x_on_edge = (xj - xi) * (y - yi) / denominator + xi
            if x < x_on_edge:
                inside = not inside
        j = i
    return inside


def detect_zone_hits(
    bbox: tuple[int, int, int, int],
    zones: list[dict[str, object]],
) -> ZoneHit:
    center = bbox_center(bbox)
    include_hits: list[str] = []
    ignore_hits: list[str] = []

    for zone in zones:
        points = zone.get("points", [])
        if not isinstance(points, list):
            continue
        polygon: list[tuple[float, float]] = []
        for pt in points:
            if isinstance(pt, (list, tuple)) and len(pt) == 2:
                polygon.append((float(pt[0]), float(pt[1])))
        if not polygon:
            continue

        if point_in_polygon(center, polygon):
            zone_id = str(zone.get("id", "zone"))
            mode = str(zone.get("mode", "include"))
            if mode == "ignore":
                ignore_hits.append(zone_id)
            elif mode == "include":
                include_hits.append(zone_id)

    return ZoneHit(include_zones=include_hits, ignore_zones=ignore_hits)