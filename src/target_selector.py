"""Single-target selection helpers."""

from typing import Iterable, Optional

from src.geometry import bbox_area, bbox_center, point_in_bbox


class SingleTargetSelector:
    """Pick one target using center distance first and area second."""

    @staticmethod
    def pick_primary(detections: Iterable, frame_shape) -> Optional[object]:
        detections = list(detections)
        if not detections:
            return None
        frame_h, frame_w = frame_shape[:2]

        def sort_key(det):
            cx, cy = bbox_center(det.bbox)
            dist = ((cx - frame_w / 2) / max(frame_w, 1)) ** 2 + ((cy - frame_h / 2) / max(frame_h, 1)) ** 2
            return (dist, -bbox_area(det.bbox))

        return min(detections, key=sort_key)

    @staticmethod
    def pick_face(detections: Iterable, frame_shape, parent_bbox=None):
        detections = list(detections)
        if not detections:
            return None
        if parent_bbox is not None:
            inside = []
            for det in detections:
                cx, cy = bbox_center(det.bbox)
                if point_in_bbox(cx, cy, parent_bbox):
                    inside.append(det)
            if inside:
                detections = inside
        return SingleTargetSelector.pick_primary(detections, frame_shape)
