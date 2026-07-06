from typing import List


def xywh_to_xyxy(box: List[float]) -> List[float]:
    """Convert COCO [x, y, w, h] box to [xmin, ymin, xmax, ymax]."""
    x, y, w, h = box
    return [x, y, x + w, y + h]


def normalize_xyxy(box: List[float], width: int, height: int) -> List[float]:
    """Normalize [xmin, ymin, xmax, ymax] by original image width and height."""
    x1, y1, x2, y2 = box
    return [
        x1 / width,
        y1 / height,
        x2 / width,
        y2 / height,
    ]


def count_to_class(num_targets: int) -> int:
    """Map target count to cardinality class: 0, 1, 2, or 3+."""
    if num_targets == 0:
        return 0
    if num_targets == 1:
        return 1
    if num_targets == 2:
        return 2
    return 3
