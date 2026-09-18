"""Structuring façade and deterministic reading order."""
from typing import List, Sequence
from .contracts import OCRBlock


def order_blocks_column_aware(blocks: Sequence[OCRBlock], page_width: float) -> List[OCRBlock]:
    blocks = list(blocks)
    if len(blocks) < 2:
        return blocks
    midpoint = float(page_width) / 2.0
    left = [b for b in blocks if len(b.bbox) < 4 or (b.bbox[0] + b.bbox[2]) / 2 < midpoint]
    right = [b for b in blocks if b not in left]
    key = lambda b: (b.bbox[1] if len(b.bbox) > 1 else 0, b.bbox[0] if b.bbox else 0)
    # Use column order only when both sides contain a meaningful stream.
    if left and right:
        return sorted(left, key=key) + sorted(right, key=key)
    return sorted(blocks, key=key)


class StructuringService:
    @staticmethod
    def reconcile(package):
        from homework_extractor import reconcile_exam_package
        return reconcile_exam_package(package)
