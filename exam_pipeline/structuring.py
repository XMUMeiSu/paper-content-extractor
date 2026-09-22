"""Structuring façade and deterministic reading order."""
from typing import List, Sequence
from .contracts import OCRBlock


def order_blocks_column_aware(blocks: Sequence[OCRBlock], page_width: float) -> List[OCRBlock]:
    from .reading_order import column_groups
    return [block for group in column_groups(list(blocks), page_width) for block in group]



class StructuringService:
    @staticmethod
    def reconcile(package):
        from homework_extractor import reconcile_exam_package
        return reconcile_exam_package(package)
