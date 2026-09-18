"""Shared same-column question corridors and conservative diagram masks."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple, Union

from .contracts import DiagramRef, ExamItem, ExamPackage, Page
from .roi import xyxy_to_yxyx


def _intersection(a, b) -> float:
    return max(0.0, min(a[2], b[2])-max(a[0], b[0])) * max(
        0.0, min(a[3], b[3])-max(a[1], b[1])
    )


class QuestionLayoutService:
    """Build one reading-order corridor per item on its physical column."""

    @staticmethod
    def corridors(package: ExamPackage, pages: Union[Sequence[Page], Dict[int, Page]]) -> Dict[str, List[int]]:
        page_map = pages if isinstance(pages, dict) else {page.index: page for page in pages}
        by_page: Dict[int, List[Tuple[ExamItem, List[int]]]] = defaultdict(list)
        for section in package.sections:
            for question in section.questions:
                for item in question.items:
                    if item.stem_region and len(item.stem_region.bbox) >= 4:
                        box = xyxy_to_yxyx(item.stem_region.bbox)
                        by_page[item.stem_region.page_index].append((item, box))

        result: Dict[str, List[int]] = {}
        for page_index, entries in by_page.items():
            page = page_map.get(page_index)
            if page is None:
                continue
            height, width = int(page.height or 2338), int(page.width or 1654)
            narrow = [(item, box) for item, box in entries
                      if box[3]-box[1] <= width*.62]
            left = [(item, box) for item, box in narrow
                    if (box[1]+box[3])/2 < width*.46]
            right = [(item, box) for item, box in narrow
                     if (box[1]+box[3])/2 > width*.54]
            two_column = len(left) >= 2 and len(right) >= 2
            groups: Dict[str, List[Tuple[ExamItem, List[int]]]] = defaultdict(list)
            for item, box in entries:
                center = (box[1]+box[3])/2
                column = ("left" if center < width*.5 else "right") if two_column else "full"
                groups[column].append((item, box))
            for column, values in groups.items():
                values.sort(key=lambda pair: (pair[1][0], pair[1][1]))
                if column == "left":
                    x1, x2 = 20, int(width*.53)
                elif column == "right":
                    x1, x2 = int(width*.47), width-20
                else:
                    x1, x2 = 45, width-45
                for index, (item, stem) in enumerate(values):
                    next_y = (values[index+1][1][0]-8
                              if index+1 < len(values) else height-20)
                    kind = str(item.item_type or "").casefold()
                    if not any(token in kind for token in (
                            "essay", "composition", "proof", "drawing", "作文", "证明", "作图")):
                        next_y = min(next_y, stem[0]+int(height*.34))
                    y1 = max(0, stem[0]-12)
                    y2 = min(height, max(stem[2]+35, next_y))
                    result[item.item_id] = [y1, x1, y2, x2]
        return result

    @staticmethod
    def _automatic_page_diagrams(image: Any, page: Page) -> List[List[int]]:
        """Detect line-dense figure regions without treating ordinary text as art."""
        import cv2
        import numpy as np

        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        height, width = ink.shape[:2]
        horizontal = cv2.morphologyEx(
            ink, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(35, int(width*.045)), 1)),
        )
        vertical = cv2.morphologyEx(
            ink, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, int(height*.025)))),
        )
        joined = cv2.dilate(
            cv2.bitwise_or(horizontal, vertical), np.ones((15, 15), np.uint8), iterations=2
        )
        contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < width*.075 or h < height*.02 or w*h > width*height*.28:
                continue
            crop_h = horizontal[y:y+h, x:x+w]
            crop_v = vertical[y:y+h, x:x+w]
            if cv2.countNonZero(crop_h) < 25 or cv2.countNonZero(crop_v) < 18:
                continue
            boxes.append([y, x, y+h, x+w])
        return boxes

    @classmethod
    def enrich_diagram_masks(cls, package: ExamPackage, pages: Sequence[Page]) -> Dict[str, int]:
        import cv2

        page_map = {page.index: page for page in pages}
        corridors = cls.corridors(package, page_map)
        automatic = {
            index: cls._automatic_page_diagrams(cv2.imread(page.path), page)
            for index, page in page_map.items()
        }
        added = 0
        for section in package.sections:
            for question in section.questions:
                for item in question.items:
                    if item.stem_region is None:
                        continue
                    kind = str(item.item_type or "").casefold()
                    # In grid/drawing questions, orthogonal lines are themselves
                    # legitimate response geometry, not an exclusion mask.
                    if any(token in kind for token in (
                            "grid", "tianzige", "drawing", "田字格", "作图")):
                        continue
                    page_index = item.stem_region.page_index
                    corridor = corridors.get(item.item_id)
                    if not corridor:
                        continue
                    existing = [xyxy_to_yxyx(diagram.bbox) for diagram in item.diagrams
                                if len(diagram.bbox) >= 4]
                    stem = xyxy_to_yxyx(item.stem_region.bbox)
                    for box in automatic.get(page_index, []):
                        center_y, center_x = (box[0]+box[2])/2, (box[1]+box[3])/2
                        if not (corridor[0] <= center_y <= corridor[2]
                                and corridor[1] <= center_x <= corridor[3]):
                            continue
                        if _intersection(box, stem) > .35 * max(1, (stem[2]-stem[0])*(stem[3]-stem[1])):
                            continue
                        if any(_intersection(box, old) > .65 * max(1, (box[2]-box[0])*(box[3]-box[1]))
                               for old in existing):
                            continue
                        # DiagramRef uses the package-wide xyxy convention.
                        item.diagrams.append(DiagramRef(
                            "auto_layout_diagram", "",
                            [box[1], box[0], box[3], box[2]],
                            {"detector": "orthogonal_line_density_v1", "status": "MASK_ONLY"},
                        ))
                        existing.append(box)
                        added += 1
        return {"automatic_diagram_masks": added}


__all__ = ["QuestionLayoutService"]
