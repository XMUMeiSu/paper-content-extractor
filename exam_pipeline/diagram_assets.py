"""Deterministic diagram crops and relative links for extraction results.

Diagram coordinates are evidence discovered from the current page.  This
module never asks a model to invent a box: explicit boxes are retained and
line-dense candidates are assigned using the question reading corridor.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2

from .contracts import DiagramRef, ExamItem, ExamPackage, Page
from .layout import QuestionLayoutService


_GRAPHIC_TYPES = {
    "choice", "solve", "large_writing", "drawing", "proof", "calculation",
    "图形选择", "选择", "大题", "解答", "作图", "证明",
}


def _safe(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "item")).strip("._") or "item"


def _xyxy(box: Any) -> List[int]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return []
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    except (TypeError, ValueError):
        return []
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else []


def _iou(a: Sequence[int], b: Sequence[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if not inter:
        return 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(area_a + area_b - inter)


def _contains_center(box: Sequence[int], corridor: Sequence[int]) -> bool:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return corridor[0] <= cy <= corridor[2] and corridor[1] <= cx <= corridor[3]


def _printed_overlap(page: Page, box: Sequence[int]) -> bool:
    """Reject OCR-covered section banners and text blocks mistaken for art."""
    area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    for block in page.ocr or []:
        text = str(getattr(block, "text", "") or "").strip()
        b = _xyxy(getattr(block, "bbox", []))
        if not text or not b:
            continue
        inter = max(0, min(box[2], b[2]) - max(box[0], b[0])) * max(
            0, min(box[3], b[3]) - max(box[1], b[1]))
        # A figure can contain small axis labels, but a large OCR-covered
        # region is a printed banner/question line rather than a diagram.
        if inter / float(area) >= 0.38 and len(re.sub(r"\s+", "", text)) >= 3:
            return True
    return False


class DiagramAssetService:
    """Export stable, relative image links for choice and large-question art."""

    def __init__(self, padding: int = 10, jpeg_quality: int = 95):
        self.padding = max(0, int(padding))
        self.jpeg_quality = max(70, min(100, int(jpeg_quality)))

    @staticmethod
    def _target_item(item: ExamItem) -> bool:
        kind = str(item.item_type or "").casefold()
        return kind in _GRAPHIC_TYPES or any(token in kind for token in (
            "choice", "solve", "writing", "drawing", "proof", "calculation",
            "选择", "解答", "作图", "证明", "大题"))

    @staticmethod
    def _page_candidates(page: Page) -> List[List[int]]:
        image = cv2.imread(str(page.path))
        # The layout detector returns yxyx; the public diagram contract is xyxy.
        return [[box[1], box[0], box[3], box[2]]
                for box in QuestionLayoutService._automatic_page_diagrams(image, page)]

    @staticmethod
    def _item_corridors(package: ExamPackage, pages: Sequence[Page]) -> Dict[str, List[int]]:
        return QuestionLayoutService.corridors(package, {page.index: page for page in pages})

    def _collect_boxes(self, item: ExamItem, page: Page,
                       corridor: Sequence[int] | None) -> List[Tuple[List[int], str]]:
        boxes: List[Tuple[List[int], str]] = []
        for diagram in item.diagrams:
            box = _xyxy(diagram.bbox)
            if box:
                boxes.append((box, "explicit"))
        if self._target_item(item):
            for candidate in self._page_candidates(page):
                if corridor and not _contains_center(candidate, corridor):
                    continue
                # Avoid tiny line fragments and page decorations.  The
                # detector already requires both horizontal and vertical line
                # evidence; this bound removes isolated punctuation/marks.
                if (candidate[2] - candidate[0]) < 55 or (candidate[3] - candidate[1]) < 35:
                    continue
                if _printed_overlap(page, candidate):
                    continue
                if any(_iou(candidate, old) >= 0.55 for old, _ in boxes):
                    continue
                boxes.append((candidate, "orthogonal_line_density"))
        # Stable reading order and de-duplication make links reproducible.
        boxes.sort(key=lambda entry: (entry[0][1], entry[0][0], entry[0][3], entry[0][2]))
        unique: List[Tuple[List[int], str]] = []
        for box, source in boxes:
            if any(_iou(box, old) >= 0.75 for old, _ in unique):
                continue
            unique.append((box, source))
        return unique

    def export(self, package: ExamPackage, pages: Sequence[Page], output_root: Path,
               asset_dir: Path | None = None) -> Dict[str, Any]:
        output_root = Path(output_root).resolve()
        asset_dir = Path(asset_dir or output_root / "diagram_assets")
        asset_dir.mkdir(parents=True, exist_ok=True)
        page_map = {page.index: page for page in pages}
        corridors = self._item_corridors(package, pages)
        count = 0
        item_count = 0
        errors: List[Dict[str, Any]] = []
        for section in package.sections:
            for question in section.questions:
                for item in question.items:
                    if not self._target_item(item):
                        continue
                    if item.stem_region:
                        default_page = item.stem_region.page_index
                    else:
                        regions = item.answer_regions or item.student_regions
                        default_page = regions[0].page_index if regions else None
                    page = page_map.get(default_page)
                    if page is None:
                        continue
                    boxes = self._collect_boxes(item, page, corridors.get(item.item_id))
                    if not boxes:
                        continue
                    item_count += 1
                    # Preserve explicit model/template titles where available;
                    # generated candidates receive a deterministic title.
                    old = list(item.diagrams)
                    item.diagrams = []
                    item_dir = asset_dir / _safe(item.item_id)
                    item_dir.mkdir(parents=True, exist_ok=True)
                    for index, (box, source) in enumerate(boxes, 1):
                        diagram = old[index - 1] if index <= len(old) else DiagramRef(
                            title=f"{item.item_name or item.item_id} 图形 {index}")
                        diagram.bbox = list(box)
                        diagram.page_index = int(page.index)
                        image = cv2.imread(str(page.path))
                        if image is None:
                            errors.append({"item_id": item.item_id, "page_index": page.index,
                                           "bbox": box, "reason": "IMAGE_UNREADABLE"})
                            continue
                        height, width = image.shape[:2]
                        x1 = max(0, box[0] - self.padding)
                        y1 = max(0, box[1] - self.padding)
                        x2 = min(width, box[2] + self.padding)
                        y2 = min(height, box[3] + self.padding)
                        crop = image[y1:y2, x1:x2]
                        target = item_dir / f"diagram_{index:02d}.jpg"
                        if crop.size == 0 or not cv2.imwrite(
                                str(target), crop,
                                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]):
                            errors.append({"item_id": item.item_id, "page_index": page.index,
                                           "bbox": box, "reason": "ASSET_WRITE_FAILED"})
                            continue
                        relative = target.resolve().relative_to(output_root).as_posix()
                        diagram.image_url = relative
                        diagram.audit = {
                            **(diagram.audit or {}),
                            "source": source,
                            "status": "EXPORTED",
                            "asset_path": relative,
                            "crop_bbox": [x1, y1, x2, y2],
                            "coordinate_format": "xyxy",
                        }
                        item.diagrams.append(diagram)
                        count += 1
                # A large-question figure is a property of the logical
                # question, even when localization attaches its crop to one
                # physical sub-item.  Expose one de-duplicated question-level
                # list while retaining item-level links above.
                shared: List[DiagramRef] = []
                seen = set()
                for item in question.items:
                    for diagram in item.diagrams:
                        key = (diagram.page_index, tuple(diagram.bbox), diagram.image_url)
                        if key not in seen:
                            shared.append(copy.deepcopy(diagram)); seen.add(key)
                question.diagrams = shared
        return {"diagram_assets": count, "items_with_diagrams": item_count,
                "asset_dir": str(asset_dir), "errors": errors}


def export_serialized_diagram_assets(result: Dict[str, Any], output_root: Path,
                                     asset_dir: Path | None = None) -> Dict[str, Any]:
    """Backfill links in an existing serialized result without model calls.

    This is used by debug/report tooling and follows the same page corridors
    as the live package service.  It intentionally only writes assets for
    choice and large-question items.
    """
    output_root = Path(output_root).resolve()
    asset_dir = Path(asset_dir or output_root / "diagram_assets")
    asset_dir.mkdir(parents=True, exist_ok=True)
    pages = {}
    ocr_by_page: Dict[int, List[Any]] = {}
    for page in result.get("pages", []) or []:
        path = Path(str(page.get("image", "")))
        if not path.is_absolute():
            path = output_root / path
        if not path.exists():
            path = Path(str(page.get("image", "")))
        if path.exists():
            pages[int(page.get("page_index", page.get("page", 1)))] = (path, int(page.get("width", 0) or 0), int(page.get("height", 0) or 0))
    for page_data in result.get("ocr", []) or []:
        page_index = int(page_data.get("page_index", page_data.get("page", 1)))
        ocr_by_page[page_index] = list(page_data.get("blocks", []) or [])
    items: List[Dict[str, Any]] = []
    for section in result.get("sections", []) or []:
        for question in section.get("questions", []) or []:
            for item in question.get("items", []) or []:
                if DiagramAssetService._target_item_proxy(item):
                    items.append(item)
    page_items: Dict[int, List[Dict[str, Any]]] = {}
    for item in items:
        stem = item.get("stem_region") or {}
        regions = item.get("answer_regions") or item.get("student_regions") or []
        page_index = int(stem.get("page_index", 0) or
                         (regions[0].get("page_index", 1) if regions else 1))
        # Student packages may intentionally omit stem_region after inheriting
        # the teacher topology.  Their grounded answer corridor still carries
        # the physical page identity needed for diagram assignment.
        item.setdefault("_diagram_page_index", page_index)
        page_items.setdefault(page_index, []).append(item)
    count = 0
    for page_index, item_list in page_items.items():
        page_info = pages.get(page_index)
        if not page_info:
            continue
        path, _, _ = page_info
        from .contracts import OCRBlock
        page = Page(page_index, str(path), page_info[1] or None, page_info[2] or None,
                    [OCRBlock(str(block.get("text", "")), list(block.get("bbox", [])),
                              block.get("confidence"))
                     for block in ocr_by_page.get(page_index, [])])
        candidates = DiagramAssetService._page_candidates(page)
        def _serialized_item_box(value: Dict[str, Any]) -> List[int]:
            stem = value.get("stem_region") or {}
            box = _xyxy(stem.get("bbox"))
            if box:
                return box
            regions = value.get("answer_regions") or value.get("student_regions") or []
            return _xyxy(regions[0].get("bbox")) if regions else []

        item_list.sort(key=lambda item: (_serialized_item_box(item)[1]
                                         if _serialized_item_box(item) else 0,
                                         _serialized_item_box(item)[0]
                                         if _serialized_item_box(item) else 0))
        for item in item_list:
            stem = item.get("stem_region") or {}
            regions = item.get("answer_regions") or item.get("student_regions") or []
            sb = _serialized_item_box(item)
            if not sb:
                continue
            next_top = min([_serialized_item_box(other)[1]
                            for other in item_list
                            if _serialized_item_box(other)
                            and _serialized_item_box(other)[1] > sb[1]] or [None])
            corridor = [max(0, sb[1] - 12), 0,
                        next_top - 8 if next_top is not None else (page.height or 2338),
                        page.width or 1654]
            selected = [box for box in candidates if _contains_center(box, corridor)
                        and box[2] - box[0] >= 55 and box[3] - box[1] >= 35
                        and not _printed_overlap(page, box)]
            existing = [diagram for diagram in (item.get("diagrams") or [])
                        if str((diagram.get("audit") or {}).get("source", ""))
                        not in {"orthogonal_line_density", "orthogonal_line_density_v1"}]
            if not selected:
                item["diagrams"] = existing
                continue
            item_dir = asset_dir / _safe(item.get("item_id"))
            item_dir.mkdir(parents=True, exist_ok=True)
            diagrams = []
            for index, box in enumerate(selected, 1):
                image = cv2.imread(str(path))
                if image is None:
                    continue
                h, w = image.shape[:2]; x1=max(0,box[0]-10); y1=max(0,box[1]-10); x2=min(w,box[2]+10); y2=min(h,box[3]+10)
                target=item_dir/f"diagram_{index:02d}.jpg"; cv2.imwrite(str(target), image[y1:y2,x1:x2], [cv2.IMWRITE_JPEG_QUALITY,95])
                rel=target.resolve().relative_to(output_root).as_posix()
                diagrams.append({"title": f"{item.get('item_name') or item.get('item_id')} 图形 {index}", "image_url": rel,
                                 "bbox": box, "audit": {"source":"orthogonal_line_density", "status":"EXPORTED", "asset_path":rel, "crop_bbox":[x1,y1,x2,y2], "coordinate_format":"xyxy"}, "page_index":page_index})
                count += 1
            item["diagrams"] = existing + diagrams
    for item in items:
        item.pop("_diagram_page_index", None)
    for section in result.get("sections", []) or []:
        for question in section.get("questions", []) or []:
            shared = []
            seen = set()
            for item in question.get("items", []) or []:
                for diagram in item.get("diagrams", []) or []:
                    key = (diagram.get("page_index"), tuple(diagram.get("bbox") or []),
                           diagram.get("image_url", ""))
                    if key not in seen:
                        shared.append(copy.deepcopy(diagram)); seen.add(key)
            question["diagrams"] = shared
    return {"diagram_assets": count, "asset_dir": str(asset_dir), "errors": []}


# Small duck-typed helper keeps serialized backfill independent of dataclasses.
DiagramAssetService._target_item_proxy = staticmethod(lambda item: DiagramAssetService._target_item(
    type("ItemProxy", (), {"item_type": item.get("item_type", ""), "diagrams": item.get("diagrams", [])})()))


__all__ = ["DiagramAssetService", "export_serialized_diagram_assets"]
