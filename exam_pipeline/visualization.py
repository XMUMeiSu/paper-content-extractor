"""Deterministic slot overlays for reviewing persisted extraction geometry."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import cv2
import numpy as np

from .contracts import ExamPackage, Page


def _xyxy_from_runtime(box: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        # Slot runtime boxes are yxyx; visual/public boxes are xyxy.
        top, left, bottom, right = [int(round(float(value))) for value in box]
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _draw_box(image, box, color, thickness):
    if box is None:
        return False
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return True


def render_slot_overlays(package: ExamPackage, pages: Sequence[Page], output_dir: Path,
                         name: str) -> Dict[str, Any]:
    """Write per-page and contact-sheet slot images without image inference."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    page_files = []
    for page in pages:
        image = cv2.imread(str(page.path))
        if image is None:
            continue
        for section in package.sections:
            for question in section.questions:
                for item in question.items:
                    for number, slot in enumerate(item.slots, 1):
                        if slot.page_index != page.index:
                            continue
                        expected = _xyxy_from_runtime(slot.expected_bbox)
                        recognition = _xyxy_from_runtime(slot.recognition_bbox)
                        evidence = _xyxy_from_runtime(slot.evidence_bbox or slot.handwriting_bbox)
                        _draw_box(image, expected, (0, 165, 255), 1)
                        _draw_box(image, recognition, (255, 0, 0), 1)
                        _draw_box(image, evidence, (0, 180, 0), 2)
                        label_box = evidence or recognition or expected
                        if label_box:
                            label = f"{item.item_id}:s{slot.slot_idx or number} {slot.geometry_status}"
                            cv2.putText(image, label,
                                        (label_box[0], max(18, label_box[1] - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 220), 1,
                                        cv2.LINE_AA)
        cv2.putText(image, f"page {page.index} | expected orange / evidence green / recognition blue",
                    (24, 38), cv2.FONT_HERSHEY_SIMPLEX, .72, (20, 20, 20), 2, cv2.LINE_AA)
        target = output_dir / f"{name}_page_{page.index:02d}_slots.jpg"
        if cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            page_files.append(str(target))
            scale = min(1.0, 1500.0 / max(1, image.shape[0]))
            view = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
            rendered.append((page.index, view))
    if not rendered:
        return {"status": "NO_PAGE_IMAGES", "pages": [], "contact_sheet": None}
    gap = 18
    height = max(view.shape[0] for _, view in rendered)
    width = sum(view.shape[1] for _, view in rendered) + gap * (len(rendered) - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    offset = 0
    for _, view in rendered:
        canvas[:view.shape[0], offset:offset + view.shape[1]] = view
        offset += view.shape[1] + gap
    contact = output_dir / f"{name}_slots.jpg"
    if not cv2.imwrite(str(contact), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        return {"status": "CONTACT_SHEET_WRITE_FAILED", "pages": page_files,
                "contact_sheet": None}
    return {"status": "EXPORTED", "pages": page_files, "contact_sheet": str(contact),
            "page_count": len(page_files)}


__all__ = ["render_slot_overlays"]
