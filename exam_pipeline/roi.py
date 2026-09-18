"""Question-level RoI patches: the only images sent to the VLM."""
from __future__ import annotations
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from .contracts import ExamPackage, ExamSection, Page, RoIPatchRef


def xyxy_to_yxyx(box: Sequence[float]) -> List[int]:
    return [int(round(box[1])), int(round(box[0])), int(round(box[3])), int(round(box[2]))]


def yxyx_to_xyxy(box: Sequence[float]) -> List[int]:
    return [int(round(box[1])), int(round(box[0])), int(round(box[3])), int(round(box[2]))]


class RoIPatchGenerator:
    DEFAULT_PADDING = 35

    @staticmethod
    def _persist_crop(crop: Any, target: Path) -> None:
        import cv2
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=str(target.parent), prefix=f".{target.stem}.",
                                             suffix=".jpg", delete=False) as handle:
                temporary = Path(handle.name)
            if not cv2.imwrite(str(temporary), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise IOError(f"failed to persist ROI patch: {target}")
            os.replace(str(temporary), str(target))
            temporary = None
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def crop_question_roi(cls, image_bgr: Any, q_stem_bbox: Sequence[int], padding: int = 35,
                          upper_boundary: Optional[int] = None,
                          lower_boundary: Optional[int] = None) -> Tuple[List[int], Any]:
        height, width = image_bgr.shape[:2]
        ymin, xmin, ymax, xmax = [int(round(value)) for value in q_stem_bbox[:4]]
        ymin, ymax = sorted((max(0, ymin - padding), min(height, ymax + padding)))
        xmin, xmax = sorted((max(0, xmin - padding), min(width, xmax + padding)))
        if upper_boundary is not None:
            ymin = max(ymin, max(0, int(upper_boundary)))
        if lower_boundary is not None:
            ymax = min(ymax, min(height, int(lower_boundary)))
        if ymax <= ymin or xmax <= xmin:
            raise ValueError(f"invalid question ROI after clamping: {[ymin, xmin, ymax, xmax]}")
        return [ymin, xmin, ymax, xmax], image_bgr[ymin:ymax, xmin:xmax].copy()

    @staticmethod
    def _question_boxes(sections: Sequence[ExamSection], page_index: int):
        result = []
        for section in sections:
            for question in section.questions:
                regions = [region for item in question.items
                           for region in (item.answer_regions or item.student_regions)
                           if region.page_index == page_index and len(region.bbox) >= 4]
                if regions:
                    boxes = [xyxy_to_yxyx(region.bbox) for region in regions]
                    result.append((question, [min(b[0] for b in boxes), min(b[1] for b in boxes),
                                              max(b[2] for b in boxes), max(b[3] for b in boxes)]))
        return sorted(result, key=lambda pair: (pair[1][0], pair[1][1]))

    def generate(self, image_path: Path, sections: Sequence[ExamSection], page_index: int,
                 output_dir: Path, padding: int = DEFAULT_PADDING) -> List[Dict[str, object]]:
        import cv2
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        question_boxes = self._question_boxes(sections, page_index)
        records = []
        for index, (question, box) in enumerate(question_boxes):
            upper = None
            lower = None
            if index > 0 and question_boxes[index - 1][1][2] <= box[0]:
                upper = round((question_boxes[index - 1][1][2] + box[0]) / 2)
            if index + 1 < len(question_boxes) and box[2] <= question_boxes[index + 1][1][0]:
                lower = round((box[2] + question_boxes[index + 1][1][0]) / 2)
            roi_bbox, crop = self.crop_question_roi(image, box, padding, upper, lower)
            safe_id = re.sub(r"[^\w.-]+", "_", str(question.question_id)).strip("._") or f"q{index + 1}"
            target = Path(output_dir) / f"page_{page_index:02d}_{safe_id}.jpg"
            self._persist_crop(crop, target)
            for item in question.items:
                if any(region.page_index == page_index for region in (item.answer_regions or item.student_regions)):
                    item.roi_patch = RoIPatchRef(roi_bbox, str(target), padding)
            records.append({"question_id": question.question_id, "page_index": page_index,
                            "roi_crop_bbox": roi_bbox, "image_path": str(target), "padding": padding})
        return records

    def generate_page_fallback(self, image_path: Path, page_index: int, output_dir: Path) -> Dict[str, object]:
        import cv2
        image = cv2.imread(str(image_path))
        if image is None:
            raise IOError(f"cannot read page for ROI fallback: {image_path}")
        height, width = image.shape[:2]
        bbox, crop = self.crop_question_roi(image, [0, 0, height, width], padding=0)
        target = Path(output_dir) / f"page_{page_index:02d}_unsegmented.jpg"
        self._persist_crop(crop, target)
        return {"question_id": "UNSEGMENTED", "page_index": page_index, "roi_crop_bbox": bbox,
                "image_path": str(target), "padding": 0, "status": "NEEDS_LAYOUT_REVIEW"}

    def generate_package(self, package: ExamPackage, pages: Sequence[Page], output_dir: Path):
        records = []
        for page in pages:
            records.extend(self.generate(Path(page.path), package.sections, page.index, output_dir))
        return records


__all__ = ["RoIPatchGenerator", "xyxy_to_yxyx", "yxyx_to_xyxy"]
