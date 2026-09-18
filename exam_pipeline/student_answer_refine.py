"""Student answer region refinement service.

Refines coarse student_regions to precise handwritten answer bounding boxes,
eliminating OCR garbage from question text, headers, and margins.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict
import numpy as np
from PIL import Image

from homework_extractor import ExamPackage, ExamQuestion, ExamItem, PageRegion, Page


@dataclass
class HandwritingCandidate:
    """A candidate handwritten region."""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    text: Optional[str] = None
    source: str = "layout"  # "layout" or "ocr_filter"


class StudentAnswerRefineService:
    """Refine student answer regions to exclude printed text and focus on handwriting."""

    def __init__(self, policy=None):
        self.policy = policy
        self.ocr_engine = None  # Will be set during refine() if needed

    def refine(
        self,
        package: ExamPackage,
        pages: List[Page],
        ocr_engine: str,
        language: str,
    ) -> Dict[str, Any]:
        """Refine all student_regions in the package to focus on handwritten answers.

        Returns summary statistics.
        """
        self.ocr_engine = ocr_engine

        total_items = 0
        refined_items = 0
        improved_items = 0  # regions that got smaller/more precise

        for section in package.sections:
            for question in section.questions:
                for item in question.items:
                    if not item.student_regions:
                        continue

                    total_items += 1

                    # Find the page for this item
                    page = self._find_page_for_item(item, pages)
                    if not page:
                        continue

                    # Refine each region
                    original_regions = item.student_regions.copy()
                    refined_regions = []

                    for region in original_regions:
                        if len(region.bbox) < 4:
                            refined_regions.append(region)
                            continue

                        # Attempt refinement
                        candidates = self._find_handwriting_in_region(
                            region, page, question, item
                        )

                        if candidates:
                            # Use the best candidate(s)
                            for candidate in candidates[:3]:  # Top 3 at most
                                refined_regions.append(PageRegion(
                                    page_index=region.page_index,
                                    page_file=region.page_file,
                                    bbox=candidate.bbox,
                                    ocr_text=candidate.text or region.ocr_text,
                                    confidence=candidate.confidence,
                                ))

                            # Check if refined region is smaller (improvement)
                            original_area = self._bbox_area(region.bbox)
                            refined_area = sum(self._bbox_area(c.bbox) for c in candidates[:3])
                            if refined_area < original_area * 0.8:
                                improved_items += 1
                        else:
                            # Keep original if refinement fails
                            refined_regions.append(region)

                    if refined_regions != original_regions:
                        item.student_regions = refined_regions
                        refined_items += 1

        return {
            "total_items": total_items,
                            "refined_items": refined_items,
            "improved_items": improved_items,
        }

    def _find_page_for_item(self, item: ExamItem, pages: List[Page]) -> Optional[Page]:
        """Find the page containing this item."""
        if not item.student_regions:
            return None

        # Use the first region's page number
        region = item.student_regions[0]
        page_num = region.page_index

        for page in pages:
            if page.index == page_num:
                return page

        return None

    def _find_handwriting_in_region(
        self,
        region: PageRegion,
        page: Page,
        question: ExamQuestion,
        item: ExamItem,
    ) -> List[HandwritingCandidate]:
        """Find handwritten answer candidates within a coarse region.

        Strategy:
        1. Extract sub-region image
        2. Use layout detection to find "handwriting" or "text" blocks
        3. Filter out printed text using OCR + heuristics
        4. Return sorted candidates (highest confidence first)
        """
        bbox = region.bbox
        if len(bbox) < 4:
            return []

        # Load page image
        try:
            img = Image.open(page.path)
            img_width, img_height = img.size
        except Exception:
            return []

        # Convert normalized coordinates to pixels
        x1, y1, x2, y2 = bbox
        px1 = int(x1 * img_width) if x1 <= 1 else int(x1)
        py1 = int(y1 * img_height) if y1 <= 1 else int(y1)
        px2 = int(x2 * img_width) if x2 <= 1 else int(x2)
        py2 = int(y2 * img_height) if y2 <= 1 else int(y2)

        # Ensure valid crop
        px1 = max(0, min(px1, img_width - 1))
        px2 = max(px1 + 1, min(px2, img_width))
        py1 = max(0, min(py1, img_height - 1))
        py2 = max(py1 + 1, min(py2, img_height))

        # Extract sub-region
        try:
            sub_img = img.crop((px1, py1, px2, py2))
        except Exception:
            return []

        candidates = []

        # Strategy 1: Layout-based detection
        layout_candidates = self._detect_handwriting_layout(
            sub_img, (px1, py1, px2, py2), (img_width, img_height)
        )
        candidates.extend(layout_candidates)

        # Strategy 2: OCR-based filtering (identify and exclude printed text regions)
        if not candidates:
            ocr_candidates = self._filter_printed_text(
                sub_img, region, (px1, py1, px2, py2), (img_width, img_height)
            )
            candidates.extend(ocr_candidates)

        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)

        return candidates

    def _detect_handwriting_layout(
        self,
        img: Image.Image,
        crop_box: Tuple[int, int, int, int],
        img_size: Tuple[int, int],
    ) -> List[HandwritingCandidate]:
        """Use layout detection to find handwriting blocks."""
        # TODO: Integrate with actual layout detector (e.g., PaddleOCR layout analysis)
        # For now, return empty - this is a placeholder for future enhancement
        return []

    def _filter_printed_text(
        self,
        img: Image.Image,
        region: PageRegion,
        crop_box: Tuple[int, int, int, int],
        img_size: Tuple[int, int],
    ) -> List[HandwritingCandidate]:
        """Filter out printed text areas, keeping likely handwritten regions.

        Heuristics:
        - Printed text: uniform spacing, consistent font, horizontal alignment
        - Handwriting: irregular spacing, varying stroke width, may be tilted
        """
        # Run OCR on the region
        if not self.ocr_engine or self.ocr_engine == "none":
            return []

        try:
            from exam_pipeline.ocr_service import run_ocr_on_image
            ocr_result = run_ocr_on_image(img, self.ocr_engine)
        except Exception:
            return []

        if not ocr_result or "lines" not in ocr_result:
            return []

        px1, py1, px2, py2 = crop_box
        img_width, img_height = img_size

        # Analyze OCR lines to distinguish printed vs handwritten
        handwriting_boxes = []

        for line in ocr_result.get("lines", []):
            bbox = line.get("bbox")
            text = line.get("text", "")
            conf = line.get("confidence", 0.0)

            if not bbox or len(bbox) < 4:
                continue

            # Convert line bbox (relative to sub-image) back to full image coords
            lx1, ly1, lx2, ly2 = bbox
            full_x1 = px1 + lx1
            full_y1 = py1 + ly1
            full_x2 = px1 + lx2
            full_y2 = py1 + ly2

            # Normalize to [0, 1]
            norm_bbox = [
                full_x1 / img_width,
                full_y1 / img_height,
                full_x2 / img_width,
                full_y2 / img_height,
            ]

            # Heuristic: Low confidence often indicates handwriting (harder to OCR)
            # Also: Non-horizontal text, large character spacing
            is_likely_handwriting = (
                conf < 0.7  # Low OCR confidence
                or self._has_irregular_spacing(text)
                or not self._is_horizontal(bbox)
            )

            if is_likely_handwriting:
                handwriting_boxes.append(HandwritingCandidate(
                    bbox=norm_bbox,
                    confidence=max(0.3, 1.0 - conf),  # Inverse of OCR confidence
                    text=text,
                    source="ocr_filter",
                ))

        # Merge nearby handwriting boxes
        merged = self._merge_nearby_boxes(handwriting_boxes, threshold=0.02)

        return merged

    def _has_irregular_spacing(self, text: str) -> bool:
        """Check if text has irregular character spacing (handwriting indicator)."""
        if len(text) < 2:
            return False

        # Heuristic: Multiple spaces or very short words suggest irregular layout
        space_count = text.count(" ")
        return space_count > len(text) / 3

    def _is_horizontal(self, bbox: List[float]) -> bool:
        """Check if bbox is roughly horizontal."""
        if len(bbox) < 4:
            return True

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Horizontal if width > 2 * height
        return width > 2 * height

    def _merge_nearby_boxes(
        self, candidates: List[HandwritingCandidate], threshold: float = 0.02
    ) -> List[HandwritingCandidate]:
        """Merge candidates that are close to each other."""
        if not candidates:
            return []

        # Simple greedy merge
        merged = []
        used = set()

        for i, c1 in enumerate(candidates):
            if i in used:
                continue

            # Find all nearby candidates
            group = [c1]
            used.add(i)

            for j, c2 in enumerate(candidates[i+1:], start=i+1):
                if j in used:
                    continue

                if self._boxes_are_close(c1.bbox, c2.bbox, threshold):
                    group.append(c2)
                    used.add(j)

            # Merge the group
            if len(group) == 1:
                merged.append(c1)
            else:
                merged_bbox = self._merge_bboxes([c.bbox for c in group])
                merged_text = " ".join(c.text or "" for c in group).strip()
                avg_conf = sum(c.confidence for c in group) / len(group)

                merged.append(HandwritingCandidate(
                    bbox=merged_bbox,
                    confidence=avg_conf,
                    text=merged_text,
                    source="merged",
                ))

        return merged

    def _boxes_are_close(self, bbox1: List[float], bbox2: List[float], threshold: float) -> bool:
        """Check if two boxes are within threshold distance."""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return False

        # Calculate center distance
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2

        dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

        return dist < threshold

    def _merge_bboxes(self, bboxes: List[List[float]]) -> List[float]:
        """Merge multiple bboxes into one encompassing box."""
        x1 = min(b[0] for b in bboxes)
        y1 = min(b[1] for b in bboxes)
        x2 = max(b[2] for b in bboxes)
        y2 = max(b[3] for b in bboxes)
        return [x1, y1, x2, y2]

    def _bbox_area(self, bbox: List[float]) -> float:
        """Calculate bbox area."""
        if len(bbox) < 4:
            return 0.0
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
