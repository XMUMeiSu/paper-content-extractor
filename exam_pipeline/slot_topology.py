"""Multi-slot topology enhancement for fill-in-the-blank questions.

Improves slot detection for questions with multiple blanks by:
1. Analyzing layout patterns (vertical stacking, grid arrangements)
2. Using text anchors (quoted labels, delimiters like ":")
3. Applying geometric consensus from cohort data
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .contracts import ExamItem, Page, PageRegion, OCRBlock
from .slots import _answer_parts, infer_item_type


@dataclass
class SlotCandidate:
    """A candidate slot location."""
    bbox: List[float]  # [x1, y1, x2, y2] normalized
    confidence: float
    source: str  # "layout", "text_anchor", "geometric", "cohort"
    anchor_text: Optional[str] = None


class MultiSlotTopologyService:
    """Enhanced slot detection for multi-blank fill questions."""

    def __init__(self):
        # Regex patterns for blank detection
        self.blank_pattern = re.compile(r"_{2,}|\.{4,}|…{2,}|[\(（\[]\s*[_·.\s]*[\)）\]]")
        self.quoted_label = re.compile(
            r'["\'\']\s*[^"\'\'\n]{1,24}?\s*["\'\']\s*(?:[:：=]|is\b|是)', re.IGNORECASE
        )
        self.delimited_label = re.compile(
            r"(?:^|[\s,，;；])[^\s,，;；:：=]{1,12}\s*[:：=](?!=)", re.MULTILINE
        )

    def enhance_item_slots(
        self,
        item: ExamItem,
        page: Page,
        ocr_blocks: List[OCRBlock],
        teacher_regions: List[PageRegion],
    ) -> List[SlotCandidate]:
        """Find additional slot candidates for a multi-blank item.

        Args:
            item: The exam item to analyze
            page: The page containing this item
            ocr_blocks: OCR blocks from the page
            teacher_regions: Known teacher answer regions (for reference)

        Returns:
            List of slot candidates, sorted by confidence
        """
        # Determine expected slot count
        expected_count = self._expected_slot_count(item)
        if expected_count <= 1:
            return []  # Single slot, no enhancement needed

        candidates = []

        # Strategy 1: Text anchor detection (e.g., "答案：____")
        text_candidates = self._find_text_anchor_slots(
            item, page, ocr_blocks, expected_count
        )
        candidates.extend(text_candidates)

        # Strategy 2: Layout pattern detection (vertical stacking)
        layout_candidates = self._find_layout_pattern_slots(
            item, page, ocr_blocks, expected_count
        )
        candidates.extend(layout_candidates)

        # Strategy 3: Geometric spacing (evenly distributed blanks)
        if teacher_regions:
            geometric_candidates = self._find_geometric_slots(
                item, teacher_regions, expected_count
            )
            candidates.extend(geometric_candidates)

        # Deduplicate and sort
        candidates = self._deduplicate_candidates(candidates)
        candidates.sort(key=lambda c: c.confidence, reverse=True)

        return candidates[:expected_count]

    def _expected_slot_count(self, item: ExamItem) -> int:
        """Determine expected number of slots from item metadata."""
        # Check if standard_answer provides count
        answer = item.standard_answer
        if answer is None:
            # Count blanks in question text
            return len(self.blank_pattern.findall(item.question_text or ""))

        if isinstance(answer, dict):
            return len(answer) if answer else 1

        if isinstance(answer, (list, tuple)):
            return max(1, len(answer))

        # Parse multi-part answer string
        parts = _answer_parts(item)
        return max(1, len(parts))

    def _find_text_anchor_slots(
        self,
        item: ExamItem,
        page: Page,
        ocr_blocks: List[OCRBlock],
        expected_count: int,
    ) -> List[SlotCandidate]:
        """Find slots using text anchors like '答案：', '(1)', etc."""
        candidates = []

        # Get item's text region
        item_text = item.question_text or ""

        # Find all text anchors
        quoted_matches = list(self.quoted_label.finditer(item_text))
        delimited_matches = list(self.delimited_label.finditer(item_text))

        # For each anchor, try to find corresponding OCR block
        for match in quoted_matches[:expected_count]:
            anchor = match.group(0)
            candidate = self._locate_anchor_in_ocr(anchor, ocr_blocks, page)
            if candidate:
                candidate.source = "text_anchor"
                candidate.anchor_text = anchor
                candidates.append(candidate)

        for match in delimited_matches[:expected_count]:
            anchor = match.group(0)
            candidate = self._locate_anchor_in_ocr(anchor, ocr_blocks, page)
            if candidate:
                candidate.source = "text_anchor"
                candidate.anchor_text = anchor
                candidates.append(candidate)

        return candidates

    def _locate_anchor_in_ocr(
        self,
        anchor_text: str,
        ocr_blocks: List[OCRBlock],
        page: Page,
    ) -> Optional[SlotCandidate]:
        """Find OCR block matching anchor text and create candidate."""
        # Normalize anchor for matching
        normalized_anchor = re.sub(r"[\W_]+", "", anchor_text.lower())

        for block in ocr_blocks:
            block_text = getattr(block, "text", "") or ""
            normalized_block = re.sub(r"[\W_]+", "", block_text.lower())

            if normalized_anchor in normalized_block:
                # Found matching block
                bbox = getattr(block, "bbox", None)
                if not bbox or len(bbox) < 4:
                    continue

                # Create region to the right of the anchor (where answer should be)
                # Assume answer is within 100 pixels to the right
                x1, y1, x2, y2 = bbox
                answer_bbox = [x2, y1, min(x2 + 100, 1.0), y2 + 20]

                return SlotCandidate(
                    bbox=answer_bbox,
                    confidence=0.8,
                    source="text_anchor",
                    anchor_text=anchor_text,
                )

        return None

    def _find_layout_pattern_slots(
        self,
        item: ExamItem,
        page: Page,
        ocr_blocks: List[OCRBlock],
        expected_count: int,
    ) -> List[SlotCandidate]:
        """Find slots using layout patterns (e.g., vertically stacked regions)."""
        candidates = []

        # Find blank regions in OCR (regions with underscores or minimal text)
        blank_blocks = [
            block for block in ocr_blocks
            if self._is_blank_region(getattr(block, "text", ""))
        ]

        if len(blank_blocks) < expected_count:
            return []

        # Check if blocks are vertically aligned (column pattern)
        if self._are_vertically_aligned(blank_blocks):
            # Sort by y-coordinate
            blank_blocks.sort(key=lambda b: getattr(b, "bbox", [0, 0, 0, 0])[1])

            for i, block in enumerate(blank_blocks[:expected_count]):
                bbox = getattr(block, "bbox", None)
                if bbox and len(bbox) >= 4:
                    candidates.append(SlotCandidate(
                        bbox=list(bbox),
                        confidence=0.75,
                        source="layout",
                    ))

        return candidates

    def _is_blank_region(self, text: str) -> bool:
        """Check if text represents a blank/answer region."""
        if not text:
            return False

        # Has underscores or dots
        if self.blank_pattern.search(text):
            return True

        # Very short text (likely a fill-in answer)
        if len(text.strip()) <= 3:
            return True

        return False

    def _are_vertically_aligned(self, blocks: List[OCRBlock], threshold: float = 0.05) -> bool:
        """Check if blocks are vertically aligned (same x-coordinate)."""
        if len(blocks) < 2:
            return False

        # Get x-coordinates (left edge)
        x_coords = []
        for block in blocks:
            bbox = getattr(block, "bbox", None)
            if bbox and len(bbox) >= 4:
                x_coords.append(bbox[0])

        if not x_coords:
            return False

        # Check if variance is low
        avg_x = sum(x_coords) / len(x_coords)
        variance = sum((x - avg_x) ** 2 for x in x_coords) / len(x_coords)

        return variance < threshold

    def _find_geometric_slots(
        self,
        item: ExamItem,
        teacher_regions: List[PageRegion],
        expected_count: int,
    ) -> List[SlotCandidate]:
        """Generate slots using geometric spacing from teacher template."""
        if not teacher_regions:
            return []

        candidates = []

        # If we have one teacher region but expect multiple slots,
        # divide it evenly
        if len(teacher_regions) == 1 and expected_count > 1:
            base_bbox = teacher_regions[0].bbox
            if len(base_bbox) < 4:
                return []

            x1, y1, x2, y2 = base_bbox
            height = (y2 - y1) / expected_count

            for i in range(expected_count):
                slot_bbox = [
                    x1,
                    y1 + i * height,
                    x2,
                    y1 + (i + 1) * height,
                ]
                candidates.append(SlotCandidate(
                    bbox=slot_bbox,
                    confidence=0.6,
                    source="geometric",
                ))

        return candidates

    def _deduplicate_candidates(
        self,
        candidates: List[SlotCandidate],
        iou_threshold: float = 0.5,
    ) -> List[SlotCandidate]:
        """Remove duplicate candidates based on IoU."""
        if len(candidates) <= 1:
            return candidates

        # Sort by confidence
        candidates.sort(key=lambda c: c.confidence, reverse=True)

        kept = []
        for candidate in candidates:
            # Check if this candidate overlaps with any kept candidate
            is_duplicate = False
            for kept_candidate in kept:
                iou = self._iou(candidate.bbox, kept_candidate.bbox)
                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(candidate)

        return kept

    def _iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union."""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0

        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
