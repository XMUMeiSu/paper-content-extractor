"""
Student Answer Localization Service

Refines student answer regions based on ink detection instead of inheriting
teacher template bboxes verbatim. This addresses the core issue where teacher
blank boxes don't correspond to actual student handwriting positions.

Key improvements:
1. Ink-based region detection: Identifies actual handwritten content
2. Spatial clustering: Groups ink into coherent answer blocks
3. Slot association: Maps ink regions to teacher slots by proximity
4. Bbox tightening: Generates compact regions containing only student answers
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class InkRegion:
    """Represents a detected ink region (student handwriting)"""
    bbox: List[float]  # [x1, y1, x2, y2]
    page_index: int
    area: float
    density: float  # Ink pixel ratio within bbox
    centroid: Tuple[float, float]
    confidence: float = 1.0
    associated_slot_id: Optional[str] = None


@dataclass
class RefinedStudentRegion:
    """Refined student answer region for a specific slot"""
    slot_id: str
    item_id: str
    bbox: List[float]  # Tight bbox around actual handwriting
    page_index: int
    ink_regions: List[InkRegion] = field(default_factory=list)
    confidence: float = 1.0
    method: str = "ink_based"  # vs "template_fallback"


class StudentAnswerLocalizationService:
    """
    Localizes student answers by detecting handwritten ink regions
    rather than inheriting teacher template coordinates.
    """

    # Morphology parameters for ink detection
    INK_MORPH_KERNEL_SIZE = (3, 3)
    INK_CLOSE_KERNEL_SIZE = (15, 5)  # Horizontal closure to connect characters

    # Minimum ink region thresholds
    MIN_INK_AREA = 50  # pixels
    MIN_INK_DENSITY = 0.01  # 1% ink coverage

    # Proximity thresholds for slot association
    MAX_SLOT_DISTANCE = 200  # pixels from teacher slot center
    VERTICAL_WEIGHT = 0.5  # Vertical distance matters less than horizontal

    # Bbox expansion for OCR context
    BBOX_EXPAND_X = 5  # pixels
    BBOX_EXPAND_Y = 5

    def __init__(self, policy=None):
        self.policy = policy
        self._ink_cache: Dict[Tuple[str, int], np.ndarray] = {}

    def _extract_ink_mask(self, image_path: str, page_index: int) -> np.ndarray:
        """
        Extract binary ink mask from student page image.
        Reuses existing ink separation logic if available.
        """
        cache_key = (image_path, page_index)
        if cache_key in self._ink_cache:
            return self._ink_cache[cache_key]

        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive threshold to separate ink from background
        # This works for both pen and pencil handwriting
        ink_mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 15
        )

        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.INK_MORPH_KERNEL_SIZE)
        ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, kernel)

        # Close operation to connect nearby characters
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.INK_CLOSE_KERNEL_SIZE)
        ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, close_kernel)

        self._ink_cache[cache_key] = ink_mask
        return ink_mask

    def _detect_ink_regions(self, ink_mask: np.ndarray, page_index: int,
                           search_area: Optional[List[float]] = None) -> List[InkRegion]:
        """
        Detect individual ink regions (connected components) in the mask.

        Args:
            ink_mask: Binary mask where 255 = ink
            page_index: Page number
            search_area: Optional [x1, y1, x2, y2] to limit search
        """
        # Apply search area mask if provided
        if search_area:
            x1, y1, x2, y2 = map(int, search_area)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(ink_mask.shape[1], x2), min(ink_mask.shape[0], y2)

            search_mask = np.zeros_like(ink_mask)
            search_mask[y1:y2, x1:x2] = ink_mask[y1:y2, x1:x2]
            ink_mask = search_mask

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            ink_mask, connectivity=8
        )

        regions = []
        for i in range(1, num_labels):  # Skip background (label 0)
            x, y, w, h, area = stats[i]

            # Filter out noise
            if area < self.MIN_INK_AREA:
                continue

            bbox = [float(x), float(y), float(x + w), float(y + h)]

            # Calculate ink density
            roi = ink_mask[y:y+h, x:x+w]
            density = np.sum(roi > 0) / (w * h) if w * h > 0 else 0

            if density < self.MIN_INK_DENSITY:
                continue

            regions.append(InkRegion(
                bbox=bbox,
                page_index=page_index,
                area=float(area),
                density=density,
                centroid=(float(centroids[i][0]), float(centroids[i][1])),
                confidence=min(1.0, density * 2)  # Higher density = higher confidence
            ))

        return regions

    def _cluster_ink_regions(self, regions: List[InkRegion],
                            max_gap: float = 50) -> List[List[InkRegion]]:
        """
        Cluster nearby ink regions into coherent answer blocks.
        Uses horizontal proximity as primary criterion.
        """
        if not regions:
            return []

        # Sort by reading order (top to bottom, left to right)
        sorted_regions = sorted(regions, key=lambda r: (r.centroid[1], r.centroid[0]))

        clusters = []
        current_cluster = [sorted_regions[0]]

        for region in sorted_regions[1:]:
            prev = current_cluster[-1]

            # Calculate horizontal and vertical gaps
            h_gap = region.bbox[0] - prev.bbox[2]  # Left edge to right edge
            v_gap = abs(region.centroid[1] - prev.centroid[1])

            # Cluster if within gap threshold and roughly same line
            if h_gap < max_gap and v_gap < 30:
                current_cluster.append(region)
            else:
                clusters.append(current_cluster)
                current_cluster = [region]

        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def _associate_ink_to_slot(self, ink_cluster: List[InkRegion],
                               teacher_slot: Dict[str, Any]) -> float:
        """
        Calculate association score between ink cluster and teacher slot.
        Returns confidence score (0-1).
        """
        if not ink_cluster or not teacher_slot.get("expected_bbox"):
            return 0.0

        # Get cluster center
        cluster_x = np.mean([r.centroid[0] for r in ink_cluster])
        cluster_y = np.mean([r.centroid[1] for r in ink_cluster])

        # Get teacher slot center
        teacher_bbox = teacher_slot["expected_bbox"]
        slot_cx = (teacher_bbox[0] + teacher_bbox[2]) / 2
        slot_cy = (teacher_bbox[1] + teacher_bbox[3]) / 2

        # Calculate weighted distance
        dx = abs(cluster_x - slot_cx)
        dy = abs(cluster_y - slot_cy) * self.VERTICAL_WEIGHT
        distance = np.sqrt(dx**2 + dy**2)

        # Convert distance to confidence (closer = higher confidence)
        if distance > self.MAX_SLOT_DISTANCE:
            return 0.0

        confidence = 1.0 - (distance / self.MAX_SLOT_DISTANCE)
        return confidence

    def _compute_cluster_bbox(self, cluster: List[InkRegion]) -> List[float]:
        """Compute tight bounding box around ink cluster with expansion for OCR."""
        if not cluster:
            return [0.0, 0.0, 0.0, 0.0]

        x1 = min(r.bbox[0] for r in cluster) - self.BBOX_EXPAND_X
        y1 = min(r.bbox[1] for r in cluster) - self.BBOX_EXPAND_Y
        x2 = max(r.bbox[2] for r in cluster) + self.BBOX_EXPAND_X
        y2 = max(r.bbox[3] for r in cluster) + self.BBOX_EXPAND_Y

        return [max(0, x1), max(0, y1), x2, y2]

    def refine_student_regions(self,
                              student_page_path: str,
                              page_index: int,
                              teacher_slots: List[Dict[str, Any]],
                              fallback_search_area: Optional[List[float]] = None
                              ) -> List[RefinedStudentRegion]:
        """
        Main entry point: Refine student answer regions for a page.

        Args:
            student_page_path: Path to student page image
            page_index: Page number
            teacher_slots: Teacher slots with expected_bbox
            fallback_search_area: Fallback corridor if no ink detected

        Returns:
            List of refined student regions, one per associated slot
        """
        # Extract ink mask
        try:
            ink_mask = self._extract_ink_mask(student_page_path, page_index)
        except Exception as e:
            # Fallback: return template-based regions
            return self._fallback_regions(teacher_slots, fallback_search_area, str(e))

        # Detect ink regions
        ink_regions = self._detect_ink_regions(ink_mask, page_index, fallback_search_area)

        if not ink_regions:
            # No ink detected: fallback to template
            return self._fallback_regions(teacher_slots, fallback_search_area, "no_ink")

        # Cluster ink into answer blocks
        ink_clusters = self._cluster_ink_regions(ink_regions)

        # Associate clusters to teacher slots
        refined_regions = []
        used_clusters = set()

        for slot in teacher_slots:
            best_cluster = None
            best_score = 0.0
            best_idx = -1

            for idx, cluster in enumerate(ink_clusters):
                if idx in used_clusters:
                    continue

                score = self._associate_ink_to_slot(cluster, slot)
                if score > best_score:
                    best_score = score
                    best_cluster = cluster
                    best_idx = idx

            if best_cluster and best_score > 0.3:  # Minimum confidence threshold
                used_clusters.add(best_idx)
                refined_bbox = self._compute_cluster_bbox(best_cluster)

                refined_regions.append(RefinedStudentRegion(
                    slot_id=slot.get("slot_id", ""),
                    item_id=slot.get("parent_item_id", ""),
                    bbox=refined_bbox,
                    page_index=page_index,
                    ink_regions=best_cluster,
                    confidence=best_score,
                    method="ink_based"
                ))
            else:
                # Fallback for this slot: use template bbox with low confidence
                refined_regions.append(RefinedStudentRegion(
                    slot_id=slot.get("slot_id", ""),
                    item_id=slot.get("parent_item_id", ""),
                    bbox=slot.get("expected_bbox", [0, 0, 0, 0]),
                    page_index=page_index,
                    confidence=0.2,
                    method="template_fallback"
                ))

        return refined_regions

    def _fallback_regions(self, teacher_slots: List[Dict[str, Any]],
                         search_area: Optional[List[float]],
                         reason: str) -> List[RefinedStudentRegion]:
        """Generate fallback regions using template bboxes."""
        regions = []
        for slot in teacher_slots:
            bbox = slot.get("expected_bbox") or search_area or [0, 0, 0, 0]
            regions.append(RefinedStudentRegion(
                slot_id=slot.get("slot_id", ""),
                item_id=slot.get("parent_item_id", ""),
                bbox=bbox,
                page_index=slot.get("page_index", 0),
                confidence=0.1,
                method=f"template_fallback:{reason}"
            ))
        return regions

    def localize(self, package, pages, source_pages, ocr_engine, language):
        """
        Integration method for homework_extractor.py pipeline.

        Localizes student answers by detecting ink regions and updating
        student_regions in the package items.

        Args:
            package: ExamPackage with items to localize
            pages: Student pages
            source_pages: Teacher reference pages (for slot info)
            ocr_engine: OCR engine (unused, for API compatibility)
            language: Language code (unused, for API compatibility)

        Returns:
            Summary dict with statistics
        """
        from .contracts import PageRegion

        total_items = 0
        localized_items = 0
        total_reduction = 0.0

        # Iterate through the package hierarchy: sections -> questions -> items
        for section in package.sections:
            for question in section.questions:
                for item in question.items:
                    total_items += 1

                    # Skip if no slots to localize
                    if not item.slots:
                        continue

                    # Get page info
                    page_index = item.slots[0].page_index if item.slots else 0
                    if page_index >= len(pages):
                        continue

                    page = pages[page_index]
                    page_path = page.path

                    # Prepare teacher slots for matching
                    teacher_slots = []
                    for slot in item.slots:
                        teacher_slots.append({
                            "slot_id": slot.slot_idx,
                            "parent_item_id": item.item_id,
                            "expected_bbox": slot.expected_bbox,
                            "page_index": slot.page_index
                        })

                    # Determine search area from existing student_regions
                    search_area = None
                    if item.student_regions:
                        # Use union of existing regions as search corridor
                        bboxes = [r.bbox for r in item.student_regions if len(r.bbox) >= 4]
                        if bboxes:
                            x1 = min(b[0] for b in bboxes)
                            y1 = min(b[1] for b in bboxes)
                            x2 = max(b[2] for b in bboxes)
                            y2 = max(b[3] for b in bboxes)
                            search_area = [x1, y1, x2, y2]

                    # Refine regions
                    try:
                        refined = self.refine_student_regions(
                            page_path, page_index, teacher_slots, search_area
                        )

                        # Update student_regions
                        new_regions = []
                        for refined_region in refined:
                            if refined_region.method.startswith("ink_based"):
                                localized_items += 1

                                # Calculate reduction
                                if search_area:
                                    old_area = (search_area[2] - search_area[0]) * (search_area[3] - search_area[1])
                                    new_area = (refined_region.bbox[2] - refined_region.bbox[0]) * \
                                              (refined_region.bbox[3] - refined_region.bbox[1])
                                    if old_area > 0:
                                        reduction = 100.0 * (1.0 - new_area / old_area)
                                        total_reduction += max(0, reduction)

                            new_regions.append(PageRegion(
                                page_index=refined_region.page_index,
                                page_file=page_path,
                                bbox=refined_region.bbox,
                                confidence=refined_region.confidence,
                                ocr_text=None
                            ))

                        if new_regions:
                            item.student_regions = new_regions

                    except Exception as e:
                        # Silently skip on error, keep existing regions
                        pass

        avg_reduction = total_reduction / max(1, localized_items)

        return {
            "total_items": total_items,
            "localized_items": localized_items,
            "avg_reduction_percent": avg_reduction
        }


__all__ = ["StudentAnswerLocalizationService", "RefinedStudentRegion", "InkRegion"]
