"""Deterministic geometry quality gates."""
from typing import Any, Dict, List, Sequence
from .contracts import ExamPackage, Page, PageRegion


def _area(box):
    return max(0., float(box[2])-float(box[0])) * max(0., float(box[3])-float(box[1])) if len(box) >= 4 else 0.


def validate_region(region: PageRegion, page: Page, max_page_ratio: float = .60) -> Dict[str, Any]:
    width, height = float(page.width or 1654), float(page.height or 2338)
    box, reasons = region.bbox, []
    if len(box) < 4 or box[2] <= box[0] or box[3] <= box[1]: reasons.append("invalid bbox")
    ratio = _area(box) / max(1., width*height)
    if ratio > max_page_ratio: reasons.append(f"answer box covers {ratio*100:.1f}% of page")
    if len(box) >= 4 and (box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height):
        reasons.append("bbox out of page bounds")
    return {"status": "NEED_REVIEW" if reasons else "OK", "reasons": reasons, "area_ratio": round(ratio, 4)}


def validate_item_regions(package: ExamPackage, pages: Sequence[Page], overlap_threshold: float = .35):
    page_map = {page.index: page for page in pages}
    reasons, region_count = [], 0
    for section in package.sections:
        for question in section.questions:
            for item in question.items:
                item_reasons, checks = [], []
                for region in item.student_regions or item.answer_regions:
                    page = page_map.get(region.page_index) or Page(region.page_index, region.page_file, 1654, 2338, [])
                    check = validate_region(region, page); checks.append(check); region_count += 1
                    item_reasons.extend(check["reasons"])
                if item.confidence is not None and float(item.confidence) < .70:
                    item_reasons.append(f"OCR confidence below threshold: {float(item.confidence):.2f}")
                for slot in item.slots:
                    if (package.document_type == "teacher"
                            and (slot.status == "TEACHER_ANSWER_NEEDS_REVIEW"
                                 or not slot.expected_text)):
                        item_reasons.append(
                            f"teacher answer missing for slot {slot.slot_idx}"
                        )
                    if slot.geometry_status == "MISSING":
                        item_reasons.append(f"slot {slot.slot_idx} was not found on the student page")
                    elif slot.geometry_status in {"FAILED", "UNCERTAIN"}:
                        item_reasons.append(
                            f"slot {slot.slot_idx} geometry status is {slot.geometry_status}"
                        )
                    elif slot.status == "ANOMALY_ESCALATED":
                        item_reasons.append(f"slot {slot.slot_idx} exceeded iterative verification limit")
                    if slot.content_status == "OCR_UNCERTAIN":
                        item_reasons.append(f"slot {slot.slot_idx} OCR content is uncertain")
                    separation = (slot.audit or {}).get("ink_separation", {})
                    if (separation and float(separation.get("confidence", 0.0)) < .60):
                        item_reasons.append(
                            f"slot {slot.slot_idx} handwriting separation used "
                            f"low-confidence {separation.get('mode', 'fallback')}"
                        )
                item.quality = {"status": "NEED_REVIEW" if item_reasons else "OK",
                                "reasons": item_reasons, "regions": checks}
                reasons.extend(f"{item.item_id}: {reason}" for reason in item_reasons)
    if any(str(meta.get("status")) != "REGISTERED" for meta in package.registration or []):
        reasons.append("registration confidence below threshold")
    package.quality = {"status": "NEED_REVIEW" if reasons else "OK",
                       "reasons": list(dict.fromkeys(reasons)), "region_count": region_count,
                       "overlap_count": 0}
    return package.quality


__all__ = ["validate_region", "validate_item_regions"]
