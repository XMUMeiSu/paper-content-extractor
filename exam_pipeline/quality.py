"""Deterministic geometry quality gates and non-blocking evidence warnings."""
from typing import Any, Dict, Sequence
from .contracts import ExamPackage, Page, PageRegion


def _area(box):
    return max(0., float(box[2])-float(box[0])) * max(0., float(box[3])-float(box[1])) if len(box) >= 4 else 0.


def validate_region(region: PageRegion, page: Page, max_page_ratio: float = .60) -> Dict[str, Any]:
    width, height = float(page.width or 1654), float(page.height or 2338)
    box, reasons = region.bbox, []
    from .result_contract import valid_box
    if not valid_box(box, page):
        reasons.append("invalid or non-finite coordinates")
    if region.page_file != page.path:
        reasons.append("page_file does not match page_index")
    if len(box) < 4 or box[2] <= box[0] or box[3] <= box[1]: reasons.append("invalid bbox")
    ratio = _area(box) / max(1., width*height)
    if ratio > max_page_ratio: reasons.append(f"answer box covers {ratio*100:.1f}% of page")
    if len(box) >= 4 and (box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height):
        reasons.append("bbox out of page bounds")
    return {"status": "NEED_REVIEW" if reasons else "OK", "reasons": reasons, "area_ratio": round(ratio, 4)}


def validate_item_regions(package: ExamPackage, pages: Sequence[Page], overlap_threshold: float = .35):
    page_map = {page.index: page for page in pages}
    reasons, region_count = [], 0
    overlap_count = 0
    occupied = []
    for section in package.sections:
        for question in section.questions:
            for item in question.items:
                item_reasons, item_warnings, checks = [], [], []
                if not item.slots or item.answer_status in {"UNRESOLVED", "PARTIAL"}:
                    item_reasons.append("answer extraction incomplete")
                for region in item.student_regions or item.answer_regions:
                    page = page_map.get(region.page_index) or Page(region.page_index, region.page_file, 1654, 2338, [])
                    check = validate_region(region, page); checks.append(check); region_count += 1
                    item_reasons.extend(check["reasons"])
                if item.confidence is not None and float(item.confidence) < .70:
                    authority = {
                        (slot.audit or {}).get("coordinate_authority")
                        for slot in item.slots
                    }
                    source = ("VLM" if "vlm_original_page_pixels" in authority
                              else "OCR")
                    item_warnings.append(
                        f"{source} confidence below threshold: {float(item.confidence):.2f}"
                    )
                for slot in item.slots:
                    from .roi import yxyx_to_xyxy
                    box = slot.handwriting_bbox
                    if box and len(box) == 4:
                        xy = yxyx_to_xyxy(box)
                        for old_page, owner, old in occupied:
                            if old_page == slot.page_index and owner != item.item_id:
                                inter = max(0,min(xy[2],old[2])-max(xy[0],old[0])) * max(0,min(xy[3],old[3])-max(xy[1],old[1]))
                                if inter / max(1,min(_area(xy),_area(old))) > overlap_threshold:
                                    overlap_count += 1
                                    owner_item = next(
                                        (
                                            candidate
                                            for candidate_section in package.sections
                                            for candidate_question in candidate_section.questions
                                            for candidate in candidate_question.items
                                            if candidate.item_id == owner
                                        ),
                                        None,
                                    )
                                    current_vlm = all(
                                        (slot.audit or {}).get("coordinate_authority")
                                        == "vlm_original_page_pixels"
                                        for slot in item.slots
                                        if slot.expected_bbox
                                    )
                                    owner_vlm = owner_item is not None and all(
                                        (slot.audit or {}).get("coordinate_authority")
                                        == "vlm_original_page_pixels"
                                        for slot in owner_item.slots
                                        if slot.expected_bbox
                                    )
                                    if current_vlm and owner_vlm:
                                        item_warnings.append(
                                            "VLM answer overlaps item " + owner
                                        )
                                    else:
                                        item_reasons.append("answer overlaps item " + owner)
                        occupied.append((slot.page_index,item.item_id,xy))
                    if (package.document_type == "teacher"
                            and (slot.status == "TEACHER_ANSWER_NEEDS_REVIEW"
                                 or not slot.expected_text)):
                        item_reasons.append(
                            f"teacher answer missing for slot {slot.slot_idx}"
                        )
                    if slot.geometry_status == "MISSING":
                        if slot.content_status == "BLANK" and (slot.audit or {}).get("blank_confirmed"):
                            item_warnings.append(
                                f"slot {slot.slot_idx} is blank; no answer pixels to localize"
                            )
                        else:
                            item_reasons.append(
                                f"slot {slot.slot_idx} was not found on the student page"
                            )
                    elif slot.geometry_status in {"FAILED", "UNCERTAIN"}:
                        item_reasons.append(
                            f"slot {slot.slot_idx} geometry status is {slot.geometry_status}"
                        )
                    elif slot.geometry_status == "ALIGNED_WITH_WARNING":
                        item_warnings.append(
                            f"slot {slot.slot_idx} geometry aligned with evidence warning"
                        )
                    elif slot.status == "ANOMALY_ESCALATED":
                        item_reasons.append(f"slot {slot.slot_idx} exceeded iterative verification limit")
                    if slot.content_status == "OCR_UNCERTAIN":
                        item_warnings.append(f"slot {slot.slot_idx} OCR content is uncertain")
                    elif slot.content_status == "VLM_UNCERTAIN":
                        item_warnings.append(f"slot {slot.slot_idx} VLM content is uncertain")
                item.quality = {**item.quality, "status": "NEED_REVIEW" if item_reasons else "OK",
                                "reasons": item_reasons,
                                "warnings": list(dict.fromkeys(
                                    list(item.quality.get("warnings") or []) + item_warnings)),
                                "regions": checks}
                reasons.extend(f"{item.item_id}: {reason}" for reason in item_reasons)
    slots = [slot for section in package.sections for question in section.questions
             for item in question.items for slot in item.slots]
    registration_required = package.document_type == "student" and any(
        (slot.audit or {}).get("coordinate_authority") not in {
            "ocr_boxes_only", "vlm_original_page_pixels",
        }
        and (slot.audit or {}).get("topology_source") != "student_self"
        for slot in slots
    )
    if (registration_required
            and any(str(meta.get("status")) != "REGISTERED"
                    for meta in package.registration or [])):
        reasons.append("registration confidence below threshold")
    if package.structure_audit and package.structure_audit.get("status") != "COMPLETE":
        reasons.append("structure coverage unresolved")
    package.quality = {**package.quality, "status": "NEED_REVIEW" if reasons else "OK",
                       "reasons": list(dict.fromkeys(reasons)), "region_count": region_count,
                       "overlap_count": overlap_count,
                       "registration_required_for_coordinates": registration_required}
    return package.quality


__all__ = ["validate_region", "validate_item_regions"]
