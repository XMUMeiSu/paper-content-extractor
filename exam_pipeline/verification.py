"""OCR-coordinate verification and independent answer transcription."""
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .answer_parser import parse_student_answer
from .contracts import ExamPackage, Page, Slot
from .ocr import OCRService


class OCRCoordinateVerificationController:
    """Accept a valid OCR-grounded box without pixel or ink refinement."""

    def __init__(self, policy=None):
        self.policy = policy

    def verify(self, image, slot: Slot, reference_image=None,
               recognize: Optional[Callable[[Sequence[int]], str]] = None,
               observed_text: str = "", reference_kind: str = "teacher",
               reference_confidence: Optional[float] = None):
        del image, reference_image, recognize, observed_text, reference_kind, reference_confidence
        box = list(slot.expected_bbox)
        local = (slot.audit or {}).get("local_validation") or {}
        warnings = list(local.get("warnings") or [])
        geometry_status = "ALIGNED_WITH_WARNING" if warnings else "ALIGNED"
        return {
            "status": ("COORDINATE_ACCEPTED_WITH_WARNING" if warnings
                       else "COORDINATE_ACCEPTED"),
            "iterations_used": 0,
            "initial_bbox": box,
            "final_bbox": box,
            "detected_bbox": box,
            "expected_text": slot.expected_text,
            "recognized_text": "",
            "has_ink": False,
            "content_observed": None,
            "shrink_rate": "0.0%",
            "history": [],
            "geometry_status": geometry_status,
            "content_status": "NOT_EVALUATED",
            "semantic_status": "NOT_EVALUATED",
            "review_status": "NEED_REVIEW",
            "geometry_evidence": {
                "status": geometry_status,
                "reason": "OCR_COORDINATE_ACCEPTED",
                "coordinate_authority": "ocr_boxes_only",
                "warnings": warnings,
            },
            "evidence_bbox": box,
            "recognition_bbox": box,
            "reason": "" if not warnings else ";".join(warnings),
        }


def _failure_result(slot, geometry_status, reason):
    return {
        "status": "ANOMALY_ESCALATED",
        "iterations_used": 0,
        "initial_bbox": list(slot.expected_bbox),
        "final_bbox": None,
        "expected_text": slot.expected_text,
        "recognized_text": "",
        "has_ink": False,
        "shrink_rate": "0.0%",
        "history": [],
        "geometry_status": geometry_status,
        "content_status": "NOT_EVALUATED",
        "semantic_status": "NOT_EVALUATED",
        "review_status": "NEED_REVIEW",
        "reason": reason,
    }


class SlotVerificationService:
    def __init__(self, controller=None, ocr_service=None, policy=None, formula_client=None):
        self.policy = policy
        from .answer_recognition import AnswerRecognizer
        self.controller = controller or OCRCoordinateVerificationController(policy)
        self.ocr = ocr_service or OCRService()
        self.recognizer = AnswerRecognizer(self.ocr, formula_client)

    def verify_package(self, package: ExamPackage, pages: Sequence[Page], reference_pages=(),
                       engine: str = "paddle", language: str = "chi_sim+eng",
                       reference_context: Optional[Dict[str, Any]] = None,
                       debug_dir: Optional[Path] = None,
                       independent_page: bool = False):
        # Retained for API compatibility. OCR-grounded student coordinates do
        # not depend on teacher references or registration state.
        del reference_pages, reference_context, debug_dir, independent_page
        import cv2
        from .result_contract import valid_box

        page_by_index = {page.index: page for page in pages}
        images = {index: cv2.imread(page.path) for index, page in page_by_index.items()}
        records: Dict[str, List[Dict[str, Any]]] = {}
        counts = {"COORDINATE_ACCEPTED": 0,
                  "COORDINATE_ACCEPTED_WITH_WARNING": 0,
                  "ANOMALY_ESCALATED": 0}
        geometry_counts = {"ALIGNED": 0, "ALIGNED_WITH_WARNING": 0,
                           "UNCERTAIN": 0, "FAILED": 0, "MISSING": 0}
        content_counts = {"RECOGNIZED": 0, "OCR_UNCERTAIN": 0, "BLANK": 0,
                          "NOT_EVALUATED": 0}
        pending = []
        geometry_results = {}

        for section in package.sections:
            for question in section.questions:
                for item in question.items:
                    for slot in item.slots:
                        page = page_by_index.get(slot.page_index)
                        image = images.get(slot.page_index)
                        missing_slot = (slot.audit or {}).get("topology_source") == "missing_placeholder"
                        authority = (slot.audit or {}).get("coordinate_authority")

                        if page is None or image is None:
                            result = _failure_result(slot, "MISSING", "page unavailable")
                            slot.errors.append({"code": "PAGE_UNAVAILABLE", "retryable": False})
                        elif missing_slot:
                            result = _failure_result(
                                slot, "MISSING", "expected slot was not found on the student page")
                        elif authority != "ocr_boxes_only":
                            result = _failure_result(
                                slot, "FAILED", "coordinate source is not current-page OCR")
                            slot.errors.append({"code": "NON_OCR_COORDINATE_SOURCE",
                                                "retryable": False})
                        elif not valid_box(slot.expected_bbox, page, "yxyx"):
                            result = _failure_result(slot, "FAILED", "invalid OCR coordinate box")
                            slot.errors.append({"code": "INVALID_EXPECTED_BOX", "retryable": False})
                        else:
                            result = self.controller.verify(image, slot)

                        slot.handwriting_bbox = result["final_bbox"]
                        slot.recognized_text = result["recognized_text"]
                        slot.evidence_bbox = result.get("evidence_bbox") or result.get("detected_bbox")
                        slot.recognition_bbox = result.get("recognition_bbox") or slot.handwriting_bbox
                        slot.has_ink = result["has_ink"]
                        slot.status = result["status"]
                        slot.geometry_status = result.get("geometry_status", "UNCERTAIN")
                        slot.content_status = result.get("content_status", "NOT_EVALUATED")
                        slot.review_status = result.get("review_status", "NEED_REVIEW")
                        slot.iterations_used = result["iterations_used"]
                        slot.shrink_rate = result["shrink_rate"]
                        slot.history = result["history"]
                        slot.geometry_evidence = result.get("geometry_evidence", {})
                        slot.audit["geometry_evidence"] = slot.geometry_evidence
                        slot.audit["geometry_mode"] = "ocr_coordinates_only"
                        geometry_results[id(slot)] = result
                        if slot.handwriting_bbox:
                            pending.append({
                                "key": id(slot),
                                "slot_id": ((slot.semantic_id or
                                             f"{item.item_id}:slot:{slot.slot_idx}")
                                            + f":page:{slot.page_index}"),
                                "page": page,
                                "box": slot.recognition_bbox or slot.handwriting_bbox,
                                "question": item.question_text,
                                "kind": item.item_type,
                                "slot": slot,
                                "item": item,
                                "question_id": question.question_id,
                                "result": result,
                            })

        if hasattr(self.recognizer, "recognize_many"):
            recognitions = self.recognizer.recognize_many(
                pending, engine=engine, language=language)
        else:
            recognitions = {
                entry["key"]: self.recognizer.recognize(
                    entry["page"], entry["box"], entry["question"],
                    entry["kind"], None, engine, language)
                for entry in pending
            }

        for entry in pending:
            slot = entry["slot"]
            item = entry["item"]
            result = entry["result"]
            recognition = recognitions[entry["key"]]
            slot.recognized_text = recognition["text"]
            slot.has_ink = bool(str(slot.recognized_text or "").strip())
            slot.content_status = ("RECOGNIZED" if recognition["status"] == "RECOGNIZED"
                                   else "OCR_UNCERTAIN")
            slot.answer_fragments = recognition["fragments"]
            slot.audit["recognition"] = recognition
            slot.review_status = (
                "AUTO_PASS" if slot.geometry_status in {
                    "ALIGNED", "ALIGNED_WITH_WARNING"
                } and slot.content_status == "RECOGNIZED" else "NEED_REVIEW")
            result.update(recognized_text=slot.recognized_text,
                          content_status=slot.content_status,
                          review_status=slot.review_status,
                          recognition=recognition,
                          semantic_status="NOT_EVALUATED")

            if slot.recognized_text:
                clean_answer, parse_audit = parse_student_answer(
                    recognized_text=slot.recognized_text,
                    expected_text=None,
                    slot_type=slot.slot_type,
                    item_type=item.item_type,
                    question_text=item.question_text or "")
                slot.student_answer = (
                    clean_answer if slot.content_status == "RECOGNIZED" else None)
                if clean_answer is None:
                    slot.content_status = "OCR_UNCERTAIN"
                    slot.review_status = "NEED_REVIEW"
                    result.update(content_status=slot.content_status,
                                  review_status=slot.review_status)
                slot.audit["answer_parsing"] = parse_audit
            else:
                slot.student_answer = None

        # Persist every geometry result, including slots that never reached
        # content recognition, after the page batches have completed.
        pending_by_slot = {entry["key"]: entry for entry in pending}
        for section in package.sections:
            for question in section.questions:
                for item in question.items:
                    for slot in item.slots:
                        entry = pending_by_slot.get(id(slot))
                        if entry is not None:
                            result = entry["result"]
                        elif id(slot) in geometry_results:
                            result = geometry_results[id(slot)]
                            slot.student_answer = None
                        else:
                            # A geometry-only accepted slot with no crop is a
                            # contract anomaly and must remain unresolved.
                            result = _failure_result(slot, slot.geometry_status, "answer crop unavailable")
                            slot.student_answer = None

                        record = {"question_id": question.question_id,
                                  "item_id": item.item_id,
                                  "slot_index": slot.slot_idx, **result}
                        records.setdefault(
                            f"{package.exam_id}:page_{slot.page_index:02d}", []).append(record)
                        counts[result["status"]] = counts.get(result["status"], 0) + 1
                        geometry_counts[slot.geometry_status] = (
                            geometry_counts.get(slot.geometry_status, 0) + 1)
                        content_counts[slot.content_status] = (
                            content_counts.get(slot.content_status, 0) + 1)

        summary = {
            "total_audit_slots": sum(counts.values()),
            "coordinates_accepted": counts["COORDINATE_ACCEPTED"],
            "coordinates_accepted_with_warning": counts[
                "COORDINATE_ACCEPTED_WITH_WARNING"],
            "anomaly_escalated": counts["ANOMALY_ESCALATED"],
            "geometry": geometry_counts,
            "content": content_counts,
            "answer_batch": dict(getattr(self.recognizer, "batch_metrics", {})),
            "knowledge_base": self.policy.audit() if self.policy else None,
        }
        return {"summary": summary, "pages": records, "bbox_format": "yxyx",
                "canonical_canvas": {"width": 1654, "height": 2338}}


__all__ = ["OCRCoordinateVerificationController", "SlotVerificationService"]
