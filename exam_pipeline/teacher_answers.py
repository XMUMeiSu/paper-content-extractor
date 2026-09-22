"""Teacher answer extraction from single-page or optional reference masks."""
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .contracts import ExamPackage, Page
from .ocr import OCRService
from .teacher_answer_quality import TeacherAnswerQualityAssessor


class TeacherAnswerExtractionService:
    def __init__(self, ocr_service=None, policy=None, formula_client=None):
        self.ocr = ocr_service or OCRService()
        self.policy = policy
        from .answer_recognition import AnswerRecognizer
        self.recognizer = AnswerRecognizer(self.ocr, formula_client)

    def extract(self, package: ExamPackage, pages: Sequence[Page], reference_pages: Sequence[Page] = (),
                engine: str = "paddle", language: str = "chi_sim+eng",
                reference_confidence: Optional[float] = None,
                debug_dir: Optional[Path] = None,
                reference_kind: str = "single_page",
                use_exam_tree_fallback: bool = False) -> Dict[str, Any]:
        """Transcribe image evidence independently; never trust stored reference text.

        ``use_exam_tree_fallback`` is retained for call compatibility only.
        Failed image recognition remains unresolved regardless of its value.
        """
        from .result_contract import valid_box

        page_by_id = {page.index: page for page in pages}
        summary = {"total_slots": 0, "answers_extracted": 0, "missing_answers": 0,
                   "exam_tree_fallback_used": 0, "image_extracted": 0,
                   "reference_kind": reference_kind, "reference_confidence": reference_confidence}
        pending = []
        all_items = []

        for section in package.sections:
            for question in section.questions:
                for item in question.items:
                    all_items.append(item)

                    for slot in item.slots:
                        summary["total_slots"] += 1
                        page = page_by_id.get(slot.page_index)
                        if not slot.expected_bbox and slot.audit.get('topology_source') == 'missing_placeholder':
                            slot.expected_text = None
                            slot.status = 'TEACHER_ANSWER_NEEDS_REVIEW'
                            slot.geometry_status = 'MISSING'
                            slot.content_status = 'NOT_EVALUATED'
                            slot.errors.append({'code':'SLOT_CANDIDATE_MISSING','retryable':False})
                            continue
                        if page is not None and not valid_box(slot.expected_bbox, page, 'yxyx'):
                            slot.expected_text = None
                            slot.status = "TEACHER_ANSWER_NEEDS_REVIEW"
                            slot.geometry_status = "FAILED"
                            slot.content_status = "NOT_EVALUATED"
                            slot.review_status = "NEED_REVIEW"
                            slot.errors.append({"code": "INVALID_EXPECTED_BOX", "retryable": False})
                            continue
                        if page is None:
                            slot.expected_text = None
                            slot.geometry_status = "MISSING"
                            slot.content_status = "NOT_EVALUATED"
                            slot.status = "TEACHER_ANSWER_NEEDS_REVIEW"
                            slot.errors.append({"code": "PAGE_UNAVAILABLE", "retryable": False})
                            slot.audit["teacher_answer_extraction"] = {
                                "status": "IMAGE_UNAVAILABLE", "reference_kind": reference_kind,
                                "fallback_eligible": False,
                            }
                            continue
                        # Slot coordinates have already been grounded to OCR on
                        # this teacher page. Do not run a second pixel/ink gate
                        # that can shrink or erase short answers and formulas.
                        slot.handwriting_bbox = list(slot.expected_bbox)
                        slot.evidence_bbox = list(slot.expected_bbox)
                        slot.recognition_bbox = list(slot.expected_bbox)
                        coordinate_warnings = list(
                            (slot.audit.get("local_validation") or {}).get("warnings") or [])
                        geometry_evidence = {
                            "status": ("ALIGNED_WITH_WARNING" if coordinate_warnings else "ALIGNED"),
                            "reason": "OCR_COORDINATE_ACCEPTED",
                            "coordinate_authority": "ocr_boxes_only",
                            "warnings": coordinate_warnings,
                        }
                        slot.geometry_evidence = geometry_evidence
                        slot.audit["geometry_evidence"] = geometry_evidence
                        slot.geometry_status = geometry_evidence["status"]
                        slot.audit["geometry_mode"] = "ocr_coordinates_only"
                        pending.append({
                            "key": id(slot),
                            "slot_id": ((slot.semantic_id or f"{item.item_id}:slot:{slot.slot_idx}")
                                        + f":page:{slot.page_index}"),
                            "page": page,
                            "box": slot.recognition_bbox,
                            "question": item.question_text,
                            "kind": item.item_type,
                            "slot": slot,
                            "item": item,
                        })

        if hasattr(self.recognizer, "recognize_many"):
            recognitions = self.recognizer.recognize_many(
                pending, engine=engine, language=language)
        else:
            recognitions = {
                entry["key"]: self.recognizer.recognize(
                    entry["page"], entry["box"], entry["question"], entry["kind"],
                    None, engine, language)
                for entry in pending
            }

        for entry in pending:
            slot = entry["slot"]
            item = entry["item"]
            recognition = recognitions[entry["key"]]
            text = recognition["text"]
            confidence = recognition.get("confidence")
            slot.has_ink = bool(str(text or "").strip())
            slot.answer_fragments = recognition["fragments"]
            slot.audit["recognition"] = recognition
            item_type = self._infer_item_type(item)
            quality_assessment = TeacherAnswerQualityAssessor.assess_answer(
                text, item_type, confidence)
            auto_accept = recognition["status"] == "RECOGNIZED" and (
                item_type != "choice" or not quality_assessment["needs_review"]
            ) and slot.geometry_status in {"ALIGNED", "ALIGNED_WITH_WARNING"}
            final_text = text.strip()
            slot.content_status = "RECOGNIZED" if auto_accept else "OCR_UNCERTAIN"
            slot.review_status = ("AUTO_PASS" if auto_accept else "NEED_REVIEW")
            if auto_accept:
                slot.expected_text = final_text
                slot.recognized_text = final_text
                slot.status = "TEACHER_ANSWER_EXTRACTED"
                summary["answers_extracted"] += 1
                summary["image_extracted"] += 1
            else:
                slot.expected_text = None
                slot.recognized_text = final_text
                slot.status = "TEACHER_ANSWER_NEEDS_REVIEW"
            slot.audit["teacher_answer_extraction"] = {
                "status": slot.status, "ocr_confidence": confidence,
                "answer_source": "vlm_primary_ocr_coordinates",
                "candidate_text": text,
                "corrected_text": final_text,
                "auto_corrected": quality_assessment['auto_corrected'],
                "quality_score": quality_assessment['confidence'],
                "quality_level": quality_assessment['quality'],
                "quality_issues": quality_assessment['issues'],
                "recognition_warnings": recognition.get("warnings", []),
                "agreement_type": recognition.get("agreement_type", "NONE"),
                "choice_evidence": recognition.get("choice_evidence"),
                "formula_checks": recognition.get("formula_checks"),
                "acceptance": "quality_approved" if auto_accept else "requires_review",
                "fallback_eligible": False,
            }

        for item in all_items:
            values = [slot.expected_text if slot.status == "TEACHER_ANSWER_EXTRACTED" else None
                      for slot in sorted(item.slots, key=lambda slot: slot.slot_idx)]
            item.standard_answer = values[0] if len(values) == 1 else values or None
        summary["missing_answers"] = summary["total_slots"] - summary["answers_extracted"]
        summary["answer_batch"] = dict(getattr(self.recognizer, "batch_metrics", {}))
        return summary

    def _infer_item_type(self, item) -> str:
        """推断题目类型用于质量评估。"""
        kind = str(item.item_type or "").lower()
        return "choice" if kind in {"choice", "single", "multiple", "mcq"} else (
            "fill_blank" if kind in {"fill", "blank", "cloze"} else "short_answer")



__all__ = ["TeacherAnswerExtractionService"]
