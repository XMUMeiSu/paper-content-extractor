"""Teacher answer extraction from single-page or optional reference masks."""
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .contracts import ExamPackage, Page
from .ocr import OCRService
from .verification import UniversalInkSnapper
from .teacher_answer_quality import TeacherAnswerQualityAssessor


class TeacherAnswerExtractionService:
    def __init__(self, ocr_service=None, policy=None):
        self.ocr = ocr_service or OCRService()
        self.policy = policy

    def extract(self, package: ExamPackage, pages: Sequence[Page], reference_pages: Sequence[Page] = (),
                engine: str = "paddle", language: str = "chi_sim+eng",
                reference_confidence: Optional[float] = None,
                debug_dir: Optional[Path] = None,
                reference_kind: str = "single_page",
                use_exam_tree_fallback: bool = True) -> Dict[str, Any]:
        """Extract teacher answers from image or fallback to exam tree standard_answer.

        Args:
            package: ExamPackage containing teacher pages
            pages: Teacher pages to extract from
            reference_pages: Optional reference pages for ink separation
            engine: OCR engine to use
            language: OCR language
            reference_confidence: Reference confidence for ink separation
            debug_dir: Optional directory for debug outputs
            reference_kind: Type of reference (single_page/cohort)
            use_exam_tree_fallback: If True, use item.standard_answer from exam tree
                                   when image extraction fails (default: True)

        Returns:
            Summary dict with extraction statistics including:
            - total_slots: Total number of answer slots
            - answers_extracted: Successfully extracted answers
            - missing_answers: Slots without answers
            - exam_tree_fallback_used: Answers from exam tree fallback
            - image_extracted: Answers extracted from images
        """
        import cv2

        images = {page.index: cv2.imread(page.path) for page in pages}
        references = {page.index: cv2.imread(page.path) for page in reference_pages}
        summary = {"total_slots": 0, "answers_extracted": 0, "missing_answers": 0,
                   "exam_tree_fallback_used": 0, "image_extracted": 0,
                   "reference_kind": reference_kind, "reference_confidence": reference_confidence}

        for section in package.sections:
            for question in section.questions:
                for item in question.items:
                    recognized_parts = []
                    failed_slots = []  # Track failed extractions for fallback

                    # Priority: Use exam tree standard_answer if available
                    if use_exam_tree_fallback and item.standard_answer:
                        # Parse exam tree standard answer
                        if isinstance(item.standard_answer, list):
                            answers = item.standard_answer
                        else:
                            # Single answer or semicolon-separated answers
                            answer_str = str(item.standard_answer).strip()
                            answers = [a.strip() for a in answer_str.split("；")] if "；" in answer_str else [answer_str]

                        # Apply to all slots directly
                        for slot in item.slots:
                            summary["total_slots"] += 1
                            slot_idx = slot.slot_idx - 1  # Convert to 0-based index
                            if slot_idx < len(answers) and answers[slot_idx]:
                                answer_text = str(answers[slot_idx]).strip()
                                if answer_text:
                                    # Use exam tree answer directly
                                    slot.expected_text = answer_text
                                    slot.recognized_text = answer_text
                                    slot.status = "EXAM_TREE_STANDARD_ANSWER"
                                    recognized_parts.append(answer_text)
                                    summary["answers_extracted"] += 1
                                    summary["exam_tree_fallback_used"] += 1

                                    # Set audit trail
                                    slot.audit["teacher_answer_extraction"] = {
                                        "status": "EXAM_TREE_STANDARD_ANSWER",
                                        "answer_source": "exam_tree_standard_answer",
                                        "answer_text": answer_text,
                                        "fallback_eligible": False,
                                        "image_extraction_skipped": True,
                                    }
                                else:
                                    summary["missing_answers"] += 1
                                    failed_slots.append(slot)
                            else:
                                summary["missing_answers"] += 1
                                failed_slots.append(slot)

                        # Skip image extraction if all slots handled
                        if recognized_parts:
                            item.standard_answer = "；".join(recognized_parts)
                        continue

                    # Fallback: Image-based extraction when no exam tree answer
                    for slot in item.slots:
                        summary["total_slots"] += 1
                        image, reference = images.get(slot.page_index), references.get(slot.page_index)
                        if image is None:
                            failed_slots.append(slot)
                            slot.audit["teacher_answer_extraction"] = {
                                "status": "IMAGE_UNAVAILABLE", "reference_kind": reference_kind,
                                "fallback_eligible": True,
                            }
                            continue
                        detection = UniversalInkSnapper.extract(
                            image, slot.expected_bbox, reference, self.policy,
                            reference_kind, reference_confidence, True,
                        )
                        mask = detection.get("_handwriting_mask")
                        if mask is None:
                            mask = detection.get("mask")
                        slot.handwriting_bbox = detection.get("bbox")
                        slot.has_ink = bool(detection.get("gate", {}).get("has_ink"))
                        blocks = self.ocr.recognize_mask(mask, engine, language) if (
                            detection.get("gate", {}).get("has_ink") and engine != "none"
                        ) else []
                        text = "".join(
                            block.text for block in sorted(blocks, key=lambda block: (block.bbox[1], block.bbox[0]))
                        ).strip()
                        confidence = min(
                            (block.confidence for block in blocks if block.confidence is not None),
                            default=None,
                        )
                        separation = detection.get("separation", {})
                        metrics = separation.get("metrics", {})
                        student_ink = max(1, int(metrics.get("student_ink_pixels", 0)))
                        print_coverage = float(metrics.get("printed_pixels", 0)) / student_ink
                        colored = separation.get("mode") == "teacher_colored_ink"
                        # Long black residual text is unsafe without semantic
                        # confirmation: registration residue can look exactly
                        # like printed Chinese. Prefer recall loss over poisoning
                        # the Golden shared by every student.
                        automatically_accepted = bool(text) and (
                            colored or (
                                len(text) <= 12
                                and float(separation.get("confidence", 0.0)) >= 0.60
                                and (reference is None or print_coverage >= 0.20)
                            )
                        )
                        # Quality assessment for automatic acceptance
                        item_type = self._infer_item_type(item)
                        quality_assessment = TeacherAnswerQualityAssessor.assess_answer(
                            text, item_type, confidence
                        )

                        # Auto-accept if high quality OR original auto-accept criteria met
                        auto_accept = (
                            automatically_accepted or
                            (quality_assessment['quality'] == 'high' and
                             not quality_assessment['needs_review'])
                        )

                        final_text = quality_assessment['corrected_text']

                        if auto_accept:
                            slot.expected_text = final_text
                            slot.recognized_text = final_text
                            slot.status = "TEACHER_ANSWER_EXTRACTED"
                            recognized_parts.append(final_text)
                            summary["answers_extracted"] += 1
                            summary["image_extracted"] += 1
                        else:
                            # Mark for potential fallback
                            failed_slots.append(slot)
                            slot.expected_text = final_text  # Still use corrected text
                            slot.recognized_text = final_text
                            slot.status = "TEACHER_ANSWER_NEEDS_REVIEW"
                        slot.audit["ink_separation"] = separation
                        slot.audit["teacher_answer_extraction"] = {
                            "status": slot.status, "ocr_confidence": confidence,
                            "answer_source": f"{reference_kind}_handwriting_mask",
                            "candidate_text": text,
                            "corrected_text": final_text,
                            "auto_corrected": quality_assessment['auto_corrected'],
                            "quality_score": quality_assessment['confidence'],
                            "quality_level": quality_assessment['quality'],
                            "quality_issues": quality_assessment['issues'],
                            "print_coverage": round(print_coverage, 4),
                            "acceptance": "colored_ink" if colored else (
                                "quality_approved" if auto_accept
                                else "requires_review"
                            ),
                            "fallback_eligible": not auto_accept,
                        }
                        if debug_dir and mask is not None:
                            path = (Path(debug_dir) / f"page_{slot.page_index:02d}"
                                    / f"{item.item_id}__slot_{slot.slot_idx}.png")
                            path.parent.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(path), mask)
                            slot.audit["teacher_answer_extraction"]["handwriting_mask_path"] = str(path)

                    # Update item standard answer if any parts recognized from image
                    if recognized_parts:
                        item.standard_answer = "；".join(recognized_parts)
                    else:
                        summary["missing_answers"] += len(failed_slots)
        return summary

    def _infer_item_type(self, item) -> str:
        """推断题目类型用于质量评估。"""
        item_id = item.item_id.lower()

        # 选择题判断
        if 'choice' in item_id or any(item_id.startswith(f'q{i}') for i in range(1, 15)):
            # 检查是否有选项标记
            if hasattr(item, 'options') and item.options:
                return 'choice'
            # 根据槽位数量判断
            if len(item.slots) == 1:
                return 'choice'

        # 填空题判断
        if 'blank' in item_id or len(item.slots) > 1:
            return 'fill_blank'

        # 默认为简答题
        return 'short_answer'


__all__ = ["TeacherAnswerExtractionService"]
