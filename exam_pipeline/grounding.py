"""Local OCR grounding for stem, diagrams and student answer evidence."""
from typing import List, Optional, Sequence
from .contracts import ExamItem, ExamPackage, OCRBlock, Page, TriTargetGrounding
from .ocr import OCRService


def _union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)] if boxes else []


class GeometryGrounder:
    def __init__(self, ocr_service=None, feedback_reader=None, policy=None):
        self.ocr = ocr_service or OCRService()
        self.feedback_reader = feedback_reader
        self.policy = policy

    @staticmethod
    def track_handwriting_streak(blocks: Sequence[OCRBlock], max_vertical_gap: float = 65.0):
        ordered = sorted(blocks, key=lambda block: (block.bbox[1], block.bbox[0]))
        if not ordered:
            return []
        streak = [ordered[0]]
        for block in ordered[1:]:
            if block.bbox[1] - streak[-1].bbox[3] > max_vertical_gap:
                break
            streak.append(block)
        return streak

    def refine_item(self, item: ExamItem, page: Page, engine: str,
                    language: str = "chi_sim+eng", is_student: bool = False):
        # 优先使用学生独立检测的区域
        regions = item.student_regions if is_student and item.student_regions else item.answer_regions
        regions = [region for region in regions if region.page_index == page.index]
        if not regions:
            return item

        # 如果有多个区域,优先选择独立检测的区域(通过ocr_text标记识别)
        independent_regions = [r for r in regions if r.ocr_text == "student_independent_detection"]
        region = independent_regions[0] if independent_regions else regions[0]
        kind = str(item.item_type or "").lower()
        default_padding = 28 if any(
            token in kind for token in ("choice", "single", "选择", "判断")
        ) else 35
        padding = int(self.policy.parameter(
            "local_ocr_choice_padding_px" if default_padding == 28
            else "local_ocr_padding_px", default_padding
        )) if self.policy else default_padding
        blocks = self.ocr.recognize_crop(page.path, region.bbox, padding=padding,
                                         engine=engine, language=language)
        streak = self.track_handwriting_streak(blocks)
        answer_boxes = [list(block.bbox) for block in streak]
        answer_box = _union(answer_boxes) or list(region.bbox)
        if self.feedback_reader:
            profile = self.feedback_reader.profile(item.item_id)
            if profile:
                answer_box = self.feedback_reader.apply(answer_box, profile)
        stem_box = list(item.stem_region.bbox) if item.stem_region else list(region.bbox)
        item.tri_target = TriTargetGrounding(
            stem_box=stem_box,
            diagram_boxes=[list(diagram.bbox) for diagram in item.diagrams if diagram.bbox],
            answer_box=answer_box,
            answer_boxes=answer_boxes or [answer_box],
            audit={"status": "LOCAL_OCR_REFINED" if streak else "NEED_REVIEW",
                   "crop_padding": padding, "local_block_count": len(blocks),
                   "knowledge_rules": (
                       self.policy.rule_ids("grounding") if self.policy else []
                   )},
        )
        if is_student and streak:
            raw_text = " ".join(block.text for block in streak).strip()

            # Apply answer_parser to clean and extract the actual answer
            from .answer_parser import parse_student_answer

            # Get expected answer and question for context
            expected = None
            question = item.question_text or ''
            item_type = str(item.item_type or '').lower()
            slot_type = str(getattr(item, 'slot_type', '')).lower()

            # Parse the answer
            answer, audit = parse_student_answer(
                recognized_text=raw_text,
                expected_text=expected,
                slot_type=slot_type,
                item_type=item_type,
                question_text=question
            )

            item.student_answer = None  # finalized exclusively from verified slots
            item.student_answer_audit = audit

            confidences = [block.confidence for block in streak if block.confidence is not None]
            if confidences:
                item.confidence = min(confidences)
        return item

    def enrich_package(self, package: ExamPackage, pages: Sequence[Page], engine: str,
                       language: str = "chi_sim+eng"):
        page_map = {page.index: page for page in pages}
        refined = 0
        for section in package.sections:
            for question in section.questions:
                for item in question.items:
                    indexes = {region.page_index for region in (item.student_regions or item.answer_regions)}
                    for index in sorted(indexes):
                        page = page_map.get(index)
                        if page:
                            self.refine_item(item, page, engine, language,
                                             is_student=package.document_type == "student")
                            refined += 1
        return {"refined_items": refined}


__all__ = ["GeometryGrounder"]
