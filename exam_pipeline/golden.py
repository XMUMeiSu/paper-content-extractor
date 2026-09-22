"""Teacher topology persistence and VLM-only student inheritance."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Sequence

from .contracts import ExamPackage, Page
from .io_utils import atomic_write_json


class GoldenTemplateService:
    """Reuse teacher-owned identities while clearing document-local evidence."""

    def inherit_student_topology(
            self, teacher: ExamPackage, candidate: ExamPackage,
            teacher_pages: Sequence[Page], student_pages: Sequence[Page],
            exam_id: str, student_id: str) -> ExamPackage:
        del teacher_pages
        result = copy.deepcopy(teacher)
        result.exam_id = exam_id
        result.document_type = "student"
        result.student_id = student_id
        result.page_files = [page.path for page in student_pages]
        result.total_pages = len(student_pages)
        result.golden_source = teacher.golden_source or f"teacher:{teacher.exam_id}"
        result.topology_locked = teacher.topology_locked
        result.quality = {}
        result.score_audit = {}

        for section in result.sections:
            for question in section.questions:
                question.diagrams = []
                for item in question.items:
                    item.student_answer = None
                    item.student_score = None
                    item.is_correct = None
                    item.eval_status = "pending"
                    item.eval_feedback = ""
                    item.stem_region = None
                    item.answer_regions = []
                    item.student_regions = []
                    item.option_regions = []
                    item.blank_regions = []
                    item.writing_regions = []
                    item.diagrams = []
                    item.roi_patch = None
                    item.slot_semantics_audit = {}
                    item.cardinality_evidence = {}
                    for key in (
                            "automatic_relocalization", "localization_retry_feedback",
                            "student_anchor_conflict"):
                        item.quality.pop(key, None)
                    for slot in item.slots:
                        slot.student_answer = None
                        slot.recognized_text = ""
                        slot.answer_fragments = []
                        slot.expected_bbox = []
                        slot.handwriting_bbox = None
                        slot.evidence_bbox = None
                        slot.recognition_bbox = None
                        slot.geometry_status = "MISSING"
                        slot.content_status = "MISSING"
                        slot.status = "UNRESOLVED"
                        slot.review_status = "NEED_REVIEW"
                        slot.audit = {}

        result.warnings = list(candidate.warnings)
        result.warnings.append(
            f"学生卷继承教师逻辑题目树：{sum(len(s.questions) for s in result.sections)} 道大题；"
            "题目、槽位和答案坐标由学生原页 VLM 独立识别"
        )
        return result

    @staticmethod
    def save_generated(package: ExamPackage, path: Path) -> None:
        atomic_write_json(path, package.to_dict())
