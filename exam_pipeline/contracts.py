"""Canonical data contracts for the exam extraction pipeline."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class OCRBlock:
    text: str
    bbox: List[float]
    confidence: Optional[float] = None


@dataclass
class Page:
    index: int
    path: str
    width: Optional[int]
    height: Optional[int]
    ocr: List[OCRBlock]


@dataclass
class PageRegion:
    page_index: int
    page_file: str
    bbox: List[float]
    confidence: Optional[float] = None
    ocr_text: str = ""


@dataclass
class DiagramRef:
    title: str
    image_url: str = ""
    bbox: List[float] = field(default_factory=list)
    audit: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TriTargetGrounding:
    stem_box: List[float] = field(default_factory=list)
    diagram_boxes: List[List[float]] = field(default_factory=list)
    answer_box: List[float] = field(default_factory=list)
    answer_boxes: List[List[float]] = field(default_factory=list)
    audit: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Slot:
    slot_idx: int
    slot_type: str
    parent_item_id: str
    expected_bbox: List[int]
    page_index: int = 1
    expected_text: Optional[str] = None
    handwriting_bbox: Optional[List[int]] = None
    recognized_text: str = ""
    student_answer: Optional[str] = None
    has_ink: bool = False
    status: str = "PENDING"
    iterations_used: int = 0
    shrink_rate: str = "0.0%"
    history: List[Dict[str, Any]] = field(default_factory=list)
    audit: Dict[str, Any] = field(default_factory=dict)
    # Keep the legacy ``status`` for workbench compatibility, while exposing
    # independent production decisions for localization, OCR and review.
    geometry_status: str = "PENDING"
    content_status: str = "PENDING"
    review_status: str = "PENDING"


@dataclass
class RoIPatchRef:
    bbox: List[int]
    image_path: str = ""
    padding: int = 35


@dataclass
class ExamItem:
    item_id: str
    item_name: str
    question_text: str = ""
    standard_answer: Any = None
    item_score: Optional[float] = None
    student_answer: Any = None
    student_score: Optional[float] = None
    answer_regions: List[PageRegion] = field(default_factory=list)
    student_regions: List[PageRegion] = field(default_factory=list)
    is_cross_page: bool = False
    diagrams: List[DiagramRef] = field(default_factory=list)
    tri_target: Optional[TriTargetGrounding] = None
    eval_status: str = "pending"
    eval_feedback: str = ""
    confidence: float = 1.0
    item_type: str = "other"
    stem_region: Optional[PageRegion] = None
    option_regions: List[PageRegion] = field(default_factory=list)
    blank_regions: List[PageRegion] = field(default_factory=list)
    writing_regions: List[PageRegion] = field(default_factory=list)
    quality: Dict[str, Any] = field(default_factory=dict)
    rubric: Optional[str] = None
    is_correct: Optional[bool] = None
    slots: List[Slot] = field(default_factory=list)
    roi_patch: Optional[RoIPatchRef] = None
    cross_page_status: str = "COMPLETE"
    # Semantic answer-point cardinality.  It constrains topology only; student
    # slot coordinates are still discovered on the student's own page.
    expected_slot_count: Optional[int] = None
    slot_count_source: str = ""
    # Independent cardinality evidence used by the production ExamTree gate.
    # It deliberately travels with the logical item, while concrete student
    # coordinates remain local to each paper.
    cardinality_evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExamQuestion:
    question_id: str
    question_num: int
    question_title: str
    items: List[ExamItem] = field(default_factory=list)
    question_score: Optional[float] = None


@dataclass
class ExamSection:
    section_id: str
    section_title: str
    questions: List[ExamQuestion] = field(default_factory=list)
    section_score: Optional[float] = None


@dataclass
class ExamPackage:
    exam_id: str
    exam_title: str
    subject: str
    document_type: str
    student_id: Optional[str]
    total_pages: int
    page_files: List[str]
    sections: List[ExamSection]
    total_score: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    golden_source: str = ""
    topology_locked: bool = False
    registration: List[Dict[str, Any]] = field(default_factory=list)
    quality: Dict[str, Any] = field(default_factory=dict)
    declared_total_score: Optional[float] = None
    score_audit: Dict[str, Any] = field(default_factory=dict)
    grounding_knowledge: Dict[str, Any] = field(default_factory=dict)
    exam_tree_id: str = ""
    exam_tree_revision: int = 0
    exam_tree_fingerprint: str = ""
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["metadata"] = {
            "exam_id": self.exam_id,
            "exam_title": self.exam_title,
            "subject": self.subject,
            "total_score": self.total_score if self.total_score is not None else 0.0,
            "student_total_score": sum(
                (item.student_score or 0.0)
                for section in self.sections
                for question in section.questions
                for item in question.items
            ) or None,
            "total_pages": self.total_pages,
            "is_teacher_golden": self.document_type == "teacher",
            "created_at": self.created_at,
            "golden_source": self.golden_source,
            "topology_locked": self.topology_locked,
            "grounding_knowledge": self.grounding_knowledge,
            "exam_tree_id": self.exam_tree_id,
            "exam_tree_revision": self.exam_tree_revision,
            "exam_tree_fingerprint": self.exam_tree_fingerprint,
        }
        data["schema_version"] = "exam_package.v4"
        data["pipeline_version"] = "2.7.0"
        data["bbox_format"] = "xyxy"
        data["slot_bbox_format"] = "yxyx"
        data["canonical_canvas"] = {"width": 1654, "height": 2338}
        return data
