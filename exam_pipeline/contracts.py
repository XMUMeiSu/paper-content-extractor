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
    coordinate_format: str = "xyxy"


@dataclass
class Page:
    index: int
    path: str
    width: Optional[int]
    height: Optional[int]
    ocr: List[OCRBlock]
    # Identity is assigned at ingestion and must survive every stage.  The
    # defaults keep legacy constructors/source fixtures valid.
    document_id: str = ""
    physical_page_id: str = ""
    file_fingerprint: str = ""
    # Fingerprint of the exact image from which ``ocr`` was produced. It may
    # differ from ``file_fingerprint`` only while a transformed page is waiting
    # for fresh OCR; such coordinates must not be used for grounding.
    ocr_source_fingerprint: str = ""
    page_index: Optional[int] = None
    schema_status: str = "VALID"


@dataclass
class PageRegion:
    page_index: int
    page_file: str
    bbox: List[float]
    confidence: Optional[float] = None
    ocr_text: str = ""
    coordinate_format: str = "xyxy"
    coordinate_role: str = "evidence"


@dataclass
class DiagramRef:
    title: str
    image_url: str = ""
    bbox: List[float] = field(default_factory=list)
    audit: Dict[str, Any] = field(default_factory=dict)
    # The diagram may be attached to a different physical page than the
    # first answer slot (for example, a cross-page large question).  Keep the
    # page identity explicit in the public result.
    page_index: int = 1


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
    # Physical ink evidence and the wider crop used for OCR/VLM are kept
    # separate. ``handwriting_bbox`` remains the canonical final evidence box
    # for backwards-compatible consumers.
    evidence_bbox: Optional[List[int]] = None
    recognition_bbox: Optional[List[int]] = None
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
    geometry_evidence: Dict[str, Any] = field(default_factory=dict)
    semantic_id: str = ""
    anchor_before: str = ""
    anchor_after: str = ""
    # Multiple physical fragments may belong to one logical answer.
    answer_fragments: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    # Existing extraction engines historically use yxyx internally.  This
    # marker makes that boundary explicit; serialized contracts are converted
    # to xyxy by ExamPackage.to_dict().
    runtime_bbox_format: str = "yxyx"


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
    semantic_slot_plan: List[Dict[str, Any]] = field(default_factory=list)
    slot_semantics_audit: Dict[str, Any] = field(default_factory=dict)
    answer_status: str = "NOT_EVALUATED"
    answer_parts: List[Dict[str, Any]] = field(default_factory=list)
    # Slot-level completion is reported separately from whole-item completion.
    slot_evaluation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExamQuestion:
    question_id: str
    question_num: int
    question_title: str
    items: List[ExamItem] = field(default_factory=list)
    question_score: Optional[float] = None
    # Question-level figures are shared by composite/sub-item answers.
    diagrams: List[DiagramRef] = field(default_factory=list)


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
    structure_audit: Dict[str, Any] = field(default_factory=dict)
    extraction_status: str = "NOT_EVALUATED"
    extraction_errors: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    schema_status: str = "VALID"
    structure_status: str = "PENDING"
    geometry_status: str = "PENDING"
    content_status: str = "PENDING"
    production_status: str = "PENDING"
    stage_status: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Runtime compatibility is isolated here.  Consumers never need to
        # guess whether a box is yxyx or xyxy.
        for section in data.get("sections", []):
            for question in section.get("questions", []):
                for item in question.get("items", []):
                    for slot in item.get("slots", []):
                        if slot.get("runtime_bbox_format") == "yxyx":
                            for key in ("expected_bbox", "handwriting_bbox",
                                        "evidence_bbox", "recognition_bbox"):
                                box = slot.get(key)
                                if isinstance(box, list) and len(box) == 4:
                                    slot[key] = [box[1], box[0], box[3], box[2]]
                            slot["runtime_bbox_format"] = "xyxy"
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
        # Keep the public package version compatible; the coordinate/status
        # additions are additive and are advertised through contract_revision.
        data["schema_version"] = "exam_package.v5"
        data["contract_revision"] = "6"
        data["pipeline_version"] = "2.12.0"
        data["bbox_format"] = "xyxy"
        data["slot_bbox_format"] = "xyxy"
        data["canonical_canvas"] = {"width": 1654, "height": 2338}
        data["status"] = {
            "schema_status": self.schema_status,
            "structure_status": self.structure_status,
            "geometry_status": self.geometry_status,
            "content_status": self.content_status,
            "production_status": self.production_status,
        }
        return data
