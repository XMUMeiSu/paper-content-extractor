"""Production exam structuring pipeline."""
from .contracts import (DiagramRef, ExamItem, ExamPackage, ExamQuestion, ExamSection,
                        OCRBlock, Page, PageRegion, RoIPatchRef, Slot, TriTargetGrounding)
from .config import PipelineSettings
from .feedback import GeometryFeedbackProfile, HITLFeedbackReader
from .ink_separation import NoBlankInkSeparator
from .cohort_consensus import CohortConsensusBuilder
from .golden import GoldenTemplateService
from .exam_tree import ExamTreeService
from .grounding import GeometryGrounder
from .hitl_export import HITLExporter
from .quality import validate_item_regions, validate_region
from .registration import register_page, register_pages
from .roi import RoIPatchGenerator
from .scoring import balance_score_tree, grade_objective_items
from .slots import MultiSlotTopologyEngine
from .subitems import FineGrainedItemSplitter
from .verification import (BlankInkGate, IterativeVerificationController,
                           SlotVerificationService, UniversalInkSnapper)
from .slot_vlm import SlotCoordinateVisionVerifier
from .teacher_answers import TeacherAnswerExtractionService
from .tree_llm import ExamTreeLLMClient
from .cardinality import SlotCardinalityConsensusService
from .vlm import VLMService

__all__ = [
    "OCRBlock", "Page", "PageRegion", "DiagramRef", "TriTargetGrounding", "Slot",
    "RoIPatchRef", "ExamItem", "ExamQuestion", "ExamSection", "ExamPackage",
    "PipelineSettings", "GoldenTemplateService", "ExamTreeService", "GeometryGrounder", "HITLExporter",
    "validate_item_regions", "validate_region", "register_page", "register_pages",
    "RoIPatchGenerator", "balance_score_tree", "grade_objective_items",
    "MultiSlotTopologyEngine", "FineGrainedItemSplitter", "BlankInkGate",
    "UniversalInkSnapper", "IterativeVerificationController", "SlotVerificationService",
    "SlotCoordinateVisionVerifier",
    "VLMService", "GeometryFeedbackProfile", "HITLFeedbackReader",
    "NoBlankInkSeparator", "CohortConsensusBuilder", "TeacherAnswerExtractionService",
    "ExamTreeLLMClient", "SlotCardinalityConsensusService",
]
