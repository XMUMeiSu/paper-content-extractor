"""VLM-only exam extraction pipeline."""

from .config import PipelineSettings
from .contracts import (
    DiagramRef, ExamItem, ExamPackage, ExamQuestion, ExamSection, OCRBlock, Page,
    PageRegion, RoIPatchRef, Slot,
)
from .diagram_assets import DiagramAssetService
from .exam_tree import ExamTreeService
from .golden import GoldenTemplateService
from .performance import PerformanceCollector
from .quality import validate_item_regions, validate_region
from .roi import RoIPatchGenerator
from .visual_extraction import VisualExamExtractionService

__all__ = [
    "DiagramRef", "ExamItem", "ExamPackage", "ExamQuestion", "ExamSection",
    "OCRBlock", "Page", "PageRegion", "RoIPatchRef", "Slot",
    "PipelineSettings", "DiagramAssetService",
    "ExamTreeService", "GoldenTemplateService", "PerformanceCollector",
    "validate_item_regions", "validate_region", "RoIPatchGenerator",
    "VisualExamExtractionService",
]
