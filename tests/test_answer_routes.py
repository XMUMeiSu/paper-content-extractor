import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from exam_pipeline.answer_parser import parse_student_answer
from exam_pipeline.answer_recognition import AnswerRecognizer, formula_structure_issues, remove_printed_horizontal_rules
from exam_pipeline.contracts import ExamItem, ExamPackage, ExamQuestion, ExamSection, OCRBlock, Page, Slot
from exam_pipeline.result_contract import finalize_answers


class _LowChoiceOCR:
    def recognize_crop(self, *args, **kwargs):
        return [OCRBlock("D", [10, 10, 35, 35], .4)]


class _StableChoiceVision:
    def __init__(self, values=("D", "D")):
        self.values = list(values)

    def analyze_choice(self, prompt, paths):
        value = self.values.pop(0)
        return {"transcription": value, "legible": True, "content_kind": "handwriting"}


class AnswerRouteTests(unittest.TestCase):
    def page(self, tmp, ocr=None):
        path = Path(tmp) / "page.png"
        cv2.imwrite(str(path), np.full((120, 180, 3), 255, np.uint8))
        return Page(1, str(path), 180, 120, ocr or [])

    def test_low_confidence_choice_is_rescued_by_primary_visual_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = AnswerRecognizer(_LowChoiceOCR(), _StableChoiceVision()).recognize(
                self.page(tmp), [10, 10, 80, 100], kind="choice")
        self.assertEqual(result["status"], "RECOGNIZED")
        self.assertEqual(result["text"], "D")
        self.assertIn("LOW_OCR_CONFIDENCE", result["warnings"])
        self.assertEqual(result["choice_evidence"]["visual_passes"], 1)

    def test_choice_conflict_keeps_vlm_result_with_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = AnswerRecognizer(
                type("OCR", (), {"recognize_crop": lambda *a, **k: [OCRBlock("C", [1, 1, 2, 2], .99)]})(),
                _StableChoiceVision(("D", "D")),
            ).recognize(self.page(tmp), [10, 10, 80, 100], kind="choice")
        self.assertEqual(result["status"], "RECOGNIZED")
        self.assertEqual(result["text"], "D")
        self.assertIn("OCR_VLM_DISAGREEMENT", result["warnings"])

    def test_mask_is_ignored_and_low_confidence_ocr_remains_a_warning(self):
        class OCR:
            def recognize_mask(self, *args, **kwargs):
                return [OCRBlock("A", [1, 1, 2, 2], .4)]
            def recognize_crop(self, *args, **kwargs):
                return [OCRBlock("B", [1, 1, 2, 2], .4)]

        with tempfile.TemporaryDirectory() as tmp:
            result = AnswerRecognizer(OCR()).recognize(
                self.page(tmp), [10, 10, 80, 100], kind="choice",
                mask=np.zeros((20, 20), np.uint8))
        self.assertEqual(result["status"], "RECOGNIZED")
        self.assertEqual(result["text"], "B")
        self.assertIn("LOW_OCR_CONFIDENCE", result["warnings"])
        self.assertEqual(len(result["attempts"]), 1)

    def test_choice_marker_is_preserved_without_inventing_a_letter(self):
        answer, audit = parse_student_answer("✓", None, "choice_response_cavity", "choice")
        self.assertEqual(answer, "✓")
        self.assertTrue(audit["marker"])

    def test_formula_structure_checks_signs_scripts_and_fractions(self):
        self.assertEqual(formula_structure_issues("y=-(x+1)^2"), [])
        self.assertIn("MISSING_OPERAND", formula_structure_issues("y=-(x+)^2"))
        self.assertEqual(formula_structure_issues(r"\\frac{1}{2}"), [])
        self.assertIn("MALFORMED_FRACTION", formula_structure_issues(r"\\frac{1}"))

    def test_printed_rule_removal_reports_separation(self):
        image = np.full((50, 200, 3), 255, np.uint8)
        cv2.line(image, (10, 25), (190, 25), (0, 0, 0), 2)
        cleaned, audit = remove_printed_horizontal_rules(image)
        self.assertTrue(audit["applied"])
        self.assertLess(int((cleaned[:, :, 0] < 128).sum()), int((image[:, :, 0] < 128).sum()))

    def test_slot_evaluation_keeps_partial_multi_blank_rate(self):
        slots = [
            Slot(1, "fill", "q", [10, 10, 30, 40], 1, student_answer="1",
                 handwriting_bbox=[10, 10, 30, 40], geometry_status="ALIGNED", content_status="RECOGNIZED"),
            Slot(2, "fill", "q", [10, 50, 30, 80], 1, geometry_status="MISSING", content_status="NOT_EVALUATED"),
            Slot(3, "fill", "q", [10, 90, 30, 120], 1, student_answer="3",
                 handwriting_bbox=[10, 90, 30, 120], geometry_status="ALIGNED", content_status="RECOGNIZED"),
        ]
        item = ExamItem("q", "q", slots=slots, expected_slot_count=3)
        package = ExamPackage("e", "", "math", "student", None, 1, ["page.png"],
                              [ExamSection("s", "", [ExamQuestion("q", 1, "", [item])])])
        with tempfile.TemporaryDirectory() as tmp:
            page = self.page(tmp)
            finalize_answers(package, [page])
        self.assertEqual(item.answer_status, "PARTIAL")
        self.assertEqual(item.slot_evaluation["recognized_slots"], 2)
        self.assertEqual(item.slot_evaluation["unresolved_slots"], 1)
        self.assertAlmostEqual(item.slot_evaluation["completion_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
