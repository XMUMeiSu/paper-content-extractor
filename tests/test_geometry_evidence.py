import unittest

from exam_pipeline.contracts import ExamItem, ExamPackage, ExamQuestion, ExamSection, Page, Slot
from exam_pipeline.quality import validate_item_regions
from exam_pipeline.result_contract import finalize_answers
from exam_pipeline.verification import OCRCoordinateVerificationController


class GeometryEvidenceTests(unittest.TestCase):
    def package(self, item, role="student"):
        return ExamPackage(
            "exam", "", "math", role, None, 1, ["page.png"],
            [ExamSection("s", "", [ExamQuestion("q", 1, "", [item])])],
        )

    def page(self):
        return Page(1, "page.png", 200, 200, [])

    def test_ocr_grounded_box_is_accepted_without_ink_evidence(self):
        slot = Slot(1, "fill", "q", [10, 10, 110, 190], 1,
                    audit={"local_validation": {"status": "VERIFIED"}})
        result = OCRCoordinateVerificationController().verify(None, slot)
        self.assertEqual(result["geometry_status"], "ALIGNED")
        self.assertEqual(result["final_bbox"], slot.expected_bbox)
        self.assertFalse(result["has_ink"])

    def test_ocr_coordinate_warning_does_not_fail_quality(self):
        slot = Slot(
            1, "fill", "q", [10, 10, 110, 190], 1,
            recognized_text="x", student_answer="x",
            handwriting_bbox=[10, 10, 110, 190],
            geometry_status="ALIGNED_WITH_WARNING", content_status="RECOGNIZED",
            status="CONVERGED_WITH_WARNING",
            audit={"coordinate_authority": "ocr_boxes_only"},
        )
        item = ExamItem("q", "q", slots=[slot], expected_slot_count=1,
                        answer_status="COMPLETE")
        package = self.package(item)
        package.registration = [{"status": "FAILED_LOW_CONFIDENCE"}]
        quality = validate_item_regions(package, [self.page()])
        self.assertEqual(quality["status"], "OK")
        self.assertFalse(quality["registration_required_for_coordinates"])

    def test_warning_geometry_keeps_recognized_answer_and_reports_warning(self):
        slot = Slot(
            1, "fill", "q", [10, 10, 110, 190], 1,
            expected_text="x", recognized_text="x", student_answer="x",
            handwriting_bbox=[45, 70, 75, 120], evidence_bbox=[45, 70, 75, 120],
            geometry_status="ALIGNED_WITH_WARNING", content_status="RECOGNIZED",
            status="CONVERGED_WITH_WARNING",
        )
        item = ExamItem("q", "q", slots=[slot], expected_slot_count=1)
        package = self.package(item)
        finalize_answers(package, [self.page()])
        self.assertEqual(item.student_answer, "x")
        self.assertEqual(item.answer_status, "COMPLETE")
        self.assertEqual(item.slot_evaluation["geometry_verified_slots"], 1)
        self.assertEqual(item.slot_evaluation["geometry_warning_slots"], 1)

    def test_failed_geometry_still_blocks_final_answer(self):
        slot = Slot(
            1, "fill", "q", [10, 10, 110, 190], 1,
            expected_text="x", recognized_text="x", student_answer="x",
            handwriting_bbox=[0, 0, 200, 200], geometry_status="FAILED",
            content_status="RECOGNIZED", status="CONVERGED_SUCCESS",
        )
        item = ExamItem("q", "q", slots=[slot], expected_slot_count=1)
        package = self.package(item)
        finalize_answers(package, [self.page()])
        self.assertIsNone(item.student_answer)
        self.assertEqual(item.answer_status, "UNRESOLVED")


if __name__ == "__main__":
    unittest.main()
