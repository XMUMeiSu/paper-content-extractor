import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from exam_pipeline.contracts import DiagramRef, ExamItem, ExamPackage, ExamQuestion, ExamSection, Page, PageRegion
from exam_pipeline.diagram_assets import DiagramAssetService


class DiagramAssetTests(unittest.TestCase):
    def test_explicit_diagram_is_cropped_and_linked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "page.jpg"
            image = np.full((240, 320, 3), 255, dtype=np.uint8)
            cv2.rectangle(image, (80, 90), (210, 180), (0, 0, 0), 3)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            page = Page(1, str(image_path), 320, 240, [])
            item = ExamItem(
                "q1", "1", item_type="choice",
                stem_region=PageRegion(1, str(image_path), [20, 20, 300, 220]),
                diagrams=[DiagramRef("figure", bbox=[80, 90, 210, 180])],
            )
            package = ExamPackage(
                "exam", "", "math", "teacher", None, 1, [str(image_path)],
                [ExamSection("s", "", [ExamQuestion("q1", 1, "", [item])])],
            )
            summary = DiagramAssetService(padding=4).export(package, [page], root)
            self.assertEqual(summary["diagram_assets"], 1)
            diagram = item.diagrams[0]
            self.assertEqual(diagram.page_index, 1)
            self.assertEqual(diagram.audit["coordinate_format"], "xyxy")
            self.assertTrue((root / diagram.image_url).is_file())
            self.assertTrue(diagram.image_url.startswith("diagram_assets/"))

    def test_student_answer_corridor_supplies_page_when_stem_is_inherited(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "student.jpg"
            image = np.full((180, 260, 3), 255, dtype=np.uint8)
            cv2.rectangle(image, (90, 55), (190, 145), (0, 0, 0), 3)
            cv2.imwrite(str(image_path), image)
            page = Page(1, str(image_path), 260, 180, [])
            item = ExamItem(
                "q9_1", "9.(1)", item_type="large_writing",
                answer_regions=[PageRegion(1, str(image_path), [20, 20, 240, 170])],
                diagrams=[DiagramRef("figure", bbox=[90, 55, 190, 145])],
            )
            package = ExamPackage(
                "exam", "", "math", "student", "student001", 1,
                [str(image_path)],
                [ExamSection("s", "", [ExamQuestion("q9", 9, "", [item])])],
            )
            DiagramAssetService(padding=2).export(package, [page], root)
            self.assertEqual(item.diagrams[0].page_index, 1)
            self.assertTrue((root / item.diagrams[0].image_url).is_file())


if __name__ == "__main__":
    unittest.main()
