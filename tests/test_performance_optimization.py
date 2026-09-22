import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from exam_pipeline.answer_recognition import AnswerRecognizer
from exam_pipeline.answer_vision import AnswerVisionClient
from exam_pipeline.contracts import (ExamItem, ExamPackage, ExamQuestion,
                                     ExamSection, OCRBlock, Page, Slot)
from exam_pipeline.ocr import OCRService
from exam_pipeline.performance import PerformanceCollector
from exam_pipeline.visualization import render_slot_overlays


class _EmptyOCR:
    def recognize_crop(self, *args, **kwargs):
        return []

    def recognize_page(self, *args, **kwargs):
        return []


class _BatchVision:
    def __init__(self, missing=()):
        self.missing = set(missing)
        self.batch_calls = 0
        self.single_calls = 0
        self.formula_calls = 0
        self.batch_routes = []

    def analyze_batch(self, entries, paths, route="handwriting"):
        self.batch_calls += 1
        self.batch_routes.append(route)
        return {
            entry["slot_id"]: {
                "transcription": "A", "legible": True,
                "content_kind": "handwriting",
            }
            for entry in entries if entry["slot_id"] not in self.missing
        }

    def analyze_choice(self, prompt, paths):
        self.single_calls += 1
        return {"transcription": "B", "legible": True,
                "content_kind": "handwriting"}

    def analyze_formula(self, prompt, paths):
        self.formula_calls += 1
        return {"transcription": "x^{2}", "legible": True,
                "content_kind": "handwriting"}

    def analyze(self, prompt, paths):
        self.single_calls += 1
        return {"transcription": "text", "legible": True,
                "content_kind": "handwriting"}


class PerformanceOptimizationTests(unittest.TestCase):
    def _page(self, folder):
        path = Path(folder) / "page.png"
        cv2.imwrite(str(path), np.full((300, 500, 3), 255, np.uint8))
        return Page(1, str(path), 500, 300, [])

    @staticmethod
    def _entries(page, count=5, kind="choice"):
        return [{
            "key": index,
            "slot_id": f"q{index}:slot:1",
            "page": page,
            "box": [20 + index * 30, 20, 45 + index * 30, 90],
            "question": "Choose one",
            "kind": kind,
        } for index in range(count)]

    def test_one_page_five_slots_use_one_answer_vlm_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            vision = _BatchVision()
            result = AnswerRecognizer(_EmptyOCR(), vision).recognize_many(
                self._entries(self._page(tmp)))
        self.assertEqual(vision.batch_calls, 1)
        self.assertEqual(vision.single_calls, 0)
        self.assertEqual([result[index]["text"] for index in range(5)], ["A"] * 5)

    def test_missing_batch_id_only_retries_that_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            vision = _BatchVision({"q2:slot:1"})
            result = AnswerRecognizer(_EmptyOCR(), vision).recognize_many(
                self._entries(self._page(tmp)))
        self.assertEqual(vision.batch_calls, 1)
        self.assertEqual(vision.single_calls, 1)
        self.assertEqual(result[2]["text"], "B")
        self.assertEqual(sum(value["text"] == "A" for value in result.values()), 4)

    def test_formula_keeps_dedicated_batch_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            vision = _BatchVision()
            entry = self._entries(self._page(tmp), 1, kind="formula")
            entry[0]["question"] = "Transcribe y=x^2"
            result = AnswerRecognizer(_EmptyOCR(), vision).recognize_many(entry)
        self.assertEqual(vision.batch_calls, 1)
        self.assertEqual(vision.batch_routes, ["formula"])
        self.assertEqual(vision.formula_calls, 0)
        self.assertEqual(result[0]["text"], "A")

    def test_batch_client_rejects_unknown_and_duplicate_ids(self):
        entries = [{"slot_id": "q1"}, {"slot_id": "q2"}]
        paths = [Path("one.png"), Path("two.png")]
        for answers, reason in (
            ([{"slot_id": "q3", "transcription": "A", "legible": True,
               "content_kind": "handwriting"}], "UNKNOWN_BATCH_SLOT_ID"),
            ([{"slot_id": "q1", "transcription": "A", "legible": True,
               "content_kind": "handwriting"}] * 2, "DUPLICATE_BATCH_SLOT_ID"),
        ):
            client = AnswerVisionClient(lambda *args, value={"answers": answers}: value)
            with self.assertRaisesRegex(ValueError, reason):
                client.analyze_batch(entries, paths, route="choice")

    def test_ocr_crop_cache_avoids_duplicate_backend_call(self):
        class CachedOCR(OCRService):
            calls = 0

            @staticmethod
            def recognize_page(path, engine="paddle", language="eng"):
                CachedOCR.calls += 1
                return [OCRBlock("D", [1, 1, 10, 10], .9)]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.png"
            cv2.imwrite(str(path), np.full((100, 100, 3), 255, np.uint8))
            service = CachedOCR()
            first = service.recognize_crop(path, [10, 10, 50, 50], 4, "paddle", "eng")
            second = service.recognize_crop(path, [10, 10, 50, 50], 4, "paddle", "eng")
        self.assertEqual(CachedOCR.calls, 1)
        self.assertEqual(first[0].text, second[0].text)

    def test_high_resolution_ocr_maps_detection_back_to_page_coordinates(self):
        class ScaledOCR(OCRService):
            @staticmethod
            def recognize_page(path, engine="paddle", language="eng"):
                return [OCRBlock("26", [10, 20, 30, 40], .9)]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.png"
            cv2.imwrite(str(path), np.full((100, 100, 3), 255, np.uint8))
            blocks = ScaledOCR().recognize_crop_high_resolution(
                path, [20, 30, 60, 70], padding=0, scale=2.0,
                engine="paddle", language="eng")
        self.assertEqual(blocks[0].bbox, [25, 40, 35, 50])

    def test_request_metrics_include_usage_attempts_and_bytes(self):
        class Audited(dict):
            response_audit = {
                "attempts": 2, "retry_reasons": ["network"],
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            path.write_bytes(b"image")
            collector = PerformanceCollector()
            request = collector.instrument(
                lambda prompt, paths, schema: Audited(ok=True),
                stage="answer_recognition", provider="test",
                operation="answer_transcription_request")
            request("prompt", [path], {})
            event = collector.summary()["events"][0]
        self.assertEqual(event["attempts"], 2)
        self.assertEqual(event["retry_reasons"], ["network"])
        self.assertEqual(event["image_bytes"], 5)
        self.assertEqual(event["usage"]["input_tokens"], 10)

    def test_slot_visualization_writes_contact_sheet(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = self._page(tmp)
            slot = Slot(1, "choice", "q1", [20, 20, 90, 45], 1,
                        handwriting_bbox=[20, 20, 90, 45],
                        geometry_status="ALIGNED")
            item = ExamItem("q1", "1", item_type="choice", slots=[slot])
            package = ExamPackage(
                "e", "", "math", "student", "student001", 1, [page.path],
                [ExamSection("s", "", [ExamQuestion("q1", 1, "", [item])])])
            summary = render_slot_overlays(package, [page], Path(tmp) / "visualizations", "student001")
            self.assertEqual(summary["status"], "EXPORTED")
            self.assertTrue(Path(summary["contact_sheet"]).is_file())


if __name__ == "__main__":
    unittest.main()
