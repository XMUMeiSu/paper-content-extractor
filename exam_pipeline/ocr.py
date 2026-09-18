"""Lazy OCR façade."""
from pathlib import Path
from typing import List, Sequence
from .contracts import OCRBlock


class OCRService:
    @staticmethod
    def recognize_page(path: Path, engine: str = "paddle",
                       language: str = "chi_sim+eng") -> List[OCRBlock]:
        from homework_extractor import ocr_page
        return ocr_page(Path(path), engine, language)

    def recognize_crop(self, path: Path, bbox: Sequence[float], padding: int = 0,
                       engine: str = "paddle", language: str = "chi_sim+eng") -> List[OCRBlock]:
        import cv2
        import tempfile

        image = cv2.imread(str(path))
        if image is None or len(bbox) < 4:
            return []
        height, width = image.shape[:2]
        left, top, right, bottom = [int(round(value)) for value in bbox[:4]]
        left, top = max(0, left - padding), max(0, top - padding)
        right, bottom = min(width, right + padding), min(height, bottom + padding)
        if right <= left or bottom <= top:
            return []
        with tempfile.NamedTemporaryFile(suffix=".jpg") as handle:
            cv2.imwrite(handle.name, image[top:bottom, left:right])
            blocks = self.recognize_page(Path(handle.name), engine, language)
        return [OCRBlock(block.text,
                         [block.bbox[0] + left, block.bbox[1] + top,
                          block.bbox[2] + left, block.bbox[3] + top],
                         block.confidence) for block in blocks]

    def recognize_mask(self, mask, engine: str = "paddle",
                       language: str = "chi_sim+eng") -> List[OCRBlock]:
        """Recognize a binary handwriting mask as black ink on white paper."""
        import cv2
        import tempfile

        if mask is None or getattr(mask, "size", 0) == 0:
            return []
        points = cv2.findNonZero(mask)
        if points is None:
            return []
        x, y, width, height = cv2.boundingRect(points)
        padding = 8
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(mask.shape[1], x + width + padding), min(mask.shape[0], y + height + padding)
        crop = 255 - mask[y1:y2, x1:x2]
        # OCR detectors are more stable on a three-channel, modestly padded image.
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        with tempfile.NamedTemporaryFile(suffix=".28.png") as handle:
            cv2.imwrite(handle.name, crop)
            return self.recognize_page(Path(handle.name), engine, language)
