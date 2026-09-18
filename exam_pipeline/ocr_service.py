"""OCR service utilities for image analysis."""
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import tempfile


def run_ocr_on_image(img: Image.Image, ocr_engine: str) -> Optional[Dict[str, Any]]:
    """Run OCR on a PIL Image and return structured result.

    Args:
        img: PIL Image to OCR
        ocr_engine: Engine name ("paddleocr" or "paddle")

    Returns:
        Dict with "lines" list, each line having:
        - "bbox": [x1, y1, x2, y2] in image pixel coordinates
        - "text": recognized text
        - "confidence": 0.0-1.0
    """
    if ocr_engine not in {"paddleocr", "paddle"}:
        return None

    try:
        from paddleocr import PaddleOCR

        # Save image to temp file (PaddleOCR prefers file paths)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = Path(tmp.name)

        try:
            # Initialize PaddleOCR (v3.x removed show_log parameter)
            ocr = PaddleOCR(use_angle_cls=True, lang='ch')

            # Run OCR
            result = ocr.ocr(str(tmp_path), cls=True)

            if not result or not result[0]:
                return {"lines": []}

            # Convert to our format
            lines = []
            for line in result[0]:
                if not line or len(line) < 2:
                    continue

                bbox_points = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = line[1]    # (text, confidence)

                if not bbox_points or not text_info:
                    continue

                # Convert 4-point polygon to axis-aligned bbox
                xs = [p[0] for p in bbox_points]
                ys = [p[1] for p in bbox_points]
                bbox = [min(xs), min(ys), max(xs), max(ys)]

                text = text_info[0] if isinstance(text_info, (tuple, list)) else str(text_info)
                conf = text_info[1] if isinstance(text_info, (tuple, list)) and len(text_info) > 1 else 0.9

                lines.append({
                    "bbox": bbox,
                    "text": text,
                    "confidence": float(conf),
                })

            return {"lines": lines}

        finally:
            # Clean up temp file
            tmp_path.unlink(missing_ok=True)

    except Exception as e:
        print(f"OCR failed: {e}")
        return None
