"""Page normalization onto the canonical 1654x2338 canvas."""
from pathlib import Path
from typing import Any, Dict

CANONICAL_WIDTH = 1654
CANONICAL_HEIGHT = 2338


def preprocess_page(source: Path, target: Path) -> Dict[str, Any]:
    import cv2
    import numpy as np

    source, target = Path(source), Path(target)
    image = cv2.imread(str(source))
    if image is None:
        raise IOError(f"无法读取图片: {source}")
    original_height, original_width = image.shape[:2]
    normalized = cv2.resize(image, (CANONICAL_WIDTH, CANONICAL_HEIGHT), interpolation=cv2.INTER_AREA)
    # Flatten illumination while retaining colored teacher/student marks.
    ycrcb = cv2.cvtColor(normalized, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    background = cv2.GaussianBlur(y, (0, 0), 35)
    y = cv2.normalize(cv2.divide(y, background, scale=255), None, 0, 255, cv2.NORM_MINMAX)
    normalized = cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), normalized, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise IOError(f"无法写入归一化图片: {target}")

    # Keep a lossless binary geometry track beside the recognition image.
    # OCR/VLM consumes the color-preserving JPEG; slot/line diagnostics can use
    # this PNG without JPEG ringing or low-frequency illumination residue.
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    geometry_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11,
    )
    # Remove isolated one-pixel noise without joining neighbouring glyphs.
    ink = cv2.bitwise_not(geometry_mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    cleaned_ink = np.zeros_like(ink)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= 2:
            cleaned_ink[labels == label] = 255
    geometry_mask = cv2.bitwise_not(cleaned_ink)
    mask_path = target.with_name(f"{target.stem}_mask.png")
    if not cv2.imwrite(str(mask_path), geometry_mask):
        raise IOError(f"无法写入几何掩码: {mask_path}")
    return {
        "original_filename": source.name,
        "source_path": str(source),
        "normalized_path": str(target),
        "geometry_mask_path": str(mask_path),
        "original_size": {"width": original_width, "height": original_height},
        "normalized_size": {"width": CANONICAL_WIDTH, "height": CANONICAL_HEIGHT},
        "scale_x": round(CANONICAL_WIDTH / max(1, original_width), 6),
        "scale_y": round(CANONICAL_HEIGHT / max(1, original_height), 6),
        "dual_track": {
            "recognition": str(target),
            "geometry": str(mask_path),
            "geometry_values": [0, 255],
        },
        "status": "NORMALIZED",
    }


class IngestionService:
    preprocess_page = staticmethod(preprocess_page)
