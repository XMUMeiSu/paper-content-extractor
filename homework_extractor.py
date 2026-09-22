#!/usr/bin/env python3
"""OCR + Doubao multimodal extraction for teacher and student homework sheets.

The module deliberately has very few hard dependencies.  OCR can be provided by
PaddleOCR, pytesseract, or a JSONL file; vision extraction uses the
Volcano Engine Ark/Doubao HTTP API directly. This makes the pipeline usable
without an SDK and easy to point at a compatible private endpoint.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import inspect
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from exam_pipeline.contracts import (
    OCRBlock as ModularOCRBlock,
    Page as ModularPage,
    PageRegion as ModularPageRegion,
    ExamItem as ModularExamItem,
    ExamQuestion as ModularExamQuestion,
    ExamSection as ModularExamSection,
    ExamPackage as ModularExamPackage,
    DiagramRef as ModularDiagramRef,
    TriTargetGrounding as ModularTriTargetGrounding,
    Slot as ModularSlot,
    RoIPatchRef as ModularRoIPatchRef,
)

# All adapters must share the same OCR cache when invoked as a script.
if __name__ == "__main__":
    sys.modules.setdefault("homework_extractor", sys.modules[__name__])

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_PADDLE_OCR = None
_PADDLE_OCR_INIT_ERROR: Optional[Exception] = None
LOGGER = logging.getLogger("exam_pipeline")


def _load_private_user_environment() -> Optional[Path]:
    """Load an allow-listed mode-600 user env file without executing shell code.

    Environment variables already injected by a deployment always win.  This
    local fallback lets IDE tasks and non-interactive runners use the same
    credential file as a terminal while keeping secrets outside the repository.
    """
    configured = os.getenv("GRADING_SECRET_ENV", "").strip()
    env_path = Path(configured).expanduser() if configured else (
        Path.home() / ".config" / "intelligent-grading-system" / "doubao.env"
    )
    if not env_path.is_file():
        return None
    try:
        # Refuse group/world-readable credential files.
        if env_path.stat().st_mode & 0o077:
            return None
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    allowed = {
        "DOUBAO_API_KEY", "DOUBAO_BASE_URL", "DOUBAO_MODEL",
        "DOUBAO_RESPONSES_ENDPOINT", "PADDLEOCR_VL_API_KEY",
        "TREE_LLM_API_KEY",
    }
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if name not in allowed or name in os.environ:
            continue
        try:
            parsed = shlex.split(raw_value.strip(), posix=True)
        except ValueError:
            continue
        if len(parsed) == 1 and parsed[0]:
            os.environ[name] = parsed[0]
    return env_path


PRIVATE_ENV_PATH = _load_private_user_environment()

# Secrets must never be committed.  The CLI can override these environment
# defaults for one-off jobs, but production deployments should inject them via
# a secret manager as DOUBAO_API_KEY.
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY", "").strip()
# 这是方舟“兼容 OpenAI 接口协议（Responses API）”的 Base URL。
# 请勿改成 https://ark.cn-beijing.volces.com/api/v3，该地址会产生额外费用。
DOUBAO_BASE_URL = os.getenv(
    "DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/plan/v3"
).rstrip("/")
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL", "doubao-seed-2.0-lite")
DOUBAO_RESPONSES_ENDPOINT = os.getenv(
    "DOUBAO_RESPONSES_ENDPOINT", DOUBAO_BASE_URL.rstrip("/") + "/responses"
)
# PaddleOCR-VL runs as an independent Python 3.9+ service.  Keeping its URL
# empty by default prevents an accidental network call in existing installs.
PADDLEOCR_VL_ENDPOINT = os.getenv("PADDLEOCR_VL_ENDPOINT", "").strip()
PADDLEOCR_VL_MODEL = os.getenv(
    "PADDLEOCR_VL_MODEL", "PaddlePaddle/PaddleOCR-VL-1.6"
).strip()
PADDLEOCR_VL_API_KEY = os.getenv("PADDLEOCR_VL_API_KEY", "").strip()
# Optional text-only instruction LLM.  It refines only the teacher's logical
# tree; all coordinates remain owned by PP-OCRv5 and deterministic grounding.
TREE_LLM_ENDPOINT = os.getenv("TREE_LLM_ENDPOINT", "").strip()
TREE_LLM_MODEL = os.getenv("TREE_LLM_MODEL", "").strip()
TREE_LLM_API_KEY = os.getenv("TREE_LLM_API_KEY", "").strip()
# PaddleOCR 3.x downloads models to this directory. Keeping it under the
# project avoids failures on machines where ~/.paddlex is read-only.
PADDLE_CACHE_HOME = os.getenv(
    "PADDLE_PDX_CACHE_HOME", str(Path(__file__).resolve().parent / ".paddlex-cache")
)
# PaddlePaddle 3.0.0 can fail loading the PP-OCRv5 server packages with
# ``strides ... Expected Int32Attribute``. The mobile packages are smaller and
# work with this runtime; override these names when a server GPU model is
# available and known to be compatible.
PADDLE_OCR_VERSION = "PP-OCRv5"
PADDLE_MODEL_TIER = os.getenv("PADDLE_MODEL_TIER", "mobile").strip().lower()
if PADDLE_MODEL_TIER not in {"mobile", "server"}:
    PADDLE_MODEL_TIER = "mobile"
PADDLE_DET_MODEL = os.getenv("PADDLE_DET_MODEL", f"PP-OCRv5_{PADDLE_MODEL_TIER}_det")
PADDLE_REC_MODEL = os.getenv("PADDLE_REC_MODEL", f"PP-OCRv5_{PADDLE_MODEL_TIER}_rec")


@dataclass
class OCRBlock:
    text: str
    bbox: List[float]
    confidence: Optional[float] = None


@dataclass
class Page:
    index: int
    path: str
    width: Optional[int]
    height: Optional[int]
    ocr: List[OCRBlock]


@dataclass
class PageExitContext:
    """Page-exit snapshot used for prototype-compatible semantic stitching."""

    page_index: int
    last_question_num: int = 0
    last_question_title: str = ""
    last_text_tail: str = ""
    is_semantically_incomplete: bool = False
    page_file: str = ""
    last_question_id: str = ""
    last_item_id: str = ""
    last_item_name: str = ""
    declared_sub_count: int = 0
    actual_sub_count: int = 0
    last_region_bottom: float = 0.0


@dataclass
class PageRegion:
    """A physical region on a normalized page, using [left, top, right, bottom]."""

    page_index: int
    page_file: str
    bbox: List[float]
    confidence: Optional[float] = None
    ocr_text: str = ""


@dataclass
class ExamItem:
    """Atomic scoring item in the canonical exam contract."""

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


@dataclass
class ExamQuestion:
    question_id: str
    question_num: int
    question_title: str
    items: List[ExamItem] = field(default_factory=list)


@dataclass
class ExamSection:
    section_id: str
    section_title: str
    questions: List[ExamQuestion] = field(default_factory=list)


@dataclass
class ExamPackage:
    """Serializable Exam -> Section -> Question -> Item data contract."""

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# The script remains a backwards-compatible import surface, while all new
# integrations use the shared contracts in ``exam_pipeline.contracts``.
OCRBlock = ModularOCRBlock
Page = ModularPage
PageRegion = ModularPageRegion
ExamItem = ModularExamItem
ExamQuestion = ModularExamQuestion
ExamSection = ModularExamSection
ExamPackage = ModularExamPackage
DiagramRef = ModularDiagramRef
TriTargetGrounding = ModularTriTargetGrounding
Slot = ModularSlot
RoIPatchRef = ModularRoIPatchRef


def natural_key(path: Path) -> Tuple[Any, ...]:
    return tuple(int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name))


_PAGE_PATTERNS = (
    re.compile(r"(?:page|p|第)[-_]?(\d+)(?:页)?", re.I),
    re.compile(r"_(\d+)$"),
    re.compile(r"-(\d+)$"),
    re.compile(r"(\d+)$"),
)


def extract_page_number(path: Path) -> int:
    """Match the prototype's explicit page-number ordering rules."""
    stem = path.stem.lower()
    for pattern in _PAGE_PATTERNS:
        match = pattern.search(stem)
        if match:
            return int(match.group(1))
    digits = re.findall(r"\d+", stem)
    return int(digits[-1]) if digits else 999999


def page_sequence_key(path: Path) -> Tuple[int, str]:
    return extract_page_number(path), path.name


def input_fingerprint(paths: Sequence[Path]) -> str:
    """Content hash used to trace the exact source pages of an artifact."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8", errors="surrogatepass"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def image_size(path: Path) -> Tuple[Optional[int], Optional[int]]:
    try:
        from PIL import Image  # type: ignore
        with Image.open(path) as im:
            return im.width, im.height
    except Exception:
        return None, None


CANONICAL_PAGE_SIZE = (1654, 2338)  # Production canonical canvas: width, height


def classify_image_medium(path: Path, image: Any) -> bool:
    """Return True for likely phone photos using filename and corner evidence."""
    name = path.name.lower()
    if any(token in name for token in ("phone", "mobile", "cam", "photo", "img_", "wx")):
        return True
    if image is None:
        return False
    import cv2
    height, width = image.shape[:2]
    mh, mw = max(10, int(height * 0.05)), max(10, int(width * 0.05))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = (gray[:mh, :mw], gray[:mh, -mw:], gray[-mh:, :mw], gray[-mh:, -mw:])
    return any(float(c.mean()) < 160 for c in corners if c.size)


def _rectify_image(image: Any, phone_photo: bool) -> Tuple[Any, Dict[str, Any]]:
    """Rectify a page with OpenCV; failures return the original image with metadata."""
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    target_w, target_h = CANONICAL_PAGE_SIZE
    meta: Dict[str, Any] = {
        "original_size": [int(width), int(height)],
        "original_shape": [int(height), int(width)],
        "method": "resize",
        "rectification_type": "none",
    }
    working = image
    if phone_photo:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(contour) < height * width * 0.25:
                break
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(polygon) != 4:
                continue
            points = polygon.reshape(4, 2).astype("float32")
            sums, diffs = points.sum(axis=1), np.diff(points, axis=1).ravel()
            ordered = np.array([
                points[np.argmin(sums)], points[np.argmin(diffs)],
                points[np.argmax(sums)], points[np.argmax(diffs)]
            ], dtype="float32")
            destination = np.array([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]], dtype="float32")
            working = cv2.warpPerspective(
                image, cv2.getPerspectiveTransform(ordered, destination),
                (target_w, target_h), flags=cv2.INTER_LANCZOS4)
            meta.update({
                "method": "perspective_4point",
                "rectification_type": "perspective_4pt",
                "detected_corners": ordered.tolist(),
                "detected_pts": ordered.tolist(),
            })
            break
    if meta["method"] == "resize":
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        coordinates = np.column_stack(np.where(binary > 0))
        angle = 0.0
        if len(coordinates) >= 100:
            angle = float(cv2.minAreaRect(coordinates)[-1])
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            else:
                angle = -angle
        if 0.2 <= abs(angle) <= 20:
            matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
            working = cv2.warpAffine(working, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
        working = cv2.resize(working, (target_w, target_h), interpolation=cv2.INTER_AREA)
        meta.update({
            "rectification_type": "deskew_resize",
            "deskew_angle": round(angle, 3),
        })
    return working, meta


def _flatten_illumination(image: Any, kernel_size: int = 51) -> Any:
    """Remove low-frequency shadows while preserving color ink channels."""
    import cv2
    import numpy as np
    size = kernel_size if kernel_size % 2 else kernel_size + 1
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    background = cv2.dilate(y, cv2.getStructuringElement(cv2.MORPH_RECT, (size, size)))
    background = cv2.medianBlur(background, 21)
    normalized = np.clip((y.astype("float32") / np.maximum(background, 1)) * 255, 0, 255).astype("uint8")
    normalized = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(normalized)
    return cv2.cvtColor(cv2.merge([normalized, cr, cb]), cv2.COLOR_YCrCb2BGR)


def preprocess_page(path: Path, output_path: Path) -> Dict[str, Any]:
    """Compatibility facade for :mod:`exam_pipeline.ingestion`."""
    from exam_pipeline.ingestion import preprocess_page as normalize_page
    return normalize_page(path, output_path)


def discover_documents(dataset: Path, include_scan: bool = False) -> List[Tuple[str, str, List[Path]]]:
    """Return (subject, role, pages), grouping pages by dataset/leaf directory."""
    docs: List[Tuple[str, str, List[Path]]] = []
    for subject_dir in sorted((p for p in dataset.iterdir() if p.is_dir()), key=natural_key):
        for leaf in sorted((p for p in subject_dir.iterdir() if p.is_dir()), key=natural_key):
            role = "teacher" if leaf.name == "teacher" or "教师" in leaf.name else "student"
            files = [p for p in leaf.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
            if not include_scan:
                files = [p for p in files if not p.name.startswith("扫描_")]
            # Folders used for unmatched/duplicate pages are not complete student documents.
            if role == "student" and leaf.name in {"未匹配第二页", "重复图片"}:
                continue
            if files:
                docs.append((subject_dir.name, role, sorted(files, key=page_sequence_key)))
    return docs


def cohort_student_documents(
        documents: Sequence[Tuple[str, str, List[Path]]],
        subject: str) -> List[List[Path]]:
    """Select same-subject cohort inputs independently of output filters."""
    return [
        document_paths
        for doc_subject, doc_role, document_paths in documents
        if doc_subject == subject and doc_role == "student"
    ]


def _ocr_paddle(path: Path) -> List[OCRBlock]:
    global _PADDLE_OCR, _PADDLE_OCR_INIT_ERROR
    if _PADDLE_OCR_INIT_ERROR is not None:
        raise RuntimeError(str(_PADDLE_OCR_INIT_ERROR))
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(Path(PADDLE_CACHE_HOME).resolve()))
    from paddleocr import PaddleOCR  # type: ignore
    # PaddleOCR 3.x removed show_log/use_angle_cls and renamed the orientation
    # switch. Detect the installed signature so both 2.x and 3.x work.
    if _PADDLE_OCR is None:
        params = inspect.signature(PaddleOCR).parameters
        if "use_textline_orientation" in params:
            try:
                _PADDLE_OCR = PaddleOCR(
                    lang="ch",
                    ocr_version=PADDLE_OCR_VERSION,
                    text_detection_model_name=PADDLE_DET_MODEL,
                    text_recognition_model_name=PADDLE_REC_MODEL,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            except Exception as exc:
                message = str(exc)
                if "strides" in message and "Int32Attribute" in message:
                    hint = "检测到 PaddlePaddle 与 OCR 模型版本不兼容；当前代码已切换到 PP-OCRv5 mobile，请清理旧缓存并联网下载 mobile 模型，或升级 paddlepaddle。"
                elif "No available model hosting platforms" in message:
                    hint = "模型下载源不可访问；请联网下载模型，或设置 PADDLE_PDX_CACHE_HOME 指向已有模型目录。"
                else:
                    hint = "请检查 PaddlePaddle/PaddleOCR 版本、网络和模型缓存目录。"
                _PADDLE_OCR_INIT_ERROR = RuntimeError(f"PaddleOCR 初始化失败: {message}。{hint}")
                raise _PADDLE_OCR_INIT_ERROR from exc
        else:
            # Fallback for older PaddleOCR 2.x versions
            _PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang="ch")

    if hasattr(_PADDLE_OCR, "predict"):
        result = _PADDLE_OCR.predict(str(path))
    else:  # PaddleOCR 2.x
        result = _PADDLE_OCR.ocr(str(path), cls=True)

    # PaddleOCR 2.x returns [[points, (text, score)], ...].
    if result and isinstance(result[0], (list, tuple)) and result[0] and isinstance(result[0][0], (list, tuple)) and len(result[0]) == 2 and isinstance(result[0][1], (list, tuple)):
        result = result[0]
        blocks: List[OCRBlock] = []
        for line in result:
            points, (text, score) = line
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
            blocks.append(OCRBlock(text=str(text), bbox=[min(xs), min(ys), max(xs), max(ys)], confidence=float(score)))
        return blocks

    # PaddleOCR 3.x returns OCRResult objects with rec_texts/rec_scores and
    # rec_polys (or rec_boxes). These objects are dict-like in 3.0+.
    item = result[0] if result else None
    if item is None:
        return []
    def field(name: str, default: Any = None) -> Any:
        try:
            value = item.get(name, default)
        except AttributeError:
            value = None
        if value is None and hasattr(item, "json"):
            value = item.json.get("res", {}).get(name, default)
        return value
    texts = field("rec_texts", [])
    scores = field("rec_scores", [])
    polygons = field("rec_polys", None)
    if polygons is None:
        polygons = field("rec_boxes", [])
    # Paddle may expose numpy arrays; never use them directly in ``x or y``
    # because numpy truth-value checks are ambiguous.
    texts = [] if texts is None else list(texts)
    scores = [] if scores is None else list(scores)
    polygons = [] if polygons is None else list(polygons)
    blocks = []
    for i, text in enumerate(texts):
        poly = polygons[i] if i < len(polygons) else []
        try:
            # PaddleOCR 3.x commonly returns numpy arrays.  Convert through
            # ``tolist`` first so both rec_polys ``[[x,y], ...]`` and
            # rec_boxes ``[x1,y1,x2,y2]`` are handled without numpy truth
            # value/type pitfalls.
            raw_poly = poly.tolist() if hasattr(poly, "tolist") else poly
            points = list(raw_poly) if raw_poly is not None else []
            if points and isinstance(points[0], (list, tuple)):
                xs = [float(point[0]) for point in points if len(point) >= 2]
                ys = [float(point[1]) for point in points if len(point) >= 2]
            else:
                values = [float(value) for value in points]
                xs, ys = [values[0], values[2]], [values[1], values[3]]
            if not xs or not ys:
                raise ValueError("OCR 多边形为空")
            bbox = [min(xs), min(ys), max(xs), max(ys)]
        except (TypeError, ValueError, IndexError, AttributeError):
            bbox = []
        score = scores[i] if i < len(scores) else None
        blocks.append(OCRBlock(text=str(text), bbox=bbox, confidence=float(score) if score is not None else None))
    return blocks


def _ocr_tesseract(path: Path, language: str = "chi_sim+eng") -> List[OCRBlock]:
    cmd = ["tesseract", str(path), "stdout", "--psm", "6", "-l", language, "tsv"]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise RuntimeError("未找到 PaddleOCR 或 tesseract。请安装其一，或使用 --ocr-json/--skip-ocr。") from exc
    lines = proc.stdout.splitlines()
    if not lines:
        return []
    blocks: List[OCRBlock] = []
    for row in lines[1:]:
        cols = row.split("\t")
        if len(cols) < 12 or not cols[11].strip():
            continue
        try:
            x, y, w, h, conf = float(cols[6]), float(cols[7]), float(cols[8]), float(cols[9]), float(cols[10])
            confidence = conf / 100.0 if conf >= 0 else None
        except ValueError:
            continue
        blocks.append(OCRBlock(text=cols[11].strip(), bbox=[x, y, x + w, y + h], confidence=confidence))
    return blocks


def ocr_page(path: Path, engine: str = "auto", language: str = "chi_sim+eng") -> List[OCRBlock]:
    if engine == "none":
        return []
    if engine in {"auto", "paddle"}:
        try:
            return _ocr_paddle(path)
        except ImportError:
            if engine == "paddle":
                raise RuntimeError("--ocr paddle 需要安装 paddleocr 与 paddlepaddle。")
        except Exception as exc:
            if engine == "paddle":
                raise RuntimeError(f"PaddleOCR 处理 {path} 失败: {exc}") from exc
    return _ocr_tesseract(path, language)


def read_ocr_jsonl(path: Optional[Path]) -> Dict[str, List[OCRBlock]]:
    """Read optional JSONL: {"image": "relative/or/absolute", "blocks": [...]}"""
    if not path:
        return {}
    result: Dict[str, List[OCRBlock]] = {}
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                blocks = [OCRBlock(str(b.get("text", "")), list(b.get("bbox", [])), b.get("confidence")) for b in item["blocks"]]
                result[str(Path(item["image"]).resolve())] = blocks
            except Exception as exc:
                raise ValueError(f"OCR JSONL 第 {line_no} 行无效: {exc}") from exc
    return result


def page_file_fingerprint(path: Path) -> str:
    """Stable per-image fingerprint used by caches and downstream audits."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def make_pages(paths: Sequence[Path], engine: str, ocr_map: Dict[str, List[OCRBlock]],
               language: str, document_id: str = "") -> List[Page]:
    pages = []
    for i, path in enumerate(paths, 1):
        key = str(path.resolve())
        blocks = ocr_map.get(key)
        if blocks is None:
            ocr_started = time.monotonic()
            try:
                blocks = ocr_page(path, engine, language)
            except Exception as exc:
                from exam_pipeline.performance import get_active_collector
                collector = get_active_collector()
                if collector:
                    collector.record(
                        stage="ocr", provider=str(engine), operation="ocr_page",
                        duration_seconds=time.monotonic() - ocr_started,
                        paths=[path], status="FAILED", error_type=type(exc).__name__)
                raise
            from exam_pipeline.performance import get_active_collector
            collector = get_active_collector()
            if collector:
                collector.record(
                    stage="ocr", provider=str(engine), operation="ocr_page",
                    duration_seconds=time.monotonic() - ocr_started, paths=[path])
        width, height = image_size(path)
        physical_id = f"{document_id}:p{i}" if document_id else f"{path.stem}:p{i}"
        fingerprint = page_file_fingerprint(path)
        pages.append(Page(i, str(path), width, height, blocks,
                          document_id=document_id,
                          physical_page_id=physical_id,
                          file_fingerprint=fingerprint,
                          ocr_source_fingerprint=fingerprint,
                          page_index=i))
    return pages


def assign_page_identity(pages: Sequence[Page], document_id: str) -> None:
    """Attach immutable document/page identity after legacy page construction."""
    for index, page in enumerate(pages, 1):
        page.document_id = document_id
        page.page_index = index
        page.physical_page_id = f"{document_id}:p{index}"
        current_fingerprint = page_file_fingerprint(Path(page.path))
        if not page.ocr_source_fingerprint:
            page.ocr_source_fingerprint = page.file_fingerprint or current_fingerprint
        page.file_fingerprint = current_fingerprint


def ocr_text(pages: Sequence[Page]) -> str:
    return "\n".join(f"[第 {p.index} 页] " + " ".join(b.text for b in p.ocr) for p in pages)


def compose_tree_llm_evidence(
        evidence: Sequence[Tuple[int, str, str]]) -> Tuple[str, str, Dict[str, int]]:
    """Compose complete, page-aligned LLM evidence with PP-OCR fallback."""
    full_vl_text = "\n\n".join(
        "=== 第 {} 页 ===\n{}".format(
            index,
            vl_text or "[本页 PaddleOCR-VL 不可用，请使用同页 PP-OCRv5 文本]",
        )
        for index, vl_text, _ in evidence
    )
    full_ppocr_text = "\n\n".join(
        "=== 第 {} 页 ===\n{}".format(index, ppocr_text)
        for index, _, ppocr_text in evidence
    )
    audit = {
        "teacher_pages_total": len(evidence),
        "dual_source_pages": sum(
            1 for _, vl_text, ppocr_text in evidence
            if vl_text.strip() and ppocr_text.strip()
        ),
        "ppocr_fallback_pages": sum(
            1 for _, vl_text, ppocr_text in evidence
            if not vl_text.strip() and ppocr_text.strip()
        ),
    }
    return full_vl_text, full_ppocr_text, audit


def _data_url(path: Path) -> str:
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "image/jpeg")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


TEACHER_SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "required": ["document_type", "subject", "questions", "pages", "warnings"],
    "properties": {
        "document_type": {"const": "teacher"}, "subject": {"type": "string"},
        "student_id": {"type": ["string", "null"]},
        "pages": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["page", "image"], "properties": {"page": {"type": "integer"}, "image": {"type": "string"}}}},
        "questions": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "type", "prompt", "standard_answer", "score", "confidence"], "properties": {"id": {"type": "string"}, "type": {"type": "string"}, "prompt": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}}, "standard_answer": {}, "slot_count": {"type": ["integer", "null"], "minimum": 1, "maximum": 100}, "rubric": {"type": ["string", "null"]}, "score": {"type": ["number", "null"]}, "bbox": {"type": ["array", "null"]}, "confidence": {"type": "number"}}}},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}

STUDENT_SCHEMA: Dict[str, Any] = json.loads(json.dumps(TEACHER_SCHEMA).replace('"teacher"', '"student"'))
STUDENT_SCHEMA["properties"]["student_id"] = {"type": ["string", "null"]}
# Extraction is deliberately grading-free: the student model reports physical
# answer evidence only. Scoring and correctness belong to a downstream stage.
STUDENT_SCHEMA["properties"]["questions"]["items"]["required"] = [
    "id", "type", "prompt", "answer", "score", "confidence"
]
STUDENT_SCHEMA["properties"]["questions"]["items"]["properties"].update({
    "answer": {},
    "feedback": {"type": ["string", "null"]},
})


def _teacher_topology(teacher: Dict[str, Any]) -> Dict[str, Any]:
    """Return the stable teacher-owned hierarchy used to constrain students."""
    sections = []
    for section in teacher.get("sections", []) if isinstance(teacher.get("sections"), list) else []:
        if not isinstance(section, dict):
            continue
        questions = []
        for question in section.get("questions", []) if isinstance(section.get("questions"), list) else []:
            if not isinstance(question, dict):
                continue
            items = []
            for item in question.get("items", []) if isinstance(question.get("items"), list) else []:
                if not isinstance(item, dict):
                    continue
                items.append({
                    "item_id": item.get("item_id"),
                    "item_name": item.get("item_name"),
                    "question_text": item.get("question_text"),
                    "item_score": item.get("item_score"),
                    "standard_answer": item.get("standard_answer"),
                    "expected_slot_count": item.get("expected_slot_count"),
                    "slot_count_source": item.get("slot_count_source"),
                    "rubric": item.get("rubric"),
                })
            questions.append({
                "question_id": question.get("question_id"),
                "question_num": question.get("question_num"),
                "question_title": question.get("question_title"),
                "items": items,
            })
        sections.append({"section_id": section.get("section_id"), "section_title": section.get("section_title"), "questions": questions})
    return {"exam_title": teacher.get("exam_title", ""), "sections": sections}


def _prompt(role: str, subject: str, pages: Sequence[Page], teacher: Optional[Dict[str, Any]],
            roi_records: Optional[Sequence[Dict[str, Any]]] = None,
            knowledge_context: str = "") -> str:
    schema = TEACHER_SCHEMA if role == "teacher" else STUDENT_SCHEMA
    context = ""
    if teacher:
        context = (
            "\n教师卷 Golden 拓扑（题号、题干、小题和分值由教师卷唯一决定）：\n"
            + json.dumps(_teacher_topology(teacher), ensure_ascii=False)
            + "\n学生卷只能识别上述题目的 answer、feedback 和 bbox；"
              "不得创建、删除、合并或改写题号、题干及小题。未作答请保留 null。"
        )
    roi_context = ""
    if roi_records:
        roi_context = (
            "\n本次图像全部是已按题目切分的 RoI Patch，不是整页图。"
            "请按输入顺序逐块提取，不得改写学生作答语序。\nRoI 清单："
            + json.dumps(list(roi_records), ensure_ascii=False)
        )
    return (
        f"你是教育文档结构化专家。请从给定的{role}作业照片中提取题目级 JSON。\n"
        "必须只输出一个合法 JSON，不要 Markdown，不要解释。印刷题干、选项、手写内容、红笔批注要区分；看不清时填 null 并在 warnings 说明。\n"
        "每个带（1）（2）或 (1)(2) 的小题必须分别输出为独立 questions 元素，ID 使用 q<大题号>_<小题号>，禁止把多个小题拼进一个 prompt。\n"
        "题号应跨页保持一致；数学公式用 LaTeX 字符串，中文保持原文。教师卷的 standard_answer 是红笔/参考答案；学生卷只提取学生作答，教师批语放 feedback，禁止判断正误或计算分数。\n"
        "每个 questions 元素应输出语义作答点数量 slot_count：多行简答仍计 1，多空填空按独立答案点计数；无法确定时填 null。\n"
        f"学科目录名: {subject}\nOCR 坐标文本:\n{ocr_text(pages)}{context}{roi_context}"
        + (f"\n{knowledge_context}" if knowledge_context else "") + "\n"
        f"JSON Schema:\n{json.dumps(schema, ensure_ascii=False)}"
    )


class StructuredModelResult(dict):
    """Parsed structured output with transport metadata outside JSON keys."""

    def __init__(self, value: Dict[str, Any], response_audit: Optional[Dict[str, Any]] = None):
        super().__init__(value)
        self.response_audit = dict(response_audit or {})


class StructuredOutputError(ValueError):
    """Machine-readable structured-output failure safe to persist in audits."""

    def __init__(self, code: str, detail: str, *, response_audit: Optional[Dict[str, Any]] = None,
                 raw_text: str = ""):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.response_audit = dict(response_audit or {})
        self.raw_text = raw_text


def _extract_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    start = text.find("{")
    if start < 0:
        raise StructuredOutputError("JSON_OBJECT_NOT_FOUND", "模型响应中没有 JSON 对象", raw_text=text)
    decoder = json.JSONDecoder()
    try:
        result, end = decoder.raw_decode(text, start)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(
            "INVALID_JSON",
            "JSON 解析失败: line {} column {} char {}: {}".format(
                exc.lineno, exc.colno, exc.pos, exc.msg),
            raw_text=text,
        ) from exc
    if not isinstance(result, dict):
        raise StructuredOutputError("JSON_ROOT_NOT_OBJECT", "模型 JSON 根节点必须为对象", raw_text=text)
    trailing = text[end:].strip()
    while trailing:
        trailing = re.sub(r"^```(?:json)?\s*|\s*```$", "", trailing,
                          flags=re.I | re.S).strip()
        next_start = trailing.find("{")
        if next_start < 0:
            break
        try:
            extra, next_end = decoder.raw_decode(trailing, next_start)
        except json.JSONDecodeError:
            break
        if isinstance(extra, dict) and extra != result:
            raise StructuredOutputError(
                "MULTIPLE_JSON_OBJECTS", "模型返回多个不一致 JSON 对象", raw_text=text)
        trailing = trailing[next_end:].strip()
    return result


def call_doubao(model: str, api_key: str, prompt: str, paths: Sequence[Path], endpoint: str,
                timeout: int, schema: Dict[str, Any], max_attempts: int = 2) -> Dict[str, Any]:
    """Call Ark's Responses API with Doubao vision inputs."""
    if not api_key or not api_key.strip():
        raise ValueError("未配置 DOUBAO_API_KEY")
    if timeout <= 0:
        raise ValueError("timeout 必须大于 0")
    if max_attempts < 1:
        raise ValueError("max_attempts 必须大于 0")
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    content.extend({"type": "input_image", "image_url": _data_url(path), "detail": "high"} for path in paths)
    payload = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_output_tokens": 12000,
        "text": {"format": {"type": "json_schema", "name": "homework_extraction", "strict": True, "schema": schema}},
    }
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    raw: Dict[str, Any] = {}
    retry_reasons: List[str] = []
    attempts_used = 0
    for attempt in range(1, max_attempts + 1):
        attempts_used = attempt
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = json.loads(response.read().decode())
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt == max_attempts:
                error = RuntimeError(f"视觉模型 HTTP {exc.code}: {body[:500]}")
                error.response_audit = {"attempts": attempts_used,
                                        "retry_reasons": retry_reasons + [f"http_{exc.code}"]}
                raise error from exc
            retry_reasons.append(f"http_{exc.code}")
            LOGGER.warning("vlm_retry attempt=%d status=%d", attempt, exc.code)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == max_attempts:
                error = RuntimeError(f"视觉模型网络请求失败: {exc}")
                error.response_audit = {"attempts": attempts_used,
                                        "retry_reasons": retry_reasons + ["network"]}
                raise error from exc
            retry_reasons.append("network")
            LOGGER.warning("vlm_retry attempt=%d reason=network", attempt)
        time.sleep(min(4.0, 0.5 * (2 ** (attempt - 1))))
    response_audit = {
        "response_id": raw.get("id"),
        "status": raw.get("status"),
        "incomplete_details": raw.get("incomplete_details"),
        "usage": raw.get("usage"),
        "attempts": attempts_used,
        "retry_reasons": retry_reasons,
    }
    text = raw.get("output_text")
    if not text:
        chunks = []
        for item in raw.get("output", []):
            for part in item.get("content", []):
                if part.get("type") in {"output_text", "text"}:
                    chunks.append(part.get("text", ""))
        text = "".join(chunks)
    output_incomplete = raw.get("status") == "incomplete" or any(
        item.get("status") == "incomplete" for item in raw.get("output", [])
        if isinstance(item, dict)
    )
    if output_incomplete:
        raise StructuredOutputError(
            "OUTPUT_INCOMPLETE", "模型输出未完成，需缩小请求范围后重试",
            response_audit=response_audit, raw_text=text,
        )
    if not text:
        raise RuntimeError("豆包 Responses 响应中没有 output_text")
    try:
        parsed = _extract_json(text)
    except StructuredOutputError as exc:
        exc.response_audit.update(response_audit)
        raise
    return StructuredModelResult(parsed, response_audit)


def validate_result(result: Dict[str, Any], role: str) -> List[str]:
    errors = []
    if result.get("document_type") != role:
        errors.append(f"document_type 应为 {role}")
    questions = result.get("questions")
    if not isinstance(questions, list) and isinstance(result.get("sections"), list):
        questions = [question for section in result["sections"] if isinstance(section, dict)
                     for question in section.get("questions", []) if isinstance(question, dict)]
    if not isinstance(questions, list):
        errors.append("questions 必须为数组")
        questions = []
    for i, q in enumerate(questions):
        if not q.get("id") and not q.get("question_id"):
            errors.append(f"questions[{i}].id 缺失")
        required = "standard_answer" if role == "teacher" else "answer"
        items = q.get("items") if isinstance(q.get("items"), list) else [q]
        for item_index, item in enumerate(items):
            if required not in item and required not in q:
                errors.append(f"questions[{i}].items[{item_index}].{required} 缺失")
    return errors


def fallback_result(role: str, subject: str, pages: Sequence[Page], warning: str) -> Dict[str, Any]:
    return {"document_type": role, "subject": subject, "student_id": None, "pages": [{"page": p.index, "image": p.path} for p in pages], "questions": [], "warnings": [warning]}


def load_golden_template(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Load a Golden page template without coupling this project to the prototype tree."""
    if not path:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Golden 模板无法读取: {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("pages"), list):
        raise ValueError("Golden 模板必须是包含 pages 数组的 JSON 对象")
    return data


def _number_from_text(value: Any) -> Optional[int]:
    match = re.search(r"(?:^|[^0-9])(\d{1,3})(?:\s*[\.、．:：\)]|$)", str(value or ""))
    return int(match.group(1)) if match else None


def _number_from_id(value: Any) -> Optional[int]:
    match = re.search(r"(?:^|[^0-9])(\d{1,3})(?:_|$)", str(value or ""))
    return int(match.group(1)) if match else None


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def _positive_int_or_none(value: Any) -> Optional[int]:
    try:
        result = int(value)
        return result if result > 0 else None
    except (TypeError, ValueError):
        return None


def _page_region(page: Page, bbox: Any, text: str = "", confidence: Any = None,
                 golden_order: bool = False, normalized: bool = False,
                 source_size: Optional[Tuple[float, float]] = None) -> Optional[PageRegion]:
    """Convert a bbox to the project's [left, top, right, bottom] convention."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        values = [float(x) for x in bbox[:4]]
    except (TypeError, ValueError):
        return None
    if golden_order:
        # Reference Golden assets use [ymin, xmin, ymax, xmax].
        values = [values[1], values[0], values[3], values[2]]
    width, height = page.width or CANONICAL_PAGE_SIZE[0], page.height or CANONICAL_PAGE_SIZE[1]
    if source_size:
        source_w, source_h = source_size
        values = [values[0] * width / source_w, values[1] * height / source_h,
                  values[2] * width / source_w, values[3] * height / source_h]
    # VLM coordinates are commonly normalized to a 0..1000 canvas. OCR JSON
    # coordinates are left untouched because they are already pixel based.
    if normalized:
        values = [values[0] * width / 1000, values[1] * height / 1000,
                  values[2] * width / 1000, values[3] * height / 1000]
    left, top, right, bottom = values
    left, right = sorted((max(0.0, left), min(float(width), right)))
    top, bottom = sorted((max(0.0, top), min(float(height), bottom)))
    if right <= left or bottom <= top:
        return None
    return PageRegion(page.index, page.path, [round(left, 2), round(top, 2), round(right, 2), round(bottom, 2)], _float_or_none(confidence), text)


def _section_for_type(item_type: str) -> Tuple[str, str]:
    value = str(item_type or "").lower()
    if any(token in value for token in ("choice", "选择", "single", "判断")):
        return "choice", "一、选择题"
    if any(token in value for token in ("fill", "填空")):
        return "fill", "二、填空题"
    if any(token in value for token in ("solve", "calc", "解答", "计算", "证明")):
        return "solve", "三、解答题"
    return "other", "试题"


def _new_sections() -> Dict[str, ExamSection]:
    return {key: ExamSection(f"sec_{key}", title, []) for key, title in (
        ("choice", "一、选择题"), ("fill", "二、填空题"),
        ("solve", "三、解答题"), ("other", "试题"))}


def package_from_golden(template: Optional[Dict[str, Any]], pages: Sequence[Page],
                        role: str, subject: str,
                        student_id: Optional[str] = None) -> Optional[ExamPackage]:
    """Build structure from page/item coordinates and answers in a Golden asset."""
    if not template:
        return None
    template_pages = {int(p.get("page_index", i + 1)): p for i, p in enumerate(template.get("pages", [])) if isinstance(p, dict)}
    sections = _new_sections()
    matched = 0
    for page in pages:
        source = template_pages.get(page.index)
        if not source:
            continue
        for raw in source.get("items", []):
            if not isinstance(raw, dict):
                continue
            item_id = str(raw.get("id", raw.get("item_id", "")))
            q_num = _number_from_id(item_id) or _number_from_text(raw.get("stem_full_text"))
            if not q_num:
                continue
            item_type = str(raw.get("type", ""))
            sec_key, sec_title = _section_for_type(item_type)
            sec = sections[sec_key]
            sec.section_title = sec_title
            q_id = f"q{q_num}"
            question = next((q for q in sec.questions if q.question_id == q_id), None)
            if question is None:
                question = ExamQuestion(q_id, q_num, str(raw.get("stem_full_text", "")).split("\n", 1)[0], [])
                sec.questions.append(question)
            source_w = float(source.get("width", CANONICAL_PAGE_SIZE[0]) or CANONICAL_PAGE_SIZE[0])
            source_h = float(source.get("height", CANONICAL_PAGE_SIZE[1]) or CANONICAL_PAGE_SIZE[1])
            region = _page_region(page, raw.get("bbox"), str(raw.get("stem_full_text", "")), 0.98,
                                  golden_order=True, source_size=(source_w, source_h))
            answer = raw.get("ans") if role == "teacher" else raw.get("student_answer_full_text", raw.get("student_answer"))
            # Typed Golden zones are optional for backward compatibility.  If a
            # zone is absent, derive a conservative partition from the item box;
            # the student page later inherits these coordinates after homography.
            def raw_region(*keys: str) -> Optional[PageRegion]:
                for key in keys:
                    if raw.get(key) is not None:
                        candidate = _page_region(page, raw.get(key), str(raw.get("stem_full_text", "")),
                                                  raw.get("confidence", 0.98), golden_order=True,
                                                  source_size=(source_w, source_h))
                        if candidate:
                            return candidate
                return None

            typed_stem = raw_region("stem_bbox", "stem_box", "question_bbox") or copy.deepcopy(region)
            option_boxes = raw.get("option_bboxes", raw.get("option_boxes", raw.get("options", [])))
            blank_boxes = raw.get("blank_bboxes", raw.get("blank_boxes", raw.get("blanks", [])))
            writing_boxes = raw.get("writing_bboxes", raw.get("writing_boxes", raw.get("writing_box")))
            def raw_regions(value: Any) -> List[PageRegion]:
                values = value if isinstance(value, list) else [value]
                # A single bbox is a flat numeric list, not a list of bboxes.
                if values and all(isinstance(x, (int, float)) for x in values):
                    values = [values]
                values = [box.get("bbox", box.get("box", [])) if isinstance(box, dict) else box for box in values]
                return [candidate for box in values
                        if (candidate := _page_region(page, box, "", raw.get("confidence", 0.98),
                                                      golden_order=True, source_size=(source_w, source_h)))]
            option_regions = raw_regions(option_boxes)
            blank_regions = raw_regions(blank_boxes)
            writing_regions = raw_regions(writing_boxes)
            if region and not any((option_regions, blank_regions, writing_regions)):
                # Golden files that only carry an item bbox still get typed zones.
                left, top, right, bottom = region.bbox
                item_kind = sec_key
                if item_kind == "choice":
                    option_regions = [PageRegion(region.page_index, region.page_file,
                                                  [left, top + (bottom-top)*0.55, right, bottom], region.confidence)]
                elif item_kind == "fill":
                    blank_regions = [PageRegion(region.page_index, region.page_file,
                                                [left, top + (bottom-top)*0.60, right, bottom], region.confidence)]
                else:
                    writing_regions = [PageRegion(region.page_index, region.page_file,
                                                  [left, top + (bottom-top)*0.42, right, bottom], region.confidence)]
            item = ExamItem(
                item_id=item_id or f"{q_id}_1", item_name=str(raw.get("title", item_id or f"第{q_num}题")),
                question_text=str(raw.get("stem_full_text", "")),
                standard_answer=raw.get("ans", raw.get("standard_answer")) if role == "teacher" else None,
                expected_slot_count=_positive_int_or_none(
                    raw.get("slot_count", raw.get("expected_slot_count"))
                ),
                slot_count_source=("golden" if _positive_int_or_none(
                    raw.get("slot_count", raw.get("expected_slot_count"))) else ""),
                item_score=_float_or_none(raw.get("score", raw.get("full_score"))),
                # A Golden file is a reference layout, not a source of truth for
                # an arbitrary student's handwriting. Student answers must come
                # from VLM/OCR extraction for the current page.
                student_answer=answer if role == "teacher" else None,
                answer_regions=[region] if region else [], student_regions=[region] if region and role == "student" else [],
                    eval_status="pending", eval_feedback="", confidence=_float_or_none(raw.get("confidence")) or 1.0,
                    item_type=item_type or "other",
                    rubric=(str(raw.get("rubric")) if raw.get("rubric") is not None else None),
                    stem_region=typed_stem,
                    option_regions=option_regions,
                    blank_regions=blank_regions,
                    writing_regions=writing_regions,
                )
            diagram_values = raw.get("diagram_bboxes", raw.get("diagram_boxes", raw.get("diagrams", [])))
            if isinstance(diagram_values, dict):
                diagram_values = [diagram_values.get("bbox", [])]
            if isinstance(diagram_values, list) and diagram_values and all(isinstance(x, (int, float)) for x in diagram_values):
                diagram_values = [diagram_values]
            item.diagrams = [DiagramRef(title=f"{item.item_name} 图形 {idx}", bbox=diagram.bbox)
                             for idx, box in enumerate(diagram_values or [], 1)
                             if (diagram := _page_region(page, box, "", 0.98, golden_order=True,
                                                         source_size=(source_w, source_h)))]
            item.is_cross_page = bool(raw.get("is_cross_page", False))
            question.items.append(item)
            matched += 1
    if not matched:
        return None
    sections_list = [s for s in sections.values() if s.questions]
    return ExamPackage(
        exam_id="golden_exam", exam_title=str(template.get("title", "")), subject=subject,
        document_type=role, student_id=student_id, total_pages=len(pages),
        page_files=[p.path for p in pages], sections=sections_list,
        total_score=sum((it.item_score or 0) for s in sections_list for q in s.questions for it in q.items) or None,
        warnings=[f"Golden 模板命中 {matched} 个结构化项"],
    )


def _blocks_in_span(blocks: Sequence[OCRBlock], top: float, bottom: float, width: float) -> Tuple[str, List[float], Optional[float]]:
    selected = [b for b in blocks if b.bbox and b.bbox[1] >= top - 8 and b.bbox[1] < bottom]
    selected.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    text = " ".join(b.text.strip() for b in selected if b.text.strip())
    confidence = min((b.confidence for b in selected if b.confidence is not None), default=None)
    return text, [0.0, top, width, bottom], confidence


def heuristic_questions(pages: Sequence[Page], role: str) -> List[ExamSection]:
    """Build a deterministic candidate from page-aware physical evidence."""
    from exam_pipeline.structure_recovery import build_sections
    return build_sections(pages, role)


def _page_exit_text(value: Any) -> str:
    """Convert schema-permitted scalar or structured answers into stable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":")).strip()
        except (TypeError, ValueError):
            pass
    return str(value).strip()


def analyze_page_exit(page: Page, sections: Sequence[ExamSection]) -> PageExitContext:
    """Capture question text, sub-question counts and physical page-end state."""
    questions = _ordered_questions(sections)
    if not questions:
        return PageExitContext(page.index)
    question = questions[-1]
    item = question.items[-1] if question.items else None
    text_candidates = (
        item.student_answer if item else None,
        item.standard_answer if item else None,
        item.question_text if item else None,
        question.question_title,
    )
    text = next((candidate for value in text_candidates
                 if (candidate := _page_exit_text(value))), "")
    incomplete_punctuation = set("，,:：；;、—-")
    incomplete_connectives = ("且", "而", "又", "并且", "由于", "若", "因为", "所以", "如图", "可知", "即", "则", "求", "试求", "解得", "证明", "满足")
    incomplete_math = set("=+-×÷<≥≤/\\")
    declared = re.findall(r"(?:\(|（)(\d+)(?:\)|）)", question.question_title or "")
    declared_count = max((int(value) for value in declared), default=0)
    actual_count = len(question.items)
    last_region_bottom = 0.0
    if item:
        regions = item.student_regions or item.answer_regions
        if regions and len(regions[-1].bbox) >= 4:
            last_region_bottom = float(regions[-1].bbox[3])
    is_incomplete = bool(text and (text[-1] in incomplete_punctuation or text[-1] in incomplete_math or any(text.endswith(c) for c in incomplete_connectives)))
    if declared_count > actual_count:
        is_incomplete = True
    return PageExitContext(
        page_index=page.index, page_file=page.path, last_question_id=question.question_id,
        last_question_num=question.question_num,
        last_question_title=question.question_title, last_text_tail=text[-80:],
        is_semantically_incomplete=is_incomplete,
        last_item_id=item.item_id if item else "", last_item_name=item.item_name if item else "",
        declared_sub_count=declared_count, actual_sub_count=actual_count,
        last_region_bottom=last_region_bottom,
    )


def _ordered_questions(sections: Sequence[ExamSection]) -> List[ExamQuestion]:
    """Flatten sections in physical reading order for cross-page decisions."""
    questions = [q for section in sections for q in section.questions]
    def key(question: ExamQuestion) -> Tuple[float, int]:
        regions = [r for item in question.items for r in (item.student_regions or item.answer_regions) if len(r.bbox) >= 4]
        top = min((float(r.bbox[1]) for r in regions), default=float("inf"))
        return top, question.question_num
    return sorted(questions, key=key)


def should_stitch_page(exit_ctx: PageExitContext, next_sections: Sequence[ExamSection]) -> bool:
    """Apply explicit same-number, sub-question and continuation-starter rules."""
    questions = _ordered_questions(next_sections)
    if not exit_ctx.last_question_num or not questions:
        return False
    first = questions[0]
    if first.question_num == exit_ctx.last_question_num:
        return True
    text = first.question_title.strip()
    sub_match = re.match(r"^[（(](\d+)[）)]", text)
    if sub_match and (int(sub_match.group(1)) > 1 or exit_ctx.is_semantically_incomplete):
        return True
    starters = ("解得", "综上所述", "综上", "故：", "故", "证明：", "证明", "如图", "代入得", "又因为", "所以")
    if exit_ctx.is_semantically_incomplete and text.startswith(starters):
        return True
    return exit_ctx.is_semantically_incomplete and not re.search(r"(?:^|[^0-9])\d{1,3}[.、．:：)]", text)


def stitch_page_sections(package: ExamPackage, next_sections: List[ExamSection], exit_ctx: PageExitContext) -> bool:
    """Attach the first continuation item to the previous logical question."""
    previous = _ordered_questions(package.sections)
    incoming = _ordered_questions(next_sections)
    if not previous or not incoming:
        return False
    last = previous[-1]
    if not should_stitch_page(exit_ctx, next_sections):
        return False
    first = incoming.pop(0)
    for item in first.items:
        item.is_cross_page = True
        last.items.append(item)
        if item.answer_regions and item.student_regions:
            # Keep both physical coordinate streams, as in the prototype's
            # multi-region Item contract.
            for region in item.student_regions:
                if region not in item.answer_regions:
                    item.answer_regions.append(region)
    last_title = first.question_title.strip()
    if last_title and last_title not in last.question_title:
        last.question_title = f"{last.question_title}\n{last_title}".strip()
    # Remove the consumed first question from its source section. Empty source
    # sections are discarded before final serialization.
    for section in next_sections:
        if first in section.questions:
            section.questions.remove(first)
            break
    return True


def package_from_result(result: Dict[str, Any], pages: Sequence[Page], role: str, subject: str,
                        student_id: Optional[str] = None) -> ExamPackage:
    """Convert the legacy flat VLM response (or an already nested response) to ExamPackage."""
    sections = _new_sections()
    preserve_sections = bool(result.get("_preserve_sections"))
    raw_sections = result.get("sections") if isinstance(result.get("sections"), list) else None
    raw_questions: List[Dict[str, Any]] = []
    if raw_sections:
        for sec in raw_sections:
            for q in sec.get("questions", []) if isinstance(sec, dict) else []:
                q = dict(q)
                q["_section_title"] = sec.get("section_title", "") if isinstance(sec, dict) else ""
                q["_section_id"] = sec.get("section_id", "") if isinstance(sec, dict) else ""
                raw_questions.append(q)
    else:
        raw_questions = [q for q in result.get("questions", []) if isinstance(q, dict)]
    for raw_q in raw_questions:
        q_num = int(raw_q.get("question_num") or _number_from_id(raw_q.get("id", raw_q.get("question_id"))) or _number_from_text(raw_q.get("prompt", raw_q.get("question_title"))) or 0)
        q_id = str(raw_q.get("question_id") or raw_q.get("id") or f"q{q_num or len(raw_questions)}")
        q_id = q_id if q_id.startswith("q") else f"q{q_num}" if q_num else q_id
        item_type = str(raw_q.get("type", ""))
        sec_key, sec_title = _section_for_type(item_type)
        if preserve_sections and raw_q.get("_section_id"):
            source_key = "source:{}".format(raw_q["_section_id"])
            if source_key not in sections:
                sections[source_key] = ExamSection(
                    str(raw_q["_section_id"]),
                    str(raw_q.get("_section_title") or raw_q["_section_id"]),
                    [],
                )
            sec = sections[source_key]
        else:
            sec = sections[sec_key]
        if raw_q.get("_section_title") and not preserve_sections:
            sec_title = str(raw_q["_section_title"])
            sec.section_title = sec_title
        items_raw = raw_q.get("items") if isinstance(raw_q.get("items"), list) else [raw_q]
        question = ExamQuestion(q_id, q_num, str(raw_q.get("question_title") or raw_q.get("prompt") or f"第{q_num}题"), [])
        for idx, raw_i in enumerate(items_raw, 1):
            if not isinstance(raw_i, dict):
                continue
            item_id = str(raw_i.get("item_id") or raw_i.get("id") or (f"{q_id}_{idx}" if len(items_raw) > 1 else q_id))
            item_type = str(raw_i.get("type", item_type or "other"))
            page_num = int(raw_i.get("page", raw_q.get("page", 1)) or 1)
            page = pages[min(max(page_num - 1, 0), len(pages) - 1)] if pages else Page(1, "", None, None, [])
            raw_bbox = raw_i.get("bbox", raw_q.get("bbox"))
            normalized_bbox = isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4 and max((abs(float(x)) for x in raw_bbox[:4]), default=0) <= 1000
            region = _page_region(page, raw_bbox, str(raw_i.get("prompt", raw_i.get("question_text", raw_q.get("prompt", "")))), raw_i.get("confidence", raw_q.get("confidence")), normalized=normalized_bbox)
            def result_regions(value: Any) -> List[PageRegion]:
                values = value if isinstance(value, list) else [value]
                if values and all(isinstance(x, (int, float)) for x in values):
                    values = [values]
                values = [x.get("bbox", x.get("box", [])) if isinstance(x, dict) else x for x in values]
                return [candidate for box in values
                        if (candidate := _page_region(page, box, "", raw_i.get("confidence"), normalized=normalized_bbox))]
            explicit_stem = result_regions(raw_i.get("stem_bbox", raw_i.get("stem_box")))
            explicit_options = result_regions(raw_i.get("option_bboxes", raw_i.get("option_boxes", raw_i.get("options"))))
            explicit_blanks = result_regions(raw_i.get("blank_bboxes", raw_i.get("blank_boxes", raw_i.get("blanks"))))
            explicit_writing = result_regions(raw_i.get("writing_bboxes", raw_i.get("writing_boxes", raw_i.get("writing_box"))))
            item = ExamItem(
                item_id=item_id, item_name=str(raw_i.get("item_name") or raw_i.get("title") or item_id),
                question_text=str(raw_i.get("question_text", raw_i.get("prompt", raw_q.get("prompt", "")))),
                standard_answer=raw_i.get("standard_answer", raw_q.get("standard_answer")) if role == "teacher" else None,
                expected_slot_count=_positive_int_or_none(
                    raw_i.get("slot_count", raw_i.get("expected_slot_count",
                                   raw_q.get("slot_count", raw_q.get("expected_slot_count"))))
                ),
                slot_count_source=("vlm" if _positive_int_or_none(
                    raw_i.get("slot_count", raw_i.get("expected_slot_count",
                                   raw_q.get("slot_count", raw_q.get("expected_slot_count"))))) else ""),
                item_score=_float_or_none(raw_i.get("score", raw_i.get("item_score", raw_q.get("score")))),
                student_answer=raw_i.get("answer", raw_i.get("student_answer")) if role == "student" else None,
                student_score=_float_or_none(raw_i.get("student_score")),
                answer_regions=[region] if region else [], student_regions=[region] if region and role == "student" else [],
                eval_status=str(raw_i.get("eval_status", raw_q.get("eval_status", "pending"))),
                eval_feedback=str(raw_i.get("eval_feedback", raw_i.get("feedback", raw_q.get("feedback", ""))) or ""),
                confidence=_float_or_none(raw_i.get("confidence", raw_q.get("confidence"))) or 1.0,
                item_type=item_type or "other",
                rubric=(str(raw_i.get("rubric")) if raw_i.get("rubric") is not None else None),
                is_correct=(raw_i.get("is_correct") if isinstance(raw_i.get("is_correct"), bool) else None),
                stem_region=(explicit_stem[0] if explicit_stem else (copy.deepcopy(region) if region else None)),
                option_regions=([PageRegion(region.page_index, region.page_file,
                                            [region.bbox[0], region.bbox[1] + (region.bbox[3]-region.bbox[1])*0.55, region.bbox[2], region.bbox[3]], region.confidence)]
                                if region and sec_key == "choice" else []),
                blank_regions=([PageRegion(region.page_index, region.page_file,
                                           [region.bbox[0], region.bbox[1] + (region.bbox[3]-region.bbox[1])*0.60, region.bbox[2], region.bbox[3]], region.confidence)]
                               if region and sec_key == "fill" else []),
                writing_regions=([PageRegion(region.page_index, region.page_file,
                                             [region.bbox[0], region.bbox[1] + (region.bbox[3]-region.bbox[1])*0.42, region.bbox[2], region.bbox[3]], region.confidence)]
                                 if region and sec_key in {"solve", "other"} else []),
            )
            if explicit_options:
                item.option_regions = explicit_options
            if explicit_blanks:
                item.blank_regions = explicit_blanks
            if explicit_writing:
                item.writing_regions = explicit_writing
            diagram_values = raw_i.get("diagram_boxes", raw_i.get("diagrams", raw_q.get("diagram_boxes", [])))
            if isinstance(diagram_values, dict):
                diagram_values = [diagram_values.get("bbox", [])]
            if isinstance(diagram_values, list) and diagram_values and all(isinstance(x, (int, float)) for x in diagram_values):
                diagram_values = [diagram_values]
            item.diagrams = [DiagramRef(title=f"{item.item_name} 图形 {n}", bbox=diagram.bbox)
                             for n, box in enumerate(diagram_values or [], 1)
                             if (diagram := _page_region(page, box, "", raw_i.get("confidence"), normalized=normalized_bbox))]
            question.items.append(item)
        if question.items:
            sec.questions.append(question)
    sections_list = [s for s in sections.values() if s.questions]
    return ExamPackage(
        exam_id=str(result.get("exam_id", "")), exam_title=str(result.get("exam_title", result.get("title", ""))),
        subject=str(result.get("subject", subject)), document_type=role,
        student_id=result.get("student_id", student_id), total_pages=len(pages), page_files=[p.path for p in pages],
        sections=sections_list, total_score=_float_or_none(result.get("total_score")),
        warnings=list(result.get("warnings", [])) if isinstance(result.get("warnings"), list) else [],
    )


def reconcile_exam_package(package: ExamPackage) -> List[str]:
    """Merge duplicate cross-page questions and validate sequence and score conservation."""
    warnings: List[str] = []
    by_num: Dict[str, ExamQuestion] = {}
    last_seen_num = 0
    for section in package.sections:
        repaired: List[ExamQuestion] = []
        for question in section.questions:
            if question.question_num <= 0 or not question.question_title.strip():
                if by_num:
                    target = list(by_num.values())[-1]
                    target.items.extend(question.items)
                    for item in question.items:
                        item.is_cross_page = True
                    warnings.append("发现无题号的跨页小题，已挂接到上一道大题")
                continue
            if question.question_id in by_num:
                target = by_num[question.question_id]
                existing = {item.item_id: item for item in target.items}
                for item in question.items:
                    if item.item_id in existing:
                        for region in item.answer_regions:
                            if region not in existing[item.item_id].answer_regions:
                                existing[item.item_id].answer_regions.append(region)
                        for region in item.student_regions:
                            if region not in existing[item.item_id].student_regions:
                                existing[item.item_id].student_regions.append(region)
                        if item.student_answer and item.student_answer != existing[item.item_id].student_answer:
                            existing[item.item_id].student_answer = f"{existing[item.item_id].student_answer or ''}\n{item.student_answer}".strip()
                    else:
                        target.items.append(item)
                    item.is_cross_page = True
                for item in target.items:
                    item.is_cross_page = True if len(item.answer_regions) > 1 else item.is_cross_page
                warnings.append(f"题号 Q{question.question_num} 在多页出现，已合并为一个逻辑题目")
                continue
            if last_seen_num and question.question_num < last_seen_num:
                warnings.append(
                    f"题号非单调递增：Q{last_seen_num} 后出现 Q{question.question_num}，请记录未解决状态 核对"
                )
            by_num[question.question_id] = question
            last_seen_num = question.question_num
            for index, item in enumerate(question.items, 1):
                if not item.item_id:
                    item.item_id = f"{question.question_id}_{index}"
                if not item.item_name:
                    item.item_name = f"{question.question_num}.({index})"
            repaired.append(question)
        section.questions = repaired
    nums = sorted({q.question_num for q in by_num.values()})
    if nums and nums != list(range(nums[0], nums[-1] + 1)):
        warnings.append(f"题号序列存在缺口: {nums}")
    # Page-local extraction appends sections on every page. Consolidate same
    # section IDs so the final tree is one stable section stream, as in the
    # prototype's assembled package; discard containers emptied by merges.
    merged_sections: Dict[str, ExamSection] = {}
    for section in package.sections:
        if not section.questions:
            continue
        existing = merged_sections.get(section.section_id)
        if existing is None:
            merged_sections[section.section_id] = section
        else:
            existing.questions.extend(section.questions)
    package.sections = list(merged_sections.values())
    from exam_pipeline.scoring import balance_score_tree
    warnings.extend(balance_score_tree(package))
    package.warnings.extend(warnings)
    return warnings


def _prepare_paths(paths: Sequence[Path], output: Path, subject: str, role: str) -> Tuple[List[Path], List[Dict[str, Any]], List[str]]:
    normalized: List[Path] = []
    metadata: List[Dict[str, Any]] = []
    warnings: List[str] = []
    target_dir = output / "normalized" / re.sub(r"[^\w.-]+", "_", subject) / role / re.sub(r"[^\w.-]+", "_", paths[0].parent.name)
    for index, source in enumerate(paths, 1):
        target = target_dir / f"page_{index:02d}.jpg"
        try:
            metadata.append(preprocess_page(source, target))
            normalized.append(target)
        except Exception as exc:
            warnings.append(f"页面预处理失败 {source.name}: {exc}")
            normalized.append(source)
    return normalized, metadata, warnings


def process(dataset: Path, output: Path, model: str, api_key: Optional[str], endpoint: str, ocr_engine: str, ocr_json: Optional[Path], dry_run: bool, include_scan: bool, language: str, timeout: int, golden: Optional[Path] = None, limit: Optional[int] = None, local_ocr: bool = True, feedback_db: Optional[Path] = None, grounding_kb: Optional[Path] = None, paddle_model_tier: Optional[str] = None, subject_filter: Optional[str] = None, student_id_filter: Optional[str] = None, exam_tree_overrides: Optional[Path] = None, structure_vlm: str = "auto", paddleocr_vl_endpoint: Optional[str] = None, paddleocr_vl_model: str = PADDLEOCR_VL_MODEL, paddleocr_vl_api_key: Optional[str] = None, tree_llm_endpoint: Optional[str] = None, tree_llm_model: str = TREE_LLM_MODEL, tree_llm_api_key: Optional[str] = None, tree_llm_use_doubao: bool = False, slot_vlm_verify: bool = False, production_mode: bool = False, slot_semantics: str = "auto") -> Dict[str, Any]:
    global PADDLE_MODEL_TIER, PADDLE_DET_MODEL, PADDLE_REC_MODEL, _PADDLE_OCR
    if paddle_model_tier is not None:
        requested_tier = str(paddle_model_tier).strip().lower()
        if requested_tier not in {"mobile", "server"}:
            raise ValueError(f"不支持的 PaddleOCR 模型规格: {paddle_model_tier}")
        requested_det = f"PP-OCRv5_{requested_tier}_det"
        requested_rec = f"PP-OCRv5_{requested_tier}_rec"
        if (PADDLE_DET_MODEL, PADDLE_REC_MODEL) != (requested_det, requested_rec):
            _PADDLE_OCR = None
        PADDLE_MODEL_TIER = requested_tier
        PADDLE_DET_MODEL = requested_det
        PADDLE_REC_MODEL = requested_rec
    from exam_pipeline.golden import GoldenTemplateService
    from exam_pipeline.exam_tree import ExamTreeService, resolve_override
    from exam_pipeline.knowledge import GroundingKnowledgeBase
    from exam_pipeline.structure_vlm import PaddleOCRVLStructureClient
    from exam_pipeline.tree_llm import ExamTreeLLMClient
    from exam_pipeline.performance import PerformanceCollector, set_active_collector

    performance_collector = PerformanceCollector()
    set_active_collector(performance_collector)

    requested_structure_vlm = str(structure_vlm or "auto").strip().lower()
    if requested_structure_vlm not in {"auto", "paddleocr-vl", "doubao", "none"}:
        raise ValueError(f"不支持的结构模型: {structure_vlm}")
    paddleocr_vl_endpoint = str(paddleocr_vl_endpoint or PADDLEOCR_VL_ENDPOINT or "").strip()
    if requested_structure_vlm == "auto":
        active_structure_vlm = (
            "doubao" if api_key
            else "paddleocr-vl" if paddleocr_vl_endpoint
            else "none"
        )
    else:
        active_structure_vlm = requested_structure_vlm
    if not dry_run and active_structure_vlm == "paddleocr-vl" and not paddleocr_vl_endpoint:
        raise ValueError(
            "--structure-vlm paddleocr-vl 需要 --paddleocr-vl-endpoint "
            "或 PADDLEOCR_VL_ENDPOINT"
        )
    if not dry_run and active_structure_vlm == "doubao" and not api_key:
        raise ValueError("--structure-vlm doubao 需要 DOUBAO_API_KEY 或 --api-key")
    paddleocr_vl_client = (
        PaddleOCRVLStructureClient(
            endpoint=paddleocr_vl_endpoint,
            model=paddleocr_vl_model,
            api_key=str(paddleocr_vl_api_key or ""),
            timeout=timeout,
        ) if active_structure_vlm == "paddleocr-vl" and paddleocr_vl_endpoint else None
    )
    structure_request = None
    if not dry_run and active_structure_vlm == "doubao":
        structure_request = lambda prompt, images, schema: call_doubao(
            model, str(api_key or ""), prompt, images, endpoint, timeout, schema)
    elif not dry_run and active_structure_vlm == "paddleocr-vl" and paddleocr_vl_client:
        structure_request = lambda prompt, images, schema: paddleocr_vl_client.analyze(prompt, images)
    structure_base_request = structure_request
    if structure_request is not None:
        structure_request = performance_collector.instrument(
            structure_request, stage="question_tree_generation",
            provider=active_structure_vlm, operation="structure_request")
    if tree_llm_use_doubao:
        tree_llm_endpoint = endpoint
        tree_llm_model = model
        tree_llm_api_key = api_key
        if not dry_run and not str(api_key or "").strip():
            raise ValueError("--tree-llm-use-doubao 需要 DOUBAO_API_KEY 或 --api-key")
    tree_llm_endpoint = str(tree_llm_endpoint or TREE_LLM_ENDPOINT or "").strip()
    tree_llm_model = str(tree_llm_model or TREE_LLM_MODEL or "").strip()
    if tree_llm_endpoint and not tree_llm_model:
        raise ValueError("配置 TREE_LLM_ENDPOINT 时必须同时配置 TREE_LLM_MODEL")
    tree_llm_client = (
        ExamTreeLLMClient(
            endpoint=tree_llm_endpoint,
            model=tree_llm_model,
            api_key=str(tree_llm_api_key or TREE_LLM_API_KEY or ""),
            timeout=timeout,
        ) if tree_llm_endpoint else None
    )
    # Visual models assign meaning and candidate IDs; geometry stays local.
    from exam_pipeline.slot_semantics import VisualSlotSemanticService
    requested_semantics = str(slot_semantics or "auto").lower()
    if requested_semantics not in {"auto", "doubao", "paddleocr-vl", "none"}:
        raise ValueError("不支持的槽位语义模型: " + requested_semantics)
    if slot_vlm_verify and requested_semantics == "none":
        raise ValueError("--slot-vlm-verify 已改为语义候选选择，不能与 --slot-semantics none 同时使用")
    semantic_provider = requested_semantics
    if semantic_provider == "auto":
        semantic_provider = "doubao" if api_key else "paddleocr-vl" if paddleocr_vl_endpoint else "none"
    semantic_request = None
    if not dry_run and semantic_provider == "doubao":
        if not api_key:
            raise ValueError("--slot-semantics doubao 需要 DOUBAO_API_KEY")
        semantic_request = lambda prompt, paths, schema: call_doubao(
            model, str(api_key), prompt, paths, endpoint, timeout, schema)
    elif not dry_run and semantic_provider == "paddleocr-vl":
        if not paddleocr_vl_endpoint:
            raise ValueError("--slot-semantics paddleocr-vl 需要 PADDLEOCR_VL_ENDPOINT")
        semantic_client = paddleocr_vl_client or PaddleOCRVLStructureClient(
            endpoint=paddleocr_vl_endpoint, model=paddleocr_vl_model,
            api_key=str(paddleocr_vl_api_key or ""), timeout=timeout)
        semantic_request = lambda prompt, paths, schema: semantic_client.analyze(prompt, paths)
    if slot_vlm_verify and semantic_request is None and not dry_run:
        raise ValueError("--slot-vlm-verify 需要可用的视觉模型配置")
    semantic_base_request = semantic_request
    if semantic_base_request is not None:
        semantic_request = performance_collector.instrument(
            semantic_base_request, stage="slot_semantic_proposal",
            provider=semantic_provider, operation="semantic_request")
    semantic_service = VisualSlotSemanticService(
        semantic_request, semantic_provider,
        ocr_engine=ocr_engine, language=language)
    from exam_pipeline.question_localization import VisualQuestionLocalizer
    question_localization_base = semantic_base_request
    question_localization_request = (
        performance_collector.instrument(
            question_localization_base, stage="question_localization",
            provider=semantic_provider, operation="question_localization_request")
        if question_localization_base is not None else structure_request
    )
    question_localizer = VisualQuestionLocalizer(
        question_localization_request,
        semantic_provider if semantic_request else active_structure_vlm,
    )
    from exam_pipeline.answer_vision import AnswerVisionClient
    answer_request = (
        performance_collector.instrument(
            semantic_base_request, stage="answer_recognition",
            provider=semantic_provider, operation="answer_transcription_request")
        if semantic_base_request is not None else None
    )
    answer_vision_client = (
        AnswerVisionClient(answer_request) if answer_request is not None and semantic_provider == "doubao"
        else paddleocr_vl_client
    )
    from exam_pipeline.visual_extraction import VisualExamExtractionService
    visual_extraction_base = semantic_base_request
    visual_extraction_provider = semantic_provider
    visual_extraction_request = (
        performance_collector.instrument(
            visual_extraction_base, stage="visual_page_extraction",
            provider=visual_extraction_provider, operation="page_joint_extraction")
        if visual_extraction_base is not None else None
    )
    visual_extraction_service = VisualExamExtractionService(
        visual_extraction_request, visual_extraction_provider
    )
    vlm_only = bool(
        not dry_run and structure_request is not None
        and semantic_base_request is not None
        and visual_extraction_request is not None
    )
    if production_mode and not local_ocr and not vlm_only:
        raise ValueError(
            "--production without local OCR requires configured structure and slot VLMs"
        )

    output.mkdir(parents=True, exist_ok=True)
    cache_base = Path(os.getenv(
        "EXAM_PIPELINE_CACHE_DIR",
        str(Path(__file__).resolve().parent / ".exam_pipeline_cache"),
    )).expanduser().resolve()
    dataset_cache_id = hashlib.sha256(str(dataset.resolve()).encode("utf-8")).hexdigest()[:16]
    cache_root = cache_base / "datasets" / dataset_cache_id
    cache_root.mkdir(parents=True, exist_ok=True)
    grounding_policy = (
        GroundingKnowledgeBase.load(grounding_kb) if grounding_kb else None
    )
    grounding_runtime_audit = ({
        "version": grounding_policy.version,
        "source_path": grounding_policy.source_path,
        "source_sha256": grounding_policy.source_sha256,
        "active_rules": [],
        "runtime_role": ("disabled_in_vlm_only_path" if vlm_only
                         else "question_layout_only"),
        "slot_coordinate_authority": ("vlm_original_page_pixels" if vlm_only
                                      else "ocr_boxes_only"),
        "legacy_ink_rules_enabled": False,
    } if grounding_policy else {})
    ocr_map = read_ocr_jsonl(ocr_json)
    all_docs = discover_documents(dataset, include_scan)
    docs = list(all_docs)
    if subject_filter:
        docs = [entry for entry in docs if entry[0] == subject_filter]
        if not docs:
            raise ValueError(f"未找到学科: {subject_filter}")
    if student_id_filter:
        # 支持过滤教师卷（student_id_filter='teacher'）或学生卷
        matched_teachers = [
            entry for entry in docs
            if entry[1] == "teacher" and entry[2][0].parent.name == student_id_filter
        ]
        matched_students = [
            entry for entry in docs
            if entry[1] == "student" and entry[2][0].parent.name == student_id_filter
        ]

        if matched_teachers:
            # 仅提取教师卷
            docs = matched_teachers
        elif matched_students:
            # 提取学生卷及对应学科的教师卷
            target_subjects = {entry[0] for entry in matched_students}
            teachers_for_targets = [
                entry for entry in docs
                if entry[1] == "teacher" and entry[0] in target_subjects
            ]
            docs = teachers_for_targets + matched_students
        else:
            scope = f"学科 {subject_filter} 中" if subject_filter else ""
            raise ValueError(f"{scope}未找到学生卷或教师卷: {student_id_filter}")
    teacher_subjects = {subject for subject, role, _ in docs if role == "teacher"}
    golden_data = load_golden_template(golden) if golden else None
    teachers: Dict[str, Dict[str, Any]] = {}
    teacher_packages: Dict[str, ExamPackage] = {}
    teacher_pages: Dict[str, List[Page]] = {}
    # Cohort print/handwriting templates were removed from the production
    # path. Keep the input flag for API/CLI compatibility and audit that it
    # was ignored; no template is generated, loaded, or passed downstream.
    exam_trees: Dict[str, Dict[str, Any]] = {}
    applied_tree_overrides: Dict[str, Dict[str, Any]] = {}
    batch_audit_pages: Dict[str, List[Dict[str, Any]]] = {}
    golden_service = GoldenTemplateService()
    started_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    runtime_pipeline_order = ([
        "document_discovery",
        "image_preprocessing",
        "question_tree_generation",
        "visual_page_extraction",
        "answer_recognition",
        "visual_page_retry",
        "diagram_asset_export",
        "slot_visualization",
        "result_output",
    ] if vlm_only else [
        "document_discovery",
        "image_preprocessing",
        "question_tree_generation",
        "question_localization",
        "slot_semantic_proposal",
        "ocr_vlm_cross_validation",
        "ocr_coordinate_grounding",
        "answer_recognition",
        "automatic_repair",
        "diagram_asset_export",
        "slot_visualization",
        "result_output",
    ])
    manifest = {
        "schema_version": "exam_manifest.v4",
        "status": "RUNNING",
        "started_at": started_at,
        "dataset": str(dataset),
        "documents": [],
        "errors": [],
        "generated_goldens": {},
        "exam_trees": {},
        "output_contract": "ExamPackage(metadata, sections, roi_patches, slots, diagrams, page_files)",
        "canonical_canvas": {"width": 1654, "height": 2338},
        "architecture_spec": "PRODUCTION_PIPELINE_SPEC.md",
        "runtime": {
            "pipeline_version": "2.12.0",
            "pipeline_order": runtime_pipeline_order,
            "recognition_mode": "vlm_only" if vlm_only else "legacy_ocr_compatible",
            "production_mode": bool(production_mode),
            "model": model,
            "endpoint": endpoint,
            "ocr_engine": "disabled_in_vlm_only_path" if vlm_only else ocr_engine,
            "paddle_models": {
                "tier": PADDLE_MODEL_TIER,
                "detection": PADDLE_DET_MODEL,
                "recognition": PADDLE_REC_MODEL,
            },
            "structure_vlm": {
                "requested": requested_structure_vlm,
                "active": active_structure_vlm,
                "model": (paddleocr_vl_model if active_structure_vlm == "paddleocr-vl" else model
                          if active_structure_vlm == "doubao" else None),
                "endpoint": (paddleocr_vl_endpoint if active_structure_vlm == "paddleocr-vl" else endpoint
                             if active_structure_vlm == "doubao" else None),
                "scope": "lightweight_whole_document_topology_then_page_text_enrichment",
                "primary_evidence": "all_original_pages",
                "proposal_stage": "pre_ocr_original_images",
                "validation": ("vlm_schema_page_coverage" if vlm_only else
                               "post_proposal_ocr_anchors_reading_order_subquestions_pages"),
                "max_semantic_attempts": 2,
                "page_fallback": "bounded_per_page_topology_and_text_recovery",
                "id_authority": "local_deterministic",
                "response_diagnostics": "status_incomplete_details_usage_parse_position_raw_artifact",
                "geometry_authority": ("vlm_original_page_pixels" if vlm_only else
                                       "ppocrv5_and_deterministic_grounding"),
                "model_geometry_policy": "final" if vlm_only else "discard",
            },
            "tree_llm": {
                "enabled": False,
                "configured": bool(tree_llm_client),
                "disabled_reason": "superseded_by_whole_document_visual_tree",
                "provider": "doubao" if tree_llm_use_doubao else "custom",
                "protocol": tree_llm_client.protocol if tree_llm_client else None,
                "model": tree_llm_model or None,
                "endpoint": tree_llm_endpoint or None,
                "scope": "teacher_semantic_tree_only",
                "invocation_policy": "teacher_document_once_students_reuse_locked_tree",
                "geometry_authority": "ppocrv5_and_deterministic_grounding",
                "candidate_gate": "ocr_question_number_overlap_and_contract_validation",
                "fallback": "deterministic_ocr_tree",
                "accepted_teacher_documents": 0,
                "fallback_teacher_documents": 0,
                "teacher_pages_total": 0,
                "dual_source_pages": 0,
                "ppocr_fallback_pages": 0,
            },
                "slot_semantics": {
                "requested": requested_semantics, "provider": semantic_provider,
                "enabled": bool(semantic_request),
                "authority": ("question_slots_coordinates_and_answers" if vlm_only else
                              "logical_slots_and_search_proposals_only"),
                "coordinate_authority": ("vlm_original_page_pixels" if vlm_only else
                                         "current_page_ocr_boxes"),
                "max_semantic_attempts_per_item": 2 if vlm_only else 1,
                "batch_size": "one_physical_page_all_items",
                "batch_policy": "one_page_batch_then_item_retry",
                "candidate_visual_review": False,
                "answer_visual_transcription": bool(answer_vision_client),
                "answer_authority": ("vlm_original_page" if vlm_only else
                                     "vlm_primary_ocr_auxiliary"),
                "fallback": ("page_local_vlm_retry_then_unresolved" if vlm_only else
                             "unconfigured_or_missing_ocr_coordinates_remain_unresolved"),
                "ocr_used": False if vlm_only else None,
            },
            "question_localization": {
                "enabled": bool(question_localization_request),
                "provider": semantic_provider if semantic_request else active_structure_vlm,
                "stage": "visual_page_extraction" if vlm_only else "pre_ocr",
                "authority": ("vlm_final_question_region" if vlm_only else
                              "question_search_context_only"),
                "coordinate_authority": ("vlm_original_page_pixels" if vlm_only else
                                         "post_ocr_coordinate_grounding"),
                "fallback": "page_local_vlm_retry" if vlm_only else "declared_page_context",
            },
            "slot_vlm_verification": {"enabled": False, "replaced_by": "slot_semantics"},
            "language": language,
            "local_ocr": False if vlm_only else local_ocr,
            "dry_run": dry_run,
            "feedback_db": str(feedback_db) if feedback_db else None,
            "grounding_knowledge": (
                grounding_runtime_audit or None
            ),
            "ink_detection": {"enabled": False,
                              "reason": ("vlm_direct_original_page_extraction" if vlm_only
                                         else "coordinates_from_ocr_content_from_vlm")},
            "cache_policy": {
                "root": str(cache_root),
                "dataset_cache_id": dataset_cache_id,
                "exam_tree": ("teacher_fingerprint_locked_vlm_only_v1" if vlm_only else
                              "teacher_fingerprint_locked"),
                "ocr": "disabled" if vlm_only else "source_page_fingerprint",
                "roi": "question_topology_fingerprint",
                "retry_scope": "failed_slot_or_item_only",
            },
            "input_filter": {
                "subject": subject_filter,
                "student_id": student_id_filter,
            },
            "exam_tree_overrides": str(exam_tree_overrides) if exam_tree_overrides else None,
        },
    }
    # Teacher pages establish question ids/answers used to align student pages.
    docs.sort(key=lambda item: (item[0], 0 if item[1] == "teacher" else 1, natural_key(item[2][0].parent)))
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit 必须是大于 0 的整数")
        docs = docs[:limit]
        manifest["limit"] = limit
        manifest["total_documents_available"] = len(discover_documents(dataset, include_scan))
    for subject, role, paths in docs:
        doc_id = f"{subject}__{role}__{paths[0].parent.name}"
        performance_collector.set_document(doc_id)
        safe_doc_id = re.sub(r"[^\w.-]+", "_", doc_id)
        document_started = time.monotonic()
        stage = "input_validation"
        document_audit: Optional[Dict[str, Any]] = None
        roi_records: List[Dict[str, Any]] = []
        subitem_stats: Dict[str, int] = {}
        phase_trace: List[Dict[str, Any]] = []
        phase_started: Dict[str, float] = {}

        def record_phase(name: str, status: str = "COMPLETED", **details: Any) -> None:
            now = time.monotonic()
            event = {
                "phase": name,
                "status": status,
                "at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                **details,
            }
            if status == "STARTED":
                phase_started[name] = now
            elif name in phase_started:
                event["duration_seconds"] = round(now - phase_started.pop(name), 3)
            phase_trace.append(event)
        record_phase("document_discovery", "COMPLETED", pages=len(paths),
                     source_files=[str(path) for path in paths])
        try:
            LOGGER.info("document_started id=%s role=%s pages=%d", doc_id, role, len(paths))
            fingerprint = input_fingerprint(paths)
            if not dry_run and role == "student" and subject not in teacher_subjects and not golden_data:
                raise ValueError(f"学科 {subject} 没有教师卷，无法生成 Golden 并处理学生卷")
            stage = "normalization"
            record_phase("image_preprocessing", "STARTED")
            page_paths, preprocessing, prep_warnings = (list(paths), [], []) if dry_run else _prepare_paths(paths, output, subject, role)
            record_phase("image_preprocessing", "COMPLETED", pages=len(page_paths), artifacts=preprocessing)
            registration_meta: List[Dict[str, Any]] = []
            if (not dry_run and not vlm_only and role == "student"
                    and subject in teacher_pages):
                stage = "registration"
                from exam_pipeline.registration import register_pages
                reference_paths = [Path(page.path) for page in teacher_pages[subject]]
                registered_paths, registration_meta = register_pages(
                    reference_paths, page_paths,
                    output / "normalized" / re.sub(r"[^\w.-]+", "_", subject),
                    re.sub(r"[^\w.-]+", "_", paths[0].parent.name),
                )
                page_paths = registered_paths
                preprocessing = [dict(prep, registration=registration_meta[index])
                                 for index, prep in enumerate(preprocessing)]
            # Build image-only pages first.  This is the input contract for
            # whole-document VLM tree generation; OCR is deliberately acquired
            # after the logical topology exists.
            visual_source_paths = page_paths if preprocessing else list(paths)
            visual_pages = make_pages(visual_source_paths, "none", {}, language)
            assign_page_identity(visual_pages, doc_id)
            pages = visual_pages

            def load_document_ocr_pages() -> List[Page]:
                # Cached OCR keys refer to source images. Preserve page order,
                # then attach the normalized/registered image contract used by
                # all downstream geometry stages.
                cached_map = dict(ocr_map)
                if not dry_run and ocr_engine != "none":
                    cache_dir = cache_root / "ocr" / str(ocr_engine)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    for source in (paths if preprocessing and ocr_map else visual_source_paths):
                        cache_file = cache_dir / f"{page_file_fingerprint(Path(source))}.json"
                        if cache_file.is_file():
                            try:
                                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                                cached_map[str(Path(source).resolve())] = [
                                    OCRBlock(str(block.get("text", "")), list(block.get("bbox", [])),
                                             block.get("confidence"))
                                    for block in payload.get("blocks", [])]
                            except (OSError, ValueError, TypeError):
                                pass
                if preprocessing and ocr_map:
                    loaded = make_pages(paths, "none" if dry_run else ocr_engine,
                                        cached_map, language)
                    for page, normalized in zip(loaded, visual_source_paths):
                        page.path = str(normalized)
                        page.width, page.height = image_size(normalized)
                    assign_page_identity(loaded, doc_id)
                    if not dry_run and ocr_engine != "none":
                        cache_dir = cache_root / "ocr" / str(ocr_engine)
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        for page in loaded:
                            cache_file = cache_dir / f"{page.file_fingerprint}.json"
                            if page.file_fingerprint and not cache_file.exists():
                                try:
                                    cache_file.write_text(json.dumps(
                                        {"blocks": [asdict(b) for b in page.ocr]}, ensure_ascii=False),
                                        encoding="utf-8")
                                except OSError:
                                    pass
                    return loaded
                loaded = make_pages(visual_source_paths, "none" if dry_run else ocr_engine,
                                    cached_map, language)
                assign_page_identity(loaded, doc_id)
                if not dry_run and ocr_engine != "none":
                    cache_dir = cache_root / "ocr" / str(ocr_engine)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    for page in loaded:
                        cache_file = cache_dir / f"{page.file_fingerprint}.json"
                        if page.file_fingerprint and not cache_file.exists():
                            try:
                                cache_file.write_text(json.dumps({"blocks": [asdict(b) for b in page.ocr]},
                                                                 ensure_ascii=False), encoding="utf-8")
                            except OSError:
                                pass
                return loaded
            if dry_run:
                stage = "dry_run_serialization"
                dry_result = fallback_result(role, subject, pages, "dry-run：未调用 OCR/视觉模型")
                package = package_from_result(dry_result, pages, role, subject, paths[0].parent.name if role == "student" else None)
                from exam_pipeline.localization import ground_contexts
                package.quality["localization_summary"] = ground_contexts(package, pages)
                package.grounding_knowledge = (
                    dict(grounding_runtime_audit)
                )
                result = package.to_dict()
                from exam_pipeline.stage_evaluation import stage_diagnostics
                result["stage_diagnostics"] = stage_diagnostics(package)
            else:
                structure_ocr_retries = []
                stage = "structuring"
                student_id = paths[0].parent.name if role == "student" else None
                from exam_pipeline.structure_recovery import recover_structure
                # The structure model receives original pages only.  This is
                # deliberately a single evidence path: no print/handwriting
                # classifier or pseudo-blank template can alter page identity.
                visual_tree = role == "teacher" and active_structure_vlm != "none"
                cached_tree = None
                if role == "teacher":
                    tree_cache_name = "exam_tree_vlm_only_v1" if vlm_only else "exam_tree"
                    cache_path = cache_root / tree_cache_name / subject / f"{fingerprint}.json"
                    if cache_path.is_file():
                        try:
                            cached_tree = ExamTreeService.load(
                                cache_path, expected_subject=subject,
                                require_valid=True, require_production=False,
                                allow_valid_draft=True)
                        except Exception as exc:
                            LOGGER.warning("exam_tree_cache_invalid path=%s error=%s", cache_path, exc)
                            cached_tree = None
                if cached_tree is not None:
                    visual_tree = False
                # Offline/legacy structure mode still needs OCR before it can
                # recover question anchors. The visual-primary path remains
                # image-only at this point.
                if not visual_tree and not vlm_only:
                    stage = "page_ocr"
                    record_phase("question_tree_generation", "STARTED", evidence="original_preprocessed_images")
                    pages = load_document_ocr_pages()
                    structure_ocr_retries = []
                    if role == "teacher" and local_ocr:
                        from exam_pipeline.structure_recovery import recover_unread_lines
                        from exam_pipeline.ocr import OCRService
                        structure_ocr_retries = recover_unread_lines(pages, OCRService(), ocr_engine, language)
                structure_pages = list(pages)
                structure_sources = [
                    {
                        "page_index": page.index,
                        "source": "original",
                        "evidence": "current_normalized_page",
                    }
                    for page in pages
                ]
                # The teacher is scanned first and becomes the subject Golden.
                # Student extraction is only a source of dynamic answer fields;
                # its logical topology is replaced with the teacher hierarchy.
                package = ExamPackage(
                    exam_id=doc_id, exam_title="", subject=subject,
                    document_type=role, student_id=student_id,
                    total_pages=len(pages), page_files=[p.path for p in pages],
                    sections=[], warnings=[])
                package.structure_audit["sources"] = structure_sources
                package.structure_audit["ocr_retries"] = structure_ocr_retries
                strategies: List[str] = []
                if cached_tree is not None:
                    ExamTreeService.apply_to_package(cached_tree, package)
                    package.exam_title = str(cached_tree.get("exam_title") or cached_tree.get("title") or "")
                    package.structure_audit = copy.deepcopy(
                        (cached_tree.get("provenance") or {}).get("structure_coverage") or
                        {"status": "COMPLETE", "source": "fingerprint_cache"})
                    package.golden_source = f"teacher_cache:{fingerprint}"
                    package.topology_locked = cached_tree.get("state") == "LOCKED"
                    strategies.append(
                        "teacher_exam_tree_fingerprint_cache_locked" if package.topology_locked
                        else "teacher_exam_tree_fingerprint_cache_valid_draft")
                    record_phase("question_tree_generation", "COMPLETED", source="fingerprint_cache",
                                 fingerprint=fingerprint, state=cached_tree.get("state"))
                elif visual_tree:
                    record_phase("question_tree_generation", "STARTED", evidence="original_preprocessed_images")
                    from exam_pipeline.document_structure import DocumentStructureService
                    DocumentStructureService(structure_request, active_structure_vlm).generate(
                        package, pages, structure_pages, (),
                        output / "document_structure" / safe_doc_id,
                        validate_ocr=False,
                        vlm_only=vlm_only,
                    )
                    record_phase("question_tree_generation", "COMPLETED",
                                 source="whole_document_vlm")
                    strategies.append("whole_document_vlm")
                    # A visual backend outage must not leave the deterministic
                    # OCR fallback without evidence. Obtain OCR only after the
                    # proposal attempt, then recover a clearly marked draft.
                    if not any(question.items for section in package.sections
                               for question in section.questions):
                        if vlm_only:
                            raise RuntimeError(
                                "VLM_STRUCTURE_UNRESOLVED: teacher question tree is empty"
                            )
                        stage = "page_ocr_fallback"
                        pages = load_document_ocr_pages()
                        structure_pages = list(pages)
                        if role == "teacher" and local_ocr:
                            from exam_pipeline.structure_recovery import recover_unread_lines
                            from exam_pipeline.ocr import OCRService
                            structure_ocr_retries = recover_unread_lines(
                                pages, OCRService(), ocr_engine, language)
                        package.structure_audit["fallback_after_vlm_proposal"] = True
                        strategies.append("ocr_fallback_after_vlm_proposal")
                        record_phase("question_tree_generation", "COMPLETED",
                                     source="ocr_fallback_after_vlm_proposal")
                        visual_tree = False
                    if (vlm_only
                            and package.structure_audit.get("status") != "COMPLETE"):
                        raise RuntimeError(
                            "TREE_UNRESOLVED: "
                            + json.dumps(package.structure_audit.get("failures") or [],
                                         ensure_ascii=False)
                        )
                    if tree_llm_client is not None:
                        package.warnings.append("全卷视觉构树已启用，旧 tree-llm 文本重写阶段不再覆盖视觉结构")
                elif role == "teacher":
                    legacy = package_from_golden(golden_data, structure_pages, role, subject, student_id)
                    package.sections = legacy.sections if legacy and legacy.sections else heuristic_questions(structure_pages, role)
                    strategies.append("golden" if legacy and legacy.sections else "heuristic")
                else:
                    # Students reuse the teacher's semantic tree; local OCR and
                    # visual slots locate/recognize their own paper independently.
                    strategies.append("inherited_teacher_topology+student_local_evidence")
                package.exam_id = doc_id
                if not visual_tree and cached_tree is None:
                    package.exam_title = (golden_data or {}).get("title", "") if golden_data else ""
                if active_structure_vlm == "none":
                    package.warnings.append("未启用结构 VLM，已使用 PaddleOCR/Golden 离线路径")
                package.warnings.extend(prep_warnings)
                if registration_meta:
                    registered = sum(1 for entry in registration_meta if entry.get("status") == "REGISTERED")
                    package.warnings.append(
                        f"教师页到学生页配准：{registered}/{len(registration_meta)} 页成功，"
                        "失败页保留原图并记录未解决状态"
                    )
                # Preserve order while avoiding duplicate strategy labels.
                strategy = "+".join(dict.fromkeys(strategies)) or "heuristic"
                package.warnings.append(f"提取策略: {strategy}")
                if role == "teacher" and not visual_tree and cached_tree is None:
                    recover_structure(package, structure_pages, pages)
                    record_phase("question_tree_generation", "COMPLETED", source="ocr_fallback")
                if not visual_tree and cached_tree is None:
                    reconcile_exam_package(package)
                if role == "teacher":
                    stage = "fine_grained_item_split"
                    from exam_pipeline.subitems import FineGrainedItemSplitter
                    if not visual_tree and cached_tree is None:
                        subitem_stats = FineGrainedItemSplitter().enrich_package(package, structure_pages)
                    if not any(question.items for section in package.sections for question in section.questions):
                        raise ValueError(f"教师卷 {doc_id} 未提取到题目，不能生成 Golden")
                    if not visual_tree and cached_tree is None:
                        golden_service.seed_regions_from_ocr(package, structure_pages)
                    if cached_tree is None:
                        package.golden_source = f"external:{golden}" if golden and not visual_tree else f"teacher:{doc_id}"
                        package.topology_locked = False
                    override_path = resolve_override(exam_tree_overrides, subject)
                    if override_path:
                        stage = "exam_tree_override_validation"
                        override_tree = ExamTreeService.load(
                            override_path, expected_subject=subject, require_valid=True,
                            require_production=False,  # 允许手动编辑的 Exam Tree，不强制生产级验证
                        )
                        ExamTreeService.apply_to_package(override_tree, package)
                        applied_tree_overrides[subject] = override_tree
                        package.warnings.append(
                            f"已应用人工修订 ExamTree revision={override_tree.get('revision')} "
                            f"fingerprint={override_tree.get('fingerprint')}"
                        )
                else:
                    teacher_package = teacher_packages.get(subject)
                    source_pages = teacher_pages.get(subject, [])
                    if teacher_package is None and golden_data:
                        # Compatibility for datasets that have no teacher
                        # folder but still pass an old page/item Golden file.
                        teacher_package = package_from_golden(golden_data, pages, "teacher", subject)
                        source_pages = list(pages)
                        if teacher_package:
                            teacher_package.golden_source = f"external:{golden}"
                            teacher_package.topology_locked = True
                    if teacher_package is None:
                        raise ValueError(
                            f"学科 {subject} 缺少可用教师卷，无法生成 Golden 并锁定学生题目拓扑"
                        )
                    package = golden_service.inherit_student_topology(
                        teacher_package, package, source_pages, pages, doc_id,
                        student_id or "", localize_with_ocr=not vlm_only,
                    )
                    record_phase("question_tree_generation", "COMPLETED", source="inherited_teacher_tree")

                if vlm_only:
                    stage = "visual_page_extraction"
                    record_phase(
                        "visual_page_extraction", "STARTED",
                        provider=visual_extraction_provider,
                        evidence="current_document_original_pages",
                    )
                    semantic_summary = visual_extraction_service.extract(
                        package, pages, output / "visual_extraction" / safe_doc_id
                    )
                    record_phase(
                        "visual_page_extraction", "COMPLETED",
                        summary=semantic_summary,
                    )
                    record_phase(
                        "visual_page_retry", "COMPLETED",
                        retries=semantic_summary.get("page_retries", 0),
                        failed_pages=semantic_summary.get("failed_pages", 0),
                    )
                    slot_stats = {
                        "slots": semantic_summary.get("slots", 0),
                        "items": semantic_summary.get("items", 0),
                        "source": "vlm_original_page",
                        "ocr_used": False,
                    }
                    package.grounding_knowledge = {
                        "runtime_role": "disabled_in_vlm_only_path",
                        "slot_coordinate_authority": "vlm_original_page_pixels",
                        "ocr_used": False,
                    }
                    package.registration = []
                    package.warnings.append(
                        "原页 VLM 联合提取题目区域、逻辑槽位、最终坐标和答案；"
                        "本路径未运行 OCR"
                    )
                else:
                    # Compatibility path for explicitly disabled visual models.
                    stage = "page_ocr"
                    record_phase("ocr_vlm_cross_validation", "STARTED",
                                 evidence="current_document_ocr")
                    pages = load_document_ocr_pages()
                    if visual_tree:
                        from exam_pipeline.document_structure import DocumentStructureService
                        DocumentStructureService(structure_request, active_structure_vlm).revalidate(
                            package, pages, output / "document_structure" / safe_doc_id,
                            request=structure_request, original_pages=visual_pages)
                    record_phase("ocr_vlm_cross_validation", "COMPLETED",
                                 ocr_blocks=sum(len(page.ocr) for page in pages),
                                 structure_status=package.structure_audit.get("status"))
                    from exam_pipeline.localization import ground_contexts
                    record_phase("question_localization", "STARTED",
                                 mode="ocr_anchor_refinement")
                    package.quality["localization_summary"] = ground_contexts(package, pages)
                    record_phase("question_localization", "COMPLETED",
                                 mode="ocr_anchor_refinement",
                                 summary=package.quality.get("localization_summary"))
                    record_phase("question_localization", "STARTED",
                                 mode="bounded_visual_question_refinement")
                    visual_summary, localization_audit = question_localizer.localize(
                        package, pages, output / "question_localization" / safe_doc_id)
                    if visual_summary:
                        package.quality["localization_visual"] = visual_summary
                    record_phase("question_localization", "COMPLETED",
                                 mode="bounded_visual_question_refinement",
                                 summary=visual_summary,
                                 audit_status=localization_audit.get("status"))
                    record_phase("slot_semantic_proposal", "STARTED",
                                 provider=semantic_provider,
                                 input="grounded_question_contexts")
                    proposal_summary = semantic_service.propose_package(
                        package, pages, output / "slot_semantics_proposals" / safe_doc_id,
                        reference_pages=())
                    record_phase("slot_semantic_proposal", "COMPLETED",
                                 summary=proposal_summary)
                    package.grounding_knowledge = dict(grounding_runtime_audit)
                    package.registration = list(registration_meta)
                    stage = "diagram_masking"
                    from exam_pipeline.layout import QuestionLayoutService
                    diagram_mask_stats = QuestionLayoutService.enrich_diagram_masks(
                        package, pages
                    )
                    if diagram_mask_stats["automatic_diagram_masks"]:
                        package.warnings.append(
                            "版式图表屏蔽：自动标记 {} 个线密集图表区域；区域内横线禁止作为答案槽位".format(
                                diagram_mask_stats["automatic_diagram_masks"]
                            )
                        )
                    if semantic_service.request is None:
                        slot_stats = {"slots": 0, "items": 0,
                                      "source": "visual_backend_unavailable_ocr_only"}
                    else:
                        slot_stats = {"slots": 0, "items": 0,
                                      "source": "visual_proposal_ocr_coordinates"}
                    stage = "ocr_coordinate_grounding"
                    semantic_service.validator.policy = grounding_policy
                    semantic_summary = semantic_service.enrich_package(
                        package, pages, output / "slot_semantics" / safe_doc_id,
                        reference_pages=(), reuse_proposals=True)
                    package.warnings.append(
                        "视觉提出槽位后 OCR 坐标落地：通过 {accepted}，部分 {partial}，降级 {fallback}".format(
                            **semantic_summary))
                    record_phase("ocr_coordinate_grounding", "COMPLETED",
                                 verified_regions=sum(
                                     slot.geometry_status in {"ALIGNED", "ALIGNED_WITH_WARNING"}
                                     for section in package.sections
                                     for question in section.questions
                                     for item in question.items for slot in item.slots
                                 ))

                # Persist crops only after final per-document VLM/OCR geometry.
                stage = "roi_generation"
                from exam_pipeline.roi import RoIPatchGenerator
                roi_records = RoIPatchGenerator().generate_package(
                    package, pages, output / "roi_patches" / safe_doc_id
                )

                if role == "teacher":
                    stage = "teacher_answer_extraction"
                    record_phase("answer_recognition", "STARTED", role="teacher")
                    if vlm_only:
                        from exam_pipeline.visual_extraction import visual_answer_summary
                        answer_summary = visual_answer_summary(package)
                        record_phase("answer_recognition", "COMPLETED",
                                     summary=answer_summary, authority="vlm_original_page")
                        package.warnings.append(
                            "教师答案原页 VLM 识别：{}/{} 个逻辑槽位提取成功".format(
                                answer_summary["answers_extracted"],
                                answer_summary["total_logical_slots"],
                            )
                        )
                    else:
                        from exam_pipeline.teacher_answers import TeacherAnswerExtractionService
                        teacher_references = ()
                        teacher_reference_kind = "single_page"
                        answer_summary = TeacherAnswerExtractionService(
                            policy=grounding_policy, formula_client=answer_vision_client
                        ).extract(
                            package, pages, teacher_references, ocr_engine, language,
                            None, None, teacher_reference_kind,
                            use_exam_tree_fallback=False,
                        )
                        from exam_pipeline.automatic_repair import repair_contaminated_answers
                        record_phase("answer_recognition", "COMPLETED",
                                     summary=answer_summary)
                        record_phase("automatic_repair", "STARTED", role="teacher")
                        repair_summary = repair_contaminated_answers(
                            package, pages, semantic_service,
                            lambda subset: TeacherAnswerExtractionService(
                                policy=grounding_policy,
                                formula_client=answer_vision_client,
                            ).extract(subset, pages, teacher_references,
                                      ocr_engine, language),
                            output / "automatic_relocalization" / safe_doc_id,
                            teacher_references,
                        )
                        record_phase("automatic_repair", "COMPLETED", role="teacher",
                                     summary=repair_summary)
                        fallback_count = answer_summary.get('exam_tree_fallback_used', 0)
                        if fallback_count > 0:
                            package.warnings.append(
                                f"教师答案提取：{answer_summary['answers_extracted']}/{answer_summary['total_slots']} 个槽位成功 "
                                f"({fallback_count} 个来自 Exam Tree 标准答案，"
                                f"{answer_summary['answers_extracted']-fallback_count} 个从图像提取)"
                            )
                        else:
                            package.warnings.append(
                                f"教师答案 VLM/OCR 识别："
                                f"{answer_summary['answers_extracted']}/{answer_summary['total_slots']} 个槽位提取成功"
                            )

                # Structural totals may be normalized, but correctness and
                # student marks are intentionally outside this extraction job.
                from exam_pipeline.scoring import balance_score_tree
                stage = "structure_validation"
                score_warnings = balance_score_tree(package)
                package.warnings.extend(score_warnings)
                if role == "teacher":
                    from exam_pipeline.result_contract import finalize_answers
                    finalize_answers(package, pages)
                    stage = "exam_tree_validation"
                    override_tree = applied_tree_overrides.get(subject)
                    production_tree_gate = bool(production_mode)
                    provenance = {
                        "structure_coverage": package.structure_audit,
                        "kind": ("manual_override" if override_tree else
                                 "whole_document_vlm" if vlm_only
                                 else "whole_document_vlm+ocr_validation" if visual_tree
                                 else "teacher_extraction"),
                        "golden_source": package.golden_source,
                        "override_fingerprint": (
                            override_tree.get("fingerprint") if override_tree else None
                        ),
                        "semantic_model": (
                            (model if active_structure_vlm == "doubao" else paddleocr_vl_model) if visual_tree else None
                        ),
                        "semantic_candidate_gate": (
                            "vlm_schema_page_coverage" if vlm_only
                            else "full_document_ocr_rule_validation" if visual_tree else None
                        ),
                        "stem_source": "original_pages",
                        "coordinate_authority": (
                            "vlm_original_page_pixels" if vlm_only else None
                        ),
                        "ocr_used": False if vlm_only else None,
                    }
                    tree = ExamTreeService.compile(
                        package,
                        revision=int(override_tree.get("revision", 1)) if override_tree else 1,
                        provenance=provenance,
                        production_strict=production_tree_gate,
                    )
                    safe_subject = re.sub(r"[^\w.-]+", "_", subject)
                    tree_path = output / "exam_trees" / f"{safe_subject}.json"
                    if tree["validation"]["errors"]:
                        rejected_path = (
                            output / "exam_trees" / "rejected" /
                            f"{safe_subject}.draft.json"
                        )
                        ExamTreeService.save(tree, rejected_path, lock=False)
                        package.warnings.append("结构证据未通过强校验，保留草稿: " + str(rejected_path))
                        # The deterministic package remains available for independent
                        # student extraction; it must not be advertised as locked.
                        tree = ExamTreeService.save(tree, tree_path, lock=False)
                        package.topology_locked = False
                    else:
                        tree = ExamTreeService.save(tree, tree_path, lock=True)
                        ExamTreeService.apply_to_package(tree, package)
                        tree_cache_name = "exam_tree_vlm_only_v1" if vlm_only else "exam_tree"
                        cache_path = cache_root / tree_cache_name / subject / f"{fingerprint}.json"
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            ExamTreeService.save(tree, cache_path, lock=True)
                            package.warnings.append(f"ExamTree 已按教师图像指纹缓存: {fingerprint[:12]}")
                        except OSError as exc:
                            package.warnings.append(f"ExamTree 缓存写入失败: {exc}")
                    exam_trees[subject] = tree
                    manifest["exam_trees"][subject] = {
                        "path": str(tree_path),
                        "tree_id": tree["tree_id"],
                        "revision": tree["revision"],
                        "fingerprint": tree["fingerprint"],
                        "validation": tree["validation"],
                    }
                if role == "student":
                    stage = "answer_verification"
                    record_phase("answer_recognition", "STARTED", role="student")
                    if vlm_only:
                        from exam_pipeline.visual_extraction import visual_answer_summary
                        visual_answers = visual_answer_summary(package)
                        document_audit = {"summary": visual_answers, "pages": {}}
                        record_phase(
                            "answer_recognition", "COMPLETED",
                            summary=visual_answers,
                            authority="vlm_original_page",
                        )
                    else:
                        from exam_pipeline.verification import SlotVerificationService
                        verification_references = []
                        reference_context = {"kind": "single_page", "confidence": None}
                        document_audit = SlotVerificationService(
                            policy=grounding_policy, formula_client=answer_vision_client
                        ).verify_package(
                            package, pages, verification_references, ocr_engine,
                            language, reference_context, None,
                            independent_page=True,
                        )
                        from exam_pipeline.automatic_repair import repair_contaminated_answers
                        record_phase("answer_recognition", "COMPLETED",
                                     summary=document_audit.get("summary"))
                        record_phase("automatic_repair", "STARTED", role="student")
                        repair_summary = repair_contaminated_answers(
                            package, pages, semantic_service,
                            lambda subset: SlotVerificationService(
                                policy=grounding_policy,
                                formula_client=answer_vision_client,
                            ).verify_package(
                                subset, pages, verification_references,
                                ocr_engine, language, reference_context,
                                independent_page=True,
                            ),
                            output / "automatic_relocalization" / safe_doc_id,
                            verification_references,
                        )
                        record_phase("automatic_repair", "COMPLETED", role="student",
                                     summary=repair_summary)
                        batch_audit_pages.update(document_audit.get("pages", {}))
                from exam_pipeline.result_contract import finalize_answers
                finalize_answers(package, pages)
                from exam_pipeline.quality import validate_item_regions
                stage = "quality_gate"
                validate_item_regions(package, pages)
                if package.quality.get("status") == "NEED_REVIEW":
                    package.warnings.append("框质量校验未通过，相关题目已记录未解决状态")
                # Export figure regions after all page-local geometry is
                # finalized.  The asset service is deterministic and uses
                # current-page evidence; its links are relative to output so
                # results remain portable when the output directory moves.
                stage = "diagram_asset_export"
                from exam_pipeline.diagram_assets import DiagramAssetService
                diagram_asset_summary = DiagramAssetService().export(
                    package, pages, output,
                    output / "diagram_assets" / safe_doc_id,
                )
                record_phase("diagram_asset_export", "COMPLETED",
                             assets=diagram_asset_summary.get("diagram_assets", 0),
                             items=diagram_asset_summary.get("items_with_diagrams", 0),
                             errors=len(diagram_asset_summary.get("errors", [])))
                from exam_pipeline.visualization import render_slot_overlays
                stage = "slot_visualization"
                visualization_summary = render_slot_overlays(
                    package, pages, output / "visualizations",
                    paths[0].parent.name if role == "student" else "teacher",
                )
                record_phase("slot_visualization", "COMPLETED",
                             contact_sheet=visualization_summary.get("contact_sheet"),
                             pages=visualization_summary.get("page_count", 0))
                result = package.to_dict()
                record_phase("result_output", "COMPLETED", artifact="document_json_pending")
                from exam_pipeline.stage_evaluation import stage_diagnostics
                result["stage_diagnostics"] = stage_diagnostics(package)
                result["pipeline_order"] = manifest["runtime"]["pipeline_order"]
                result["phase_trace"] = phase_trace
                result["roi_patches"] = roi_records
                result["diagram_assets"] = diagram_asset_summary
                result["slot_visualization"] = visualization_summary
                slot_stats["slots"] = sum(len(i.slots) for s in package.sections for q in s.questions for i in q.items)
                slot_stats["items"] = sum(len(q.items) for s in package.sections for q in s.questions)
                result["slot_topology"] = slot_stats
                if semantic_summary:
                    final_items = [i for s in package.sections for q in s.questions for i in q.items]
                    semantic_summary = dict(semantic_summary,
                        accepted=sum(i.slot_semantics_audit.get("status") == "ACCEPTED" for i in final_items),
                        partial=sum(i.slot_semantics_audit.get("status") == "PARTIAL" for i in final_items),
                        fallback=sum(i.slot_semantics_audit.get("status") == "FALLBACK" for i in final_items),
                        automatic_relocalizations=sum(bool(i.quality.get("automatic_relocalization")) for i in final_items))
                result["slot_semantics"] = semantic_summary
                if subitem_stats:
                    result["fine_grained_items"] = subitem_stats
                if document_audit:
                    result["answer_verification"] = document_audit["summary"]
            result.setdefault("subject", subject)
            if role == "student" and not result.get("student_id"):
                result["student_id"] = paths[0].parent.name
            result.setdefault("pages", [{"page": p.index, "page_index": p.page_index or p.index,
                                          "physical_page_id": p.physical_page_id,
                                          "document_id": p.document_id,
                                          "file_fingerprint": p.file_fingerprint,
                                          "ocr_source_fingerprint": p.ocr_source_fingerprint,
                                          "image": p.path, "width": p.width, "height": p.height}
                                         for p in pages])
            result["page_files"] = [p.path for p in pages]
            result["ocr"] = ([] if vlm_only else [
                {"page": p.index, "blocks": [asdict(b) for b in p.ocr]}
                for p in pages
            ])
            result["recognition_mode"] = "vlm_only" if vlm_only else "legacy_ocr_compatible"
            result["ocr_used"] = False if vlm_only else any(page.ocr for page in pages)
            result["coordinate_authority"] = (
                "vlm_original_page_pixels" if vlm_only else "local_ocr"
            )
            result["answer_authority"] = (
                "vlm_original_page" if vlm_only else "vlm_primary_ocr_auxiliary"
            )
            result["preprocessing"] = preprocessing
            result["phase_trace"] = phase_trace
            result["pipeline_order"] = manifest["runtime"]["pipeline_order"]
            result["performance"] = performance_collector.summary(doc_id)
            if registration_meta:
                result["registration"] = registration_meta
            expected_pages = next((len(item[2]) for item in docs if item[0] == subject and item[1] == "teacher"), None)
            if role == "student" and expected_pages and len(paths) != expected_pages:
                result.setdefault("warnings", []).append(f"页数异常：检测到 {len(paths)} 页，教师卷为 {expected_pages} 页")
            out_file = output / f"{safe_doc_id}.json"
            from exam_pipeline.io_utils import atomic_write_json
            stage = "artifact_persistence"
            atomic_write_json(out_file, result)
            manifest["documents"].append({
                "id": doc_id,
                "role": role,
                "subject": subject,
                "student_id": result.get("student_id"),
                "output": str(out_file),
                "structured_file": str(out_file),
                "normalized_pages": [str(p) for p in page_paths],
                "pages": len(paths),
                "total_score": result.get("total_score"),
                "is_teacher_golden": role == "teacher",
                "golden_source": result.get("golden_source", ""),
                "topology_locked": bool(result.get("topology_locked")),
                "exam_tree_id": result.get("exam_tree_id", ""),
                "exam_tree_revision": result.get("exam_tree_revision", 0),
                "exam_tree_fingerprint": result.get("exam_tree_fingerprint", ""),
                "strategy": next((w.split(": ", 1)[1] for w in result.get("warnings", []) if str(w).startswith("提取策略:")), ""),
                "quality_status": (result.get("quality") or {}).get("status", "NOT_EVALUATED"),
                "roi_patch_count": len(result.get("roi_patches", [])),
                "diagram_asset_count": int((result.get("diagram_assets") or {}).get("diagram_assets", 0)),
                "slot_visualization": (result.get("slot_visualization") or {}).get("contact_sheet"),
                "slot_count": sum(
                    len(item.get("slots", []))
                    for section in result.get("sections", [])
                    for question in section.get("questions", [])
                    for item in question.get("items", [])
                ),
                "answer_verification": result.get("answer_verification"),
                "input_fingerprint": f"sha256:{fingerprint}",
                "source_pages": [str(path) for path in paths],
                "duration_seconds": round(time.monotonic() - document_started, 3),
                "performance": performance_collector.summary(doc_id, include_events=False),
                "phase_trace": phase_trace,
                "pipeline_order": manifest["runtime"]["pipeline_order"],
            })
            LOGGER.info("document_completed id=%s quality=%s", doc_id,
                        (result.get("quality") or {}).get("status", "NOT_EVALUATED"))
            if role == "teacher" and not dry_run:
                teacher_packages[subject] = package
                teacher_pages[subject] = list(pages)
                teachers[subject] = result
                safe_subject = re.sub(r"[^\w.-]+", "_", subject)
                golden_path = output / "generated_golden" / f"{safe_subject}.json"
                golden_service.save_generated(package, golden_path)
                manifest["generated_goldens"][subject] = str(golden_path)
        except Exception as exc:
            if not phase_trace or phase_trace[-1].get("status") != "FAILED":
                phase_trace.append({
                    "phase": stage,
                    "status": "FAILED",
                    "at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                    "error_type": type(exc).__name__,
                })
            manifest["errors"].append({
                "id": doc_id, "error": str(exc), "error_type": type(exc).__name__,
                "stage": stage, "duration_seconds": round(time.monotonic() - document_started, 3),
                "phase_trace": phase_trace,
                "pipeline_order": manifest["runtime"]["pipeline_order"],
            })
            LOGGER.exception("document_failed id=%s stage=%s", doc_id, stage)
            if "PaddleOCR 初始化/下载模型失败" in str(exc):
                # All following pages would fail identically; stop the batch and
                # leave a concise actionable error in manifest.json.
                break
    audit_records = [record for records in batch_audit_pages.values() for record in records]
    accepted_records = [
        record for record in audit_records
        if record.get("status") in {"COORDINATE_ACCEPTED", "COORDINATE_ACCEPTED_WITH_WARNING"}
    ]
    audit_summary = {
        "total_audit_slots": len(audit_records),
        "coordinates_accepted": len(accepted_records),
        "coordinates_accepted_with_warning": sum(
            record.get("status") == "COORDINATE_ACCEPTED_WITH_WARNING"
            for record in audit_records),
        "anomaly_escalated": sum(
            record.get("status") == "ANOMALY_ESCALATED" for record in audit_records
        ),
    }
    # Verification controllers historically keep their private working boxes
    # in yxyx.  Convert the persisted audit recursively so every public
    # artifact follows the package xyxy contract.
    from exam_pipeline.roi import yxyx_to_xyxy

    def _audit_xyxy(value):
        if isinstance(value, dict):
            return {key: (_audit_xyxy(child) if not (
                isinstance(child, list) and len(child) == 4
                and "bbox" in str(key).lower()
            ) else yxyx_to_xyxy(child)) for key, child in value.items()}
        if isinstance(value, list):
            return [_audit_xyxy(child) for child in value]
        return value

    from exam_pipeline.io_utils import atomic_write_json
    if vlm_only:
        visual_phase_summaries = [
            entry.get("summary") or {}
            for document in manifest["documents"]
            for entry in document.get("phase_trace", [])
            if entry.get("phase") == "visual_page_extraction"
            and entry.get("status") == "COMPLETED"
        ]
        manifest["visual_extraction_summary"] = {
            "documents": len(manifest["documents"]),
            "page_calls": sum(summary.get("page_calls", 0)
                              for summary in visual_phase_summaries),
            "page_retries": sum(summary.get("page_retries", 0)
                                for summary in visual_phase_summaries),
            "failed_pages": sum(summary.get("failed_pages", 0)
                                for summary in visual_phase_summaries),
            "ocr_used": False,
        }
    else:
        audit_payload = {
            "summary": audit_summary,
            "pages": _audit_xyxy(batch_audit_pages),
            "bbox_format": "xyxy",
            "canonical_canvas": {"width": 1654, "height": 2338},
        }
        audit_path = output / "answer_verification_audit.json"
        atomic_write_json(audit_path, audit_payload)
        manifest["answer_verification_audit"] = str(audit_path)
        manifest["answer_verification_summary"] = audit_summary
    manifest["completed_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    manifest["duration_seconds"] = round(time.monotonic() - started_monotonic, 3)
    manifest["performance"] = performance_collector.summary(include_events=False)
    manifest["status"] = "COMPLETED_WITH_ERRORS" if manifest["errors"] else "COMPLETED"
    manifest["summary"] = {
        "documents_succeeded": len(manifest["documents"]),
        "documents_failed": len(manifest["errors"]),
        "documents_needing_review": sum(
            1 for document in manifest["documents"]
            if document.get("quality_status") == "NEED_REVIEW"
        ),
    }
    from exam_pipeline.io_utils import atomic_write_json
    atomic_write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    global PADDLE_MODEL_TIER, PADDLE_DET_MODEL, PADDLE_REC_MODEL
    parser = argparse.ArgumentParser(description="从教师卷/学生卷扫描图提取题目级 JSON")
    parser.add_argument("dataset", type=Path, help="数据集根目录；当前扫描样本使用 '扫描图片'")
    parser.add_argument("-o", "--output", type=Path, default=Path("extracted"))
    parser.add_argument("--model", default=DOUBAO_MODEL, help="豆包模型名或火山方舟 Endpoint ID")
    parser.add_argument("--endpoint", default=DOUBAO_RESPONSES_ENDPOINT, help="火山方舟 Responses API 地址")
    parser.add_argument("--api-key", default=DOUBAO_API_KEY,
                        help="优先使用 DOUBAO_API_KEY 环境变量；仅建议开发调试时临时传参")
    parser.add_argument(
        "--ocr", choices=["auto", "paddle", "tesseract", "none"], default="none",
        help="兼容离线路径的 OCR 引擎；默认 VLM-only 路径不运行 OCR",
    )
    parser.add_argument("--ocr-json", type=Path, help="预先生成的 OCR JSONL，跳过本地 OCR")
    parser.add_argument("--golden", type=Path, help="兼容旧流程的外部 Golden（可选）；默认扫描各学科 teacher 目录自动生成")
    parser.add_argument("--language", default="chi_sim+eng", help="tesseract 语言包")
    parser.add_argument("--include-scan", action="store_true", help="教师目录同时处理扫描_ 图片")
    parser.add_argument("--dry-run", action="store_true", help="只扫描分组并生成占位 JSON，不调用模型")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--limit", type=int, help="只处理前 N 份文档，用于小规模验证；默认处理全部文档")
    parser.add_argument("--slot-semantics", choices=["auto", "doubao", "paddleocr-vl", "none"], default="auto",
                        help="视觉模型联合识别题目区域、槽位、最终坐标和答案")
    parser.add_argument("--subject", help="只处理指定学科")
    parser.add_argument("--student-id", help="只处理指定学生卷；会同时加载该学科教师卷以建立题目拓扑")
    parser.add_argument(
        "--exam-tree-overrides", type=Path,
        help="人工修订 ExamTree JSON 或目录（目录内按 <subject>.json 命名）；校验通过后供全部学生复用",
    )
    parser.add_argument("--disable-local-ocr", action="store_true",
                        help="关闭教师结构漏行的局部 OCR 恢复")
    parser.add_argument("--fail-on-review", action="store_true",
                        help="有任何文档未通过质量门时返回码 2，适用于生产 CI/调度")
    parser.add_argument("--hitl-workspace", type=Path,
                        help="提取成功后直接导出到 HITL workspace（可选）")
    parser.add_argument("--hitl-overwrite", action="store_true",
                        help="允许 --hitl-workspace 覆盖已有同名批次")
    parser.add_argument("--feedback-db", type=Path,
                        help="只读吸收 HITL SQLite 坐标审计中位数先验（可选）")
    parser.add_argument(
        "--grounding-kb", type=Path,
        default=Path(__file__).resolve().parent / "GROUNDING_KNOWLEDGE_BASE.md",
        help="视觉定位知识库 Markdown；默认加载项目根目录 GROUNDING_KNOWLEDGE_BASE.md",
    )
    parser.add_argument("--paddle-model-tier", choices=["mobile", "server"],
                        default=PADDLE_MODEL_TIER,
                        help="PP-OCRv5 模型规格；server 更准但更慢且依赖兼容运行时")
    parser.add_argument(
        "--structure-vlm", choices=["auto", "paddleocr-vl", "doubao", "none"],
        default=os.getenv("STRUCTURE_VLM_PROVIDER", "auto"),
        help=("全卷视觉构树；auto 优先豆包，其次已配置的 PaddleOCR-VL；"
              "VLM-only 路径使用 Schema、页码和覆盖规则验证结构"),
    )
    parser.add_argument(
        "--paddleocr-vl-endpoint", default=PADDLEOCR_VL_ENDPOINT,
        help="独立 PaddleOCR-VL OpenAI 兼容服务的 /v1/chat/completions 地址",
    )
    parser.add_argument(
        "--paddleocr-vl-model", default=PADDLEOCR_VL_MODEL,
        help="PaddleOCR-VL 服务中的模型名",
    )
    parser.add_argument(
        "--paddleocr-vl-api-key", default=PADDLEOCR_VL_API_KEY,
        help="可选服务鉴权；生产环境建议使用 PADDLEOCR_VL_API_KEY 环境变量",
    )
    parser.add_argument(
        "--tree-llm-endpoint", default=TREE_LLM_ENDPOINT,
        help=("教师卷题目树语义 LLM 的 OpenAI 兼容 /v1/chat/completions 地址；"
              "未配置时继续使用确定性 OCR 题号解析"),
    )
    parser.add_argument(
        "--tree-llm-model", default=TREE_LLM_MODEL,
        help="题目树语义 LLM 模型名；只处理教师卷，不提供坐标",
    )
    parser.add_argument(
        "--tree-llm-api-key", default=TREE_LLM_API_KEY,
        help="可选鉴权；生产环境建议使用 TREE_LLM_API_KEY 环境变量",
    )
    parser.add_argument(
        "--tree-llm-use-doubao", action="store_true",
        help=("题目树 LLM 复用原有豆包 --model/--endpoint/--api-key；"
              "兼容方舟 Responses API，不需要重复配置 TREE_LLM_*"),
    )
    parser.add_argument(
        "--slot-vlm-verify", action="store_true",
        help="兼容开关：启用视觉槽位提议与本地证据验证，模型区域仅作搜索提示",
    )
    parser.add_argument(
        "--production", action="store_true",
        help="生产强校验：要求完整 VLM 结构、槽位坐标和答案证据",
    )
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default=os.getenv("EXAM_LOG_LEVEL", "INFO"), help="运行日志级别")
    args = parser.parse_args(argv)
    if args.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        parser.error(f"无效日志级别: {args.log_level}")
    if args.feedback_db and not args.feedback_db.is_file():
        parser.error(f"HITL 反馈数据库不存在: {args.feedback_db}")
    if args.grounding_kb and not args.grounding_kb.is_file():
        parser.error(f"视觉定位知识库不存在: {args.grounding_kb}")
    if args.exam_tree_overrides and not args.exam_tree_overrides.exists():
        parser.error(f"ExamTree 修订文件或目录不存在: {args.exam_tree_overrides}")
    PADDLE_MODEL_TIER = args.paddle_model_tier
    PADDLE_DET_MODEL = f"PP-OCRv5_{PADDLE_MODEL_TIER}_det"
    PADDLE_REC_MODEL = f"PP-OCRv5_{PADDLE_MODEL_TIER}_rec"
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if not args.dataset.is_dir():
        parser.error(f"数据集目录不存在: {args.dataset}")
    # CLI compatibility entry point; the public integration surface is the
    # modular ExamPipeline facade.
    from exam_pipeline.service import ExamPipeline
    manifest = ExamPipeline().run(
        args.dataset, args.output, args.model, args.api_key, args.endpoint,
        args.ocr, args.ocr_json, args.dry_run, args.include_scan,
        args.language, args.timeout, args.golden, args.limit,
        not args.disable_local_ocr, args.feedback_db,
        args.grounding_kb,
        args.paddle_model_tier,
        args.subject,
        args.student_id,
        args.exam_tree_overrides,
        args.structure_vlm,
        args.paddleocr_vl_endpoint,
        args.paddleocr_vl_model,
        args.paddleocr_vl_api_key,
        args.tree_llm_endpoint,
        args.tree_llm_model,
        args.tree_llm_api_key,
        args.tree_llm_use_doubao,
        args.slot_vlm_verify,
        args.production,
        slot_semantics=args.slot_semantics,
    )
    if args.hitl_workspace and not manifest["errors"]:
        from exam_pipeline.hitl_export import HITLExporter
        export_result = HITLExporter().export(
            args.output, args.hitl_workspace, overwrite=args.hitl_overwrite
        )
        manifest["hitl_export"] = {
            "workspace": export_result["workspace"],
            "batches": [batch["batch_id"] for batch in export_result["batches"]],
        }
        from exam_pipeline.io_utils import atomic_write_json
        atomic_write_json(args.output / "manifest.json", manifest)
    if manifest["errors"]:
        print("首个错误: " + str(manifest["errors"][0]["error"]), file=sys.stderr)
    print(json.dumps({"documents": len(manifest["documents"]), "errors": len(manifest["errors"]), "output": str(args.output)}, ensure_ascii=False))
    if manifest["errors"]:
        return 1
    if args.fail_on_review and manifest.get("summary", {}).get("documents_needing_review", 0):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
