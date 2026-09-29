#!/usr/bin/env python3
"""VLM-only extraction for teacher and student exam sheets."""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import logging
import os
import re
import shlex
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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
    Slot as ModularSlot,
    RoIPatchRef as ModularRoIPatchRef,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
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
        "DOUBAO_RESPONSES_ENDPOINT",
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
DOUBAO_MODEL = os.getenv("DOUBAO_MODEL", "doubao-seed-2.1-turbo")
DOUBAO_RESPONSES_ENDPOINT = os.getenv(
    "DOUBAO_RESPONSES_ENDPOINT", DOUBAO_BASE_URL.rstrip("/") + "/responses"
)
EXAM_REQUEST_TIMEOUT = int(os.getenv("EXAM_REQUEST_TIMEOUT", "180"))


def _safe_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# Upload-only image settings. Coordinates stay normalized to the original page
# frame, so preserving the aspect ratio keeps downstream geometry compatible.
UPLOAD_MAX_LONG_EDGE = max(0, _safe_int_env("EXAM_UPLOAD_MAX_LONG_EDGE", 2300))
UPLOAD_JPEG_QUALITY = min(100, max(1, _safe_int_env("EXAM_UPLOAD_JPEG_QUALITY", 85)))
UPLOAD_CACHE_DIR = os.getenv("EXAM_UPLOAD_CACHE_DIR", "").strip()


OCRBlock = ModularOCRBlock
Page = ModularPage
PageRegion = ModularPageRegion
ExamItem = ModularExamItem
ExamQuestion = ModularExamQuestion
ExamSection = ModularExamSection
ExamPackage = ModularExamPackage
DiagramRef = ModularDiagramRef
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


def make_pages(paths: Sequence[Path], document_id: str = "") -> List[Page]:
    pages = []
    for i, path in enumerate(paths, 1):
        width, height = image_size(path)
        physical_id = f"{document_id}:p{i}" if document_id else f"{path.stem}:p{i}"
        fingerprint = page_file_fingerprint(path)
        pages.append(Page(i, str(path), width, height, [],
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


def _data_url(path: Path) -> str:
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(path.suffix.lower(), "image/jpeg")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _upload_cache_root() -> Path:
    configured = Path(UPLOAD_CACHE_DIR).expanduser() if UPLOAD_CACHE_DIR else (
        Path(__file__).resolve().parent / ".exam_pipeline_cache" / "upload_images"
    )
    configured.mkdir(parents=True, exist_ok=True)
    return configured


def _compressed_upload_path(source: Path) -> Path:
    """Return a cached, aspect-preserving JPEG used only for VLM upload."""
    source = Path(source)
    if UPLOAD_MAX_LONG_EDGE <= 0:
        return source
    try:
        stat = source.stat()
        cache_key = hashlib.sha256(
            "{}:{}:{}:{}:{}".format(
                source.resolve(), stat.st_size, stat.st_mtime_ns,
                UPLOAD_MAX_LONG_EDGE, UPLOAD_JPEG_QUALITY,
            ).encode("utf-8")
        ).hexdigest()[:32]
        target = _upload_cache_root() / (cache_key + ".jpg")
        if target.is_file() and target.stat().st_size > 0:
            return target if target.stat().st_size < stat.st_size else source

        from PIL import Image
        with Image.open(source) as image:
            width, height = image.size
            scale = min(1.0, UPLOAD_MAX_LONG_EDGE / max(width, height))
            resized = image
            if scale < 1.0:
                resized = image.resize(
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    Image.Resampling.LANCZOS,
                )
            if resized.mode not in {"RGB", "L"}:
                resized = resized.convert("RGB")
            elif resized.mode == "L":
                resized = resized.convert("RGB")
            temporary = tempfile.NamedTemporaryFile(
                dir=str(target.parent), prefix=target.stem + ".", suffix=".tmp", delete=False,
            )
            temporary_path = Path(temporary.name)
            temporary.close()
            try:
                resized.save(
                    temporary_path, format="JPEG", quality=UPLOAD_JPEG_QUALITY,
                    optimize=True, progressive=True,
                )
                os.replace(temporary_path, target)
            finally:
                temporary_path.unlink(missing_ok=True)
            if resized is not image:
                resized.close()
        return target if target.stat().st_size < stat.st_size else source
    except Exception as exc:
        LOGGER.warning("upload_image_compression_failed path=%s error=%s", source, exc)
        return source


def _prepare_upload_paths(paths: Sequence[Path]) -> List[Path]:
    return [_compressed_upload_path(Path(path)) for path in paths]


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
    upload_paths = _prepare_upload_paths(paths)
    upload_audit = {
        "upload_paths": [str(path) for path in upload_paths],
        "upload_image_count": len(upload_paths),
        "upload_image_bytes": sum(
            path.stat().st_size for path in upload_paths if path.is_file()
        ),
        "upload_max_long_edge": UPLOAD_MAX_LONG_EDGE,
        "upload_jpeg_quality": UPLOAD_JPEG_QUALITY,
    }
    content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    content.extend({"type": "input_image", "image_url": _data_url(path), "detail": "high"} for path in upload_paths)
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
                error.response_audit = {
                    **upload_audit,
                    "attempts": attempts_used,
                    "retry_reasons": retry_reasons + [f"http_{exc.code}"],
                }
                raise error from exc
            retry_reasons.append(f"http_{exc.code}")
            LOGGER.warning("vlm_retry attempt=%d status=%d", attempt, exc.code)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == max_attempts:
                error = RuntimeError(f"视觉模型网络请求失败: {exc}")
                error.response_audit = {
                    **upload_audit,
                    "attempts": attempts_used,
                    "retry_reasons": retry_reasons + ["network"],
                }
                raise error from exc
            retry_reasons.append("network")
            LOGGER.warning("vlm_retry attempt=%d reason=network", attempt)
        time.sleep(min(4.0, 0.5 * (2 ** (attempt - 1))))
    response_audit = {
        **upload_audit,
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


def _process_serial(
        dataset: Path, output: Path, model: str = DOUBAO_MODEL,
        api_key: Optional[str] = DOUBAO_API_KEY,
        endpoint: str = DOUBAO_RESPONSES_ENDPOINT, *, dry_run: bool = False,
        include_scan: bool = False, timeout: int = EXAM_REQUEST_TIMEOUT,
        limit: Optional[int] = None,
        subject_filter: Optional[str] = None,
        student_id_filter: Optional[str] = None,
        exam_tree_overrides: Optional[Path] = None,
        production_mode: bool = False,
        _student_only: bool = False,
        _write_manifest: bool = True,
        _return_context: bool = False,
        _page_workers: int = 1,
        _shared_teacher_packages: Optional[Dict[str, ExamPackage]] = None,
        _shared_teacher_pages: Optional[Dict[str, List[Page]]] = None) -> Dict[str, Any]:
    from exam_pipeline.golden import GoldenTemplateService
    from exam_pipeline.exam_tree import ExamTreeService, resolve_override
    from exam_pipeline.performance import PerformanceCollector, set_active_collector

    performance_collector = PerformanceCollector()
    set_active_collector(performance_collector)

    if not dry_run and not str(api_key or "").strip():
        raise ValueError("VLM-only 流程需要 DOUBAO_API_KEY 或 --api-key")
    active_structure_vlm = "doubao"
    structure_request = (
        (lambda prompt, images, schema: call_doubao(
            model, str(api_key or ""), prompt, images, endpoint, timeout, schema))
        if not dry_run else None
    )
    if structure_request is not None:
        structure_request = performance_collector.instrument(
            structure_request, stage="question_tree_generation",
            provider=active_structure_vlm, operation="structure_request")
    semantic_provider = "doubao"
    semantic_request = (
        (lambda prompt, paths, schema: call_doubao(
            model, str(api_key or ""), prompt, paths, endpoint, timeout, schema))
        if not dry_run else None
    )
    semantic_base_request = semantic_request
    if semantic_base_request is not None:
        semantic_request = performance_collector.instrument(
            semantic_base_request, stage="slot_semantic_proposal",
            provider=semantic_provider, operation="semantic_request")
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
        visual_extraction_request, visual_extraction_provider,
        page_workers=max(1, int(_page_workers or 1)),
    )
    vlm_only = True

    output.mkdir(parents=True, exist_ok=True)
    cache_base = Path(os.getenv(
        "EXAM_PIPELINE_CACHE_DIR",
        str(Path(__file__).resolve().parent / ".exam_pipeline_cache"),
    )).expanduser().resolve()
    dataset_cache_id = hashlib.sha256(str(dataset.resolve()).encode("utf-8")).hexdigest()[:16]
    cache_root = cache_base / "datasets" / dataset_cache_id
    cache_root.mkdir(parents=True, exist_ok=True)
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
    teachers: Dict[str, Dict[str, Any]] = {}
    teacher_packages: Dict[str, ExamPackage] = dict(_shared_teacher_packages or {})
    teacher_pages: Dict[str, List[Page]] = {
        subject: list(pages) for subject, pages in (_shared_teacher_pages or {}).items()
    }
    teacher_subjects.update(teacher_packages)
    if _student_only:
        docs = [entry for entry in docs if entry[1] == "student"]
    exam_trees: Dict[str, Dict[str, Any]] = {}
    applied_tree_overrides: Dict[str, Dict[str, Any]] = {}
    golden_service = GoldenTemplateService()
    started_monotonic = time.monotonic()
    started_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    runtime_pipeline_order = [
        "document_discovery",
        "image_preprocessing",
        "question_tree_generation",
        "visual_page_extraction",
        "answer_recognition",
        "visual_page_retry",
        "diagram_asset_export",
        "slot_visualization",
        "result_output",
    ]
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
        "runtime": {
            "pipeline_version": "2.12.0",
            "pipeline_order": runtime_pipeline_order,
            "recognition_mode": "vlm_only",
            "production_mode": bool(production_mode),
            "model": model,
            "endpoint": endpoint,
            "ocr_used": False,
            "structure_vlm": {
                "active": "doubao",
                "model": model,
                "endpoint": endpoint,
                "scope": "lightweight_whole_document_topology_then_page_text_enrichment",
                "primary_evidence": "all_original_pages",
                "proposal_stage": "original_images",
                "validation": "vlm_schema_page_coverage",
                "max_semantic_attempts": 2,
                "page_fallback": "bounded_per_page_topology_and_text_recovery",
                "id_authority": "local_deterministic",
                "response_diagnostics": "status_incomplete_details_usage_parse_position_raw_artifact",
                "geometry_authority": "vlm_original_page_pixels",
                "model_geometry_policy": "final",
            },
            "slot_semantics": {
                "provider": "doubao",
                "enabled": bool(semantic_request),
                "authority": "question_slots_coordinates_and_answers",
                "coordinate_authority": "vlm_original_page_pixels",
                "max_semantic_attempts_per_item": 2,
                "batch_size": "one_physical_page_all_items",
                "batch_policy": "one_page_batch_then_item_retry",
                "candidate_visual_review": False,
                "answer_visual_transcription": bool(semantic_request),
                "answer_authority": "vlm_original_page",
                "fallback": "page_local_vlm_retry_then_unresolved",
                "ocr_used": False,
            },
            "question_localization": {
                "enabled": bool(visual_extraction_request),
                "provider": semantic_provider,
                "stage": "visual_page_extraction",
                "authority": "vlm_final_question_region",
                "coordinate_authority": "vlm_original_page_pixels",
                "fallback": "page_local_vlm_retry",
            },
            "upload_image": {
                "enabled": UPLOAD_MAX_LONG_EDGE > 0,
                "max_long_edge": UPLOAD_MAX_LONG_EDGE,
                "jpeg_quality": UPLOAD_JPEG_QUALITY,
                "coordinate_frame": "normalized_0_1000_original_page",
                "cache_root": str(_upload_cache_root()),
            },
            "dry_run": dry_run,
            "concurrency": {"page_workers": int(_page_workers or 1)},
            "cache_policy": {
                "root": str(cache_root),
                "dataset_cache_id": dataset_cache_id,
                "exam_tree": "teacher_fingerprint_locked_vlm_only_v1",
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
            if not dry_run and role == "student" and subject not in teacher_subjects:
                raise ValueError(f"学科 {subject} 没有教师卷，无法生成 Golden 并处理学生卷")
            stage = "normalization"
            record_phase("image_preprocessing", "STARTED")
            page_paths, preprocessing, prep_warnings = (list(paths), [], []) if dry_run else _prepare_paths(paths, output, subject, role)
            record_phase("image_preprocessing", "COMPLETED", pages=len(page_paths), artifacts=preprocessing)
            visual_source_paths = page_paths if preprocessing else list(paths)
            visual_pages = make_pages(visual_source_paths, doc_id)
            assign_page_identity(visual_pages, doc_id)
            pages = visual_pages
            if dry_run:
                stage = "dry_run_serialization"
                dry_result = fallback_result(role, subject, pages, "dry-run：未调用视觉模型")
                package = package_from_result(dry_result, pages, role, subject, paths[0].parent.name if role == "student" else None)
                result = package.to_dict()
                from exam_pipeline.stage_evaluation import stage_diagnostics
                result["stage_diagnostics"] = stage_diagnostics(package)
            else:
                stage = "structuring"
                student_id = paths[0].parent.name if role == "student" else None
                # The structure model receives original pages only.  This is
                # deliberately a single evidence path: no print/handwriting
                # classifier or pseudo-blank template can alter page identity.
                visual_tree = role == "teacher"
                cached_tree = None
                if role == "teacher":
                    tree_cache_name = "exam_tree_vlm_only_v1"
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
                    if not any(question.items for section in package.sections
                               for question in section.questions):
                        raise RuntimeError(
                            "VLM_STRUCTURE_UNRESOLVED: teacher question tree is empty"
                        )
                    if package.structure_audit.get("status") != "COMPLETE":
                        raise RuntimeError(
                            "TREE_UNRESOLVED: "
                            + json.dumps(package.structure_audit.get("failures") or [],
                                         ensure_ascii=False)
                        )
                else:
                    strategies.append("inherited_teacher_topology+student_vlm_evidence")
                package.exam_id = doc_id
                package.warnings.extend(prep_warnings)
                # Preserve order while avoiding duplicate strategy labels.
                strategy = "+".join(dict.fromkeys(strategies))
                package.warnings.append(f"提取策略: {strategy}")
                if role == "teacher":
                    if not any(question.items for section in package.sections for question in section.questions):
                        raise ValueError(f"教师卷 {doc_id} 未提取到题目，不能生成 Golden")
                    if cached_tree is None:
                        package.golden_source = f"teacher:{doc_id}"
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
                    if teacher_package is None:
                        raise ValueError(
                            f"学科 {subject} 缺少可用教师卷，无法生成 Golden 并锁定学生题目拓扑"
                        )
                    package = golden_service.inherit_student_topology(
                        teacher_package, package, source_pages, pages, doc_id,
                        student_id or "",
                    )
                    record_phase("question_tree_generation", "COMPLETED", source="inherited_teacher_tree")

                stage = "visual_page_extraction"
                record_phase(
                    "visual_page_extraction", "STARTED",
                    provider=visual_extraction_provider,
                    evidence="current_document_original_pages",
                )
                semantic_summary = visual_extraction_service.extract(
                    package, pages, output / "visual_extraction" / safe_doc_id
                )
                record_phase("visual_page_extraction", "COMPLETED", summary=semantic_summary)
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
                package.warnings.append(
                    "原页 VLM 联合提取题目区域、逻辑槽位、最终坐标和答案；"
                    "本流程不运行本地 OCR"
                )

                # Persist crops only after final per-document VLM geometry.
                stage = "roi_generation"
                from exam_pipeline.roi import RoIPatchGenerator
                roi_records = RoIPatchGenerator().generate_package(
                    package, pages, output / "roi_patches" / safe_doc_id
                )

                if role == "teacher":
                    stage = "teacher_answer_extraction"
                    record_phase("answer_recognition", "STARTED", role="teacher")
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
                        "kind": "manual_override" if override_tree else "whole_document_vlm",
                        "golden_source": package.golden_source,
                        "override_fingerprint": (
                            override_tree.get("fingerprint") if override_tree else None
                        ),
                        "semantic_model": model if visual_tree else None,
                        "semantic_candidate_gate": "vlm_schema_page_coverage",
                        "stem_source": "original_pages",
                        "coordinate_authority": "vlm_original_page_pixels",
                        "ocr_used": False,
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
                        tree_cache_name = "exam_tree_vlm_only_v1"
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
                    from exam_pipeline.visual_extraction import visual_answer_summary
                    visual_answers = visual_answer_summary(package)
                    document_audit = {"summary": visual_answers, "pages": {}}
                    record_phase(
                        "answer_recognition", "COMPLETED",
                        summary=visual_answers,
                        authority="vlm_original_page",
                    )
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
            result["ocr"] = []
            result["recognition_mode"] = "vlm_only"
            result["ocr_used"] = False
            result["coordinate_authority"] = "vlm_original_page_pixels"
            result["answer_authority"] = "vlm_original_page"
            result["preprocessing"] = preprocessing
            result["phase_trace"] = phase_trace
            result["pipeline_order"] = manifest["runtime"]["pipeline_order"]
            result["performance"] = performance_collector.summary(doc_id)
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
    if _write_manifest:
        from exam_pipeline.io_utils import atomic_write_json
        atomic_write_json(output / "manifest.json", manifest)
    if _return_context:
        return manifest, teacher_packages, teacher_pages
    return manifest


def _merge_performance_reports(reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine per-worker telemetry without mixing document identities."""
    operations: Dict[str, Dict[str, Any]] = {}
    request_count = cumulative = failures = 0
    for report in reports:
        if not isinstance(report, dict):
            continue
        request_count += int(report.get("request_count", 0) or 0)
        cumulative += float(report.get("cumulative_operation_seconds", 0) or 0)
        failures += int(report.get("failures", 0) or 0)
        for name, source in (report.get("operations") or {}).items():
            target = operations.setdefault(name, {
                "request_count": 0, "duration_seconds": 0.0,
                "image_count": 0, "image_bytes": 0, "attempts": 0,
                "failures": 0, "cache_hits": 0,
            })
            for key in ("request_count", "image_count", "image_bytes", "attempts",
                        "failures", "cache_hits"):
                target[key] += int(source.get(key, 0) or 0)
            target["duration_seconds"] += float(source.get("duration_seconds", 0) or 0)
    for value in operations.values():
        value["duration_seconds"] = round(value["duration_seconds"], 4)
        value["cache_hit_rate"] = round(
            value["cache_hits"] / max(1, value["request_count"]), 4)
    return {
        "request_count": request_count,
        "cumulative_operation_seconds": round(cumulative, 4),
        "failures": failures,
        "operations": dict(sorted(operations.items())),
    }


def process(
        dataset: Path, output: Path, model: str = DOUBAO_MODEL,
        api_key: Optional[str] = DOUBAO_API_KEY,
        endpoint: str = DOUBAO_RESPONSES_ENDPOINT, *, dry_run: bool = False,
        include_scan: bool = False, timeout: int = EXAM_REQUEST_TIMEOUT,
        limit: Optional[int] = None,
        subject_filter: Optional[str] = None,
        student_id_filter: Optional[str] = None,
        exam_tree_overrides: Optional[Path] = None,
        production_mode: bool = False,
        student_workers: Optional[int] = None,
        page_workers: Optional[int] = None) -> Dict[str, Any]:
    """Run teachers first, then process independent student documents concurrently."""
    workers = int(student_workers if student_workers is not None else os.getenv(
        "EXAM_STUDENT_WORKERS", "3"))
    if workers < 1:
        raise ValueError("student_workers 必须大于 0")
    pages_in_flight = int(page_workers if page_workers is not None else os.getenv(
        "EXAM_PAGE_WORKERS", "2"))
    if pages_in_flight < 1:
        raise ValueError("page_workers 必须大于 0")

    all_docs = discover_documents(dataset, include_scan)
    if subject_filter:
        all_docs = [entry for entry in all_docs if entry[0] == subject_filter]
        if not all_docs:
            raise ValueError(f"未找到学科: {subject_filter}")
    selected_students = [entry for entry in all_docs if entry[1] == "student"]
    if student_id_filter and student_id_filter != "teacher":
        selected_students = [
            entry for entry in selected_students
            if entry[2][0].parent.name == student_id_filter
        ]
        if not selected_students:
            scope = f"学科 {subject_filter} 中" if subject_filter else ""
            raise ValueError(f"{scope}未找到学生卷或教师卷: {student_id_filter}")

    # Preserve the existing global --limit semantics in the serial path. A
    # limited run is commonly used for debugging and should not silently skip
    # the teacher or reorder the selected documents.
    if dry_run or workers == 1 or limit is not None or not selected_students:
        return _process_serial(
            dataset, output, model, api_key, endpoint,
            dry_run=dry_run, include_scan=include_scan, timeout=timeout,
            limit=limit, subject_filter=subject_filter,
            student_id_filter=student_id_filter,
            exam_tree_overrides=exam_tree_overrides,
            production_mode=production_mode,
            _page_workers=pages_in_flight,
        )

    started = time.monotonic()
    # The teacher pass is deliberately complete before any student worker is
    # started. Its in-memory packages carry the Golden geometry and semantic
    # slot plan used by all student workers.
    teacher_result = _process_serial(
        dataset, output, model, api_key, endpoint,
        dry_run=False, include_scan=include_scan, timeout=timeout,
        subject_filter=subject_filter, student_id_filter="teacher",
        exam_tree_overrides=exam_tree_overrides,
        production_mode=production_mode, _return_context=True,
        _page_workers=pages_in_flight,
    )
    teacher_manifest, teacher_packages, teacher_pages = teacher_result
    teacher_manifest.setdefault("runtime", {}).setdefault("concurrency", {}).update({
        "student_workers": workers,
        "page_workers": pages_in_flight,
    })

    worker_args = []
    for subject, _, paths in selected_students:
        worker_args.append((subject, paths[0].parent.name))

    def run_student(subject: str, student_id: str):
        return _process_serial(
            dataset, output, model, api_key, endpoint,
            dry_run=False, include_scan=include_scan, timeout=timeout,
            subject_filter=subject, student_id_filter=student_id,
            exam_tree_overrides=exam_tree_overrides,
            production_mode=production_mode, _student_only=True,
            _write_manifest=False,
            _page_workers=pages_in_flight,
            _shared_teacher_packages=teacher_packages,
            _shared_teacher_pages=teacher_pages,
        )

    worker_manifests = []
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="exam-student") as pool:
        futures = {
            pool.submit(run_student, subject, student_id): (subject, student_id)
            for subject, student_id in worker_args
        }
        for future in as_completed(futures):
            subject, student_id = futures[future]
            try:
                worker_manifests.append(future.result())
            except Exception as exc:
                # _process_serial normally captures document errors in its
                # manifest. This guard also records worker-level failures.
                teacher_manifest.setdefault("errors", []).append({
                    "id": f"{subject}__student__{student_id}",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "stage": "student_worker",
                })

    manifest = teacher_manifest
    manifest.setdefault("runtime", {})["input_filter"] = {
        "subject": subject_filter,
        "student_id": student_id_filter,
    }
    manifest["documents"].extend(
        document for worker in worker_manifests
        for document in worker.get("documents", [])
    )
    document_order = {
        f"{subject}__student__{student_id}": index
        for index, (subject, student_id) in enumerate(worker_args)
    }
    manifest["documents"].sort(
        key=lambda document: (
            0 if document.get("role") == "teacher" else 1,
            document_order.get(document.get("id"), len(document_order)),
        )
    )
    manifest["errors"].extend(
        error for worker in worker_manifests
        for error in worker.get("errors", [])
    )
    manifest["duration_seconds"] = round(time.monotonic() - started, 3)
    manifest["completed_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    manifest["performance"] = _merge_performance_reports(
        [teacher_manifest.get("performance", {})]
        + [worker.get("performance", {}) for worker in worker_manifests]
    )
    manifest["visual_extraction_summary"] = {
        "documents": len(manifest["documents"]),
        "page_calls": sum(
            (entry.get("summary") or {}).get("page_calls", 0)
            for document in manifest["documents"]
            for entry in document.get("phase_trace", [])
            if entry.get("phase") == "visual_page_extraction"
            and entry.get("status") == "COMPLETED"
        ),
        "page_retries": sum(
            (entry.get("summary") or {}).get("page_retries", 0)
            for document in manifest["documents"]
            for entry in document.get("phase_trace", [])
            if entry.get("phase") == "visual_page_extraction"
            and entry.get("status") == "COMPLETED"
        ),
        "failed_pages": sum(
            (entry.get("summary") or {}).get("failed_pages", 0)
            for document in manifest["documents"]
            for entry in document.get("phase_trace", [])
            if entry.get("phase") == "visual_page_extraction"
            and entry.get("status") == "COMPLETED"
        ),
        "ocr_used": False,
    }
    manifest["status"] = "COMPLETED_WITH_ERRORS" if manifest["errors"] else "COMPLETED"
    manifest["summary"] = {
        "documents_succeeded": len(manifest["documents"]),
        "documents_failed": len(manifest["errors"]),
        "documents_needing_review": sum(
            document.get("quality_status") == "NEED_REVIEW"
            for document in manifest["documents"]
        ),
    }
    from exam_pipeline.io_utils import atomic_write_json
    atomic_write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="从教师卷/学生卷扫描图提取题目级 JSON")
    parser.add_argument("dataset", type=Path, help="数据集根目录；当前扫描样本使用 '扫描图片'")
    parser.add_argument("-o", "--output", type=Path, default=Path("extracted"))
    parser.add_argument("--model", default=DOUBAO_MODEL, help="豆包模型名或火山方舟 Endpoint ID")
    parser.add_argument("--endpoint", default=DOUBAO_RESPONSES_ENDPOINT, help="火山方舟 Responses API 地址")
    parser.add_argument("--api-key", default=DOUBAO_API_KEY,
                        help="优先使用 DOUBAO_API_KEY 环境变量；仅建议开发调试时临时传参")
    parser.add_argument("--include-scan", action="store_true", help="教师目录同时处理扫描_ 图片")
    parser.add_argument("--dry-run", action="store_true", help="只扫描分组并生成占位 JSON，不调用模型")
    parser.add_argument("--timeout", type=int, default=EXAM_REQUEST_TIMEOUT)
    parser.add_argument(
        "--student-workers", type=int,
        default=int(os.getenv("EXAM_STUDENT_WORKERS", "3")),
        help="学生卷并发数；默认 3，设置为 1 可关闭学生卷并发",
    )
    parser.add_argument(
        "--page-workers", type=int,
        default=int(os.getenv("EXAM_PAGE_WORKERS", "2")),
        help="单份卷页面并发数；默认 2，设置为 1 可关闭页面并发",
    )
    parser.add_argument("--limit", type=int, help="只处理前 N 份文档，用于小规模验证；默认处理全部文档")
    parser.add_argument("--subject", help="只处理指定学科")
    parser.add_argument("--student-id", help="只处理指定学生卷；会同时加载该学科教师卷以建立题目拓扑")
    parser.add_argument(
        "--exam-tree-overrides", type=Path,
        help="人工修订 ExamTree JSON 或目录（目录内按 <subject>.json 命名）；校验通过后供全部学生复用",
    )
    parser.add_argument("--fail-on-review", action="store_true",
                        help="有任何文档未通过质量门时返回码 2，适用于生产 CI/调度")
    parser.add_argument("--hitl-workspace", type=Path,
                        help="提取成功后直接导出到 HITL workspace（可选）")
    parser.add_argument("--hitl-overwrite", action="store_true",
                        help="允许 --hitl-workspace 覆盖已有同名批次")
    parser.add_argument(
        "--production", action="store_true",
        help="生产强校验：要求完整 VLM 结构、槽位坐标和答案证据",
    )
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default=os.getenv("EXAM_LOG_LEVEL", "INFO"), help="运行日志级别")
    args = parser.parse_args(argv)
    if args.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        parser.error(f"无效日志级别: {args.log_level}")
    if args.exam_tree_overrides and not args.exam_tree_overrides.exists():
        parser.error(f"ExamTree 修订文件或目录不存在: {args.exam_tree_overrides}")
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
        dry_run=args.dry_run, include_scan=args.include_scan,
        timeout=args.timeout, limit=args.limit, subject_filter=args.subject,
        student_id_filter=args.student_id,
        exam_tree_overrides=args.exam_tree_overrides,
        production_mode=args.production,
        student_workers=args.student_workers,
        page_workers=args.page_workers,
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
