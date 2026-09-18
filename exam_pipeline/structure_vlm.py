"""Structure-only PaddleOCR-VL adapter.

The adapter deliberately treats a document VLM as a semantic proposer.  Any
geometry returned by the model is removed before the result enters the exam
pipeline; PP-OCR and the deterministic grounding stages remain the coordinate
authority.
"""
from __future__ import annotations

import base64
import copy
import json
import logging
import mimetypes
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


LOGGER = logging.getLogger("exam_pipeline.structure_vlm")

# These fields can influence physical localization and therefore must never be
# accepted from a structure-only model response.
GEOMETRY_FIELDS = frozenset({
    "bbox", "box", "polygon", "poly", "points", "coordinate", "coordinates",
    "stem_bbox", "stem_box", "option_bboxes", "option_boxes", "blank_bboxes",
    "blank_boxes", "writing_bboxes", "writing_boxes", "writing_box",
    "answer_bbox", "answer_box", "answer_regions", "student_regions",
    "diagram_boxes", "diagrams", "slot", "slots", "expected_bbox",
    "handwriting_bbox", "ink_bbox",
})


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return "data:{};base64,{}".format(mime, encoded)


def _extract_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        value = "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in value
        )
    text = str(value or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise ValueError("PaddleOCR-VL 未返回合法 JSON")
        result = json.loads(match.group(0))
    if not isinstance(result, dict):
        raise ValueError("PaddleOCR-VL 结构结果必须是 JSON 对象")
    return result


def structure_from_ocr_text(text: str, role: str, subject: str,
                            page_index: int = 1) -> Dict[str, Any]:
    """Build a geometry-free exam hierarchy from PaddleOCR-VL OCR output.

    PaddleOCR-VL is a document recognition model rather than a general
    instruction model.  Its supported whole-page task returns reading-order
    text, so hierarchy construction is deliberately deterministic here.
    """
    lines = [re.sub(r"\s+", " ", line).strip()
             for line in str(text or "").splitlines() if line.strip()]
    top_pattern = re.compile(r"^\s*(\d{1,3})\s*[\.、．:：\)]\s*(.*)$")
    sub_pattern = re.compile(r"^\s*[\(（](\d{1,2})[\)）]\s*(.*)$")
    anchors = [(index, int(match.group(1)))
               for index, line in enumerate(lines)
               if (match := top_pattern.match(line))]
    questions: List[Dict[str, Any]] = []
    answer_key = "standard_answer" if role == "teacher" else "answer"

    def infer_type(value: str) -> str:
        if any(token in value for token in ("A.", "A、", "A．", "选择", "正确的一项")):
            return "choice"
        if any(token in value for token in ("填空", "____", "横线上", "括号里")):
            return "fill"
        if any(token in value for token in ("作文", "写作", "不少于")):
            return "essay"
        if any(token in value for token in ("阅读", "材料", "语段")):
            return "reading"
        return "solve"

    for anchor_pos, (start, number) in enumerate(anchors):
        end = anchors[anchor_pos + 1][0] if anchor_pos + 1 < len(anchors) else len(lines)
        question_lines = lines[start:end]
        if not question_lines:
            continue
        question_text = "\n".join(question_lines)
        item_starts = [(offset, int(match.group(1)))
                       for offset, line in enumerate(question_lines)
                       if (match := sub_pattern.match(line))]
        items: List[Dict[str, Any]] = []
        if item_starts:
            for item_pos, (item_start, sub_number) in enumerate(item_starts):
                item_end = (item_starts[item_pos + 1][0]
                            if item_pos + 1 < len(item_starts) else len(question_lines))
                item_text = "\n".join(question_lines[item_start:item_end])
                items.append({
                    "item_id": "q{}_{}".format(number, sub_number),
                    "item_name": "{}.({})".format(number, sub_number),
                    "type": infer_type(item_text),
                    "question_text": item_text,
                    answer_key: None,
                    "score": None,
                    "slot_count": None,
                    "confidence": 0.72,
                    "page": page_index,
                })
        else:
            items.append({
                "item_id": "q{}".format(number),
                "item_name": "第{}题".format(number),
                "type": infer_type(question_text),
                "question_text": question_text,
                answer_key: None,
                "score": None,
                "slot_count": None,
                "confidence": 0.72,
                "page": page_index,
            })
        questions.append({
            "question_id": "q{}".format(number),
            "question_num": number,
            "question_title": question_text,
            "type": infer_type(question_text),
            "items": items,
        })
    if not questions:
        raise ValueError("PaddleOCR-VL 文本中未识别到题号锚点")
    return {
        "document_type": role,
        "subject": subject,
        "pages": [{"page": page_index}],
        "sections": [{
            "section_id": "page_{}".format(page_index),
            "section_title": "试题",
            "questions": questions,
        }],
        "warnings": ["PaddleOCR-VL 使用官方 OCR 任务输出；题目层级由确定性题号解析器生成"],
    }


def remove_model_geometry(result: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Deep-copy a model result and remove every geometry-bearing field."""
    removed: List[str] = []

    def clean(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            output = {}
            for key, child in value.items():
                child_path = "{}.{}".format(path, key) if path else str(key)
                if str(key).lower() in GEOMETRY_FIELDS:
                    removed.append(child_path)
                    continue
                output[key] = clean(child, child_path)
            return output
        if isinstance(value, list):
            return [clean(child, "{}[{}]".format(path, index))
                    for index, child in enumerate(value)]
        return copy.deepcopy(value)

    sanitized = clean(result, "")
    return sanitized, removed


class PaddleOCRVLStructureClient:
    """OpenAI-compatible client for a separately deployed PaddleOCR-VL service."""

    def __init__(self, endpoint: str, model: str = "PaddlePaddle/PaddleOCR-VL-1.6",
                 api_key: str = "", timeout: int = 180, max_attempts: int = 3):
        endpoint = str(endpoint or "").strip()
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("PaddleOCR-VL endpoint 必须是 HTTP(S) URL")
        if timeout <= 0:
            raise ValueError("PaddleOCR-VL timeout 必须大于 0")
        if max_attempts < 1:
            raise ValueError("PaddleOCR-VL max_attempts 必须大于 0")
        self.endpoint = endpoint
        self.model = str(model or "PaddlePaddle/PaddleOCR-VL-1.6")
        self.api_key = str(api_key or "")
        self.timeout = int(timeout)
        self.max_attempts = int(max_attempts)

    def analyze(self, prompt: str, image_paths: Sequence[Path]) -> Dict[str, Any]:
        paths = [Path(path) for path in image_paths]
        if not paths:
            raise ValueError("PaddleOCR-VL 至少需要一张页面图像")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError("PaddleOCR-VL 输入图像不存在: {}".format(path))

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend({
            "type": "image_url",
            "image_url": {"url": _data_url(path)},
        } for path in paths)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 12000,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer {}".format(self.api_key)
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        raw: Dict[str, Any] = {}
        for attempt in range(1, self.max_attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt == self.max_attempts:
                    raise RuntimeError(
                        "PaddleOCR-VL HTTP {}: {}".format(exc.code, body[:500])
                    ) from exc
                LOGGER.warning("paddleocr_vl_retry attempt=%d status=%d", attempt, exc.code)
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt == self.max_attempts:
                    raise RuntimeError("PaddleOCR-VL 网络请求失败: {}".format(exc)) from exc
                LOGGER.warning("paddleocr_vl_retry attempt=%d reason=network", attempt)
            time.sleep(min(4.0, 0.5 * (2 ** (attempt - 1))))

        try:
            content_value = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("PaddleOCR-VL 响应缺少 choices[0].message.content") from exc
        try:
            return _extract_json(content_value)
        except ValueError:
            # The official PaddleOCR-VL OCR task normally returns plain text,
            # not instruction-following JSON.  Preserve it for deterministic
            # hierarchy parsing by the caller.
            return {"_paddleocr_vl_ocr_text": str(content_value or "")}


def structure_only_prompt(base_prompt: str, role: str = "teacher") -> str:
    """Return PaddleOCR-VL's canonical whole-page recognition task prompt."""
    # PaddleOCR-VL supports task prompts such as OCR:/Spotting: and is not a
    # general JSON instruction model.  Geometry remains outside this adapter.
    return "OCR:"


__all__ = [
    "GEOMETRY_FIELDS", "PaddleOCRVLStructureClient", "remove_model_geometry",
    "structure_from_ocr_text", "structure_only_prompt",
]
