"""Layout-first, language-tolerant multi-slot topology discovery."""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .contracts import ExamItem, ExamPackage, OCRBlock, Page, PageRegion, Slot
from .layout import QuestionLayoutService
from .roi import xyxy_to_yxyx
from .reading_order import ordered_slots


_CHOICE_TYPES = ("choice", "single", "multiple", "mcq", "judgment", "true_false", "选择", "判断")
_FILL_TYPES = ("fill", "blank", "cloze", "completion", "填空")
_GRID_TYPES = ("grid", "tianzige", "田字格", "答题卡")
_LARGE_TYPES = ("composition", "essay", "drawing", "proof", "作文", "作图", "证明", "solve", "large_writing")
_OPTION_LABEL = re.compile(r"(?:^|\s)[A-HＡ-Ｈ]\s*[.、．:)）]", re.IGNORECASE)
_BRACKET = re.compile(r"[\(（\[【]([^\)）\]】]{0,24})[\)）\]】]")
_SCORE = re.compile(
    r"^\s*\d+(?:\.\d+)?\s*(?:分|points?|pts?|marks?|점|点)\s*$", re.IGNORECASE
)
_BLANK_TEXT = re.compile(r"_{2,}|\.{4,}|…{2,}|[\(（\[]\s*[_·.\s]*[\)）\]]")
_PRINTED_RULE_TEXT = re.compile(r"_{2,}|\.{4,}|…{2,}")
_CHOICE_STEM_TEXT = re.compile(
    r"(?:下列|以下).{0,80}(?:正确|错误|属于|符合|不符合|说法|做法)|"
    r"(?:选择|选出).{0,40}(?:一项|答案)", re.IGNORECASE,
)
# Anonymous layout anchors deliberately inspect punctuation and repetition, not
# label meaning.  For example, four occurrences of “<short label>” + delimiter
# imply four answer points in Chinese, English, or mixed-language papers.
_QUOTED_LABEL_ANCHOR = re.compile(
    r"[“\"‘']\s*[^”\"’'\n]{1,24}?\s*[”\"’']\s*(?:[:：=]|is\b|是)", re.IGNORECASE
)
_DELIMITED_LABEL_ANCHOR = re.compile(
    r"(?:^|[\s,，;；])[^\s,，;；:：=]{1,12}\s*[:：=](?!=)", re.MULTILINE
)


def _iou(a, b):
    y1, x1, y2, x2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, y2-y1) * max(0, x2-x1)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    ab = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / max(1, aa + ab - inter)


def _normalized(value: Any) -> str:
    return re.sub(r"[\W_]+", "", str(value or "").casefold())


def _answer_parts(item: ExamItem) -> List[str]:
    answer = item.standard_answer
    if isinstance(answer, dict):
        values = list(answer.values())
    elif isinstance(answer, (list, tuple)):
        values = list(answer)
    elif answer is None:
        values = []
    else:
        values = str(answer).split("；")
    return [str(value).strip() if value is not None else None for value in values]


def _expected_text(item: ExamItem, index: int) -> Optional[str]:
    values = _answer_parts(item)
    return values[index] if index < len(values) else None


def _teacher_answer_count(item: ExamItem) -> int:
    """Conservatively derive cardinality from a structured teacher answer."""
    answer = item.standard_answer
    if answer is None:
        return 0
    if isinstance(answer, dict):
        # A flat keyed mapping naturally represents independent answer points;
        # nested structures often contain steps/rubrics for one response.
        return len(answer) if answer and all(
            not isinstance(value, (dict, list, tuple)) for value in answer.values()
        ) else 1
    if isinstance(answer, (list, tuple)):
        return max(1, len(answer))
    kind = infer_item_type(item.question_text, item.item_type)
    if kind in {"fill", "grid"}:
        return max(1, len(_answer_parts(item)))
    return 1


def _leading_number(text: str) -> Optional[int]:
    """Return a printed question number when a block starts with one.

    支持多种题号格式以提高通用性:
    - 标准格式: "1.", "1、", "1）", "1:"
    - 括号格式: "(1)", "（1）", "[1]"
    - 无分隔符: "1 " (后面跟空格或文字)
    - 带前缀: "第1题", "题1"
    """
    text = str(text or "").strip()
    if not text:
        return None

    # 优先级1: 标准题号格式 "1.", "1、", "1）", "1:"
    match = re.match(r"^\s*(\d{1,3})\s*[.、．:：)）]", text)
    if match:
        return int(match.group(1))

    # 优先级2: 括号格式 "(1)", "（1）", "[1]"
    match = re.match(r"^\s*[\(（\[](\d{1,3})[\)）\]]", text)
    if match:
        return int(match.group(1))

    # 优先级3: 带前缀格式 "第1题", "题1", "第1小题"
    match = re.match(r"^\s*(?:第\s*)?(?:题\s*)?(\d{1,3})\s*(?:题|小题)?[.、．:：\s]", text)
    if match:
        return int(match.group(1))

    # 优先级4: 单独数字后跟空格或非数字 "1 在平面..."
    match = re.match(r"^\s*(\d{1,2})(?:\s+[^\d])", text)
    if match:
        num = int(match.group(1))
        # 避免误匹配年份等: 只接受1-99
        if 1 <= num <= 99:
            return num

    return None


def _item_question_number(item: ExamItem) -> Optional[int]:
    """Recover the physical question number from an item.

    多级回退策略以提高通用性:
    1. 从 item_id 提取 (如 "q11_2" -> 11)
    2. 从 item_name 提取 (如 "q1", "Q1" -> 1)
    3. 从 question_text 提取 (如 "1.在平面..." -> 1)
    4. 从 question_num 字段提取
    """
    # 优先级1: 从 item_id 提取 "q11_2" -> 11
    match = re.match(r"^q(\d{1,3})", str(item.item_id or ""), re.I)
    if match:
        return int(match.group(1))

    # 优先级2: 从 item_name 提取 "q1", "Q1" -> 1
    match = re.match(r"^q(\d{1,3})", str(item.item_name or ""), re.I)
    if match:
        return int(match.group(1))

    # 优先级3: 从 question_text 开头提取题号
    text = str(item.question_text or "")
    if text:
        # 尝试用 _leading_number 提取
        num = _leading_number(text)
        if num is not None:
            return num

    # 优先级4: 从 question_num 字段直接获取
    if hasattr(item, 'question_num') and item.question_num:
        try:
            return int(item.question_num)
        except (ValueError, TypeError):
            pass

    return None


def _student_anchor_score(item: ExamItem, block: OCRBlock) -> float:
    """Score one student OCR block as the physical anchor of a logical item.

    Deliberately text-only.  The teacher owns question identity; the student
    page owns every coordinate.  A matching printed question number is the
    strongest cue and stem similarity carries the remainder.

    改进点 (v3 - 通用性增强):
    1. 题号匹配权重从 40% 提升到 70% - 题号是最可靠的跨版本匹配依据
    2. 文本相似度权重从 60% 降低到 30% - 学生卷OCR可能与教师卷有显著差异
    3. 精确题号匹配直接返回高分 - 避免被低文本相似度拖累
    4. 添加短文本特殊处理 - 纯题号块(如 "1.") 也能正确匹配
    """
    query = _normalized(item.question_text or item.item_name)
    text = _normalized(block.text)

    # 提取题号
    item_number = _item_question_number(item)
    block_number = _leading_number(block.text)

    # 策略1: 精确题号匹配 (最高优先级)
    # 如果题号完全匹配,直接给高分,不依赖文本相似度
    if item_number is not None and block_number is not None and item_number == block_number:
        # 检查是否是纯题号块(如 "1.", "2、")
        block_text_stripped = str(block.text or "").strip()
        if len(block_text_stripped) <= 10:  # 纯题号或极短文本
            return 0.95  # 高分但不是满分,留给包含题干的完整匹配

        # 题号匹配 + 有一定文本内容
        # 计算文本相似度作为置信度加成
        if query and text:
            similarity = SequenceMatcher(None, query[:120], text[:120]).ratio()
            containment = 1.0 if ((query[:10] and query[:10] in text)
                                  or (text[:10] and text[:10] in query)) else 0.0
            text_bonus = max(similarity, containment) * 0.3
            return min(1.0, 0.70 + text_bonus)  # 基础0.70 + 最多0.30文本加成
        else:
            return 0.70  # 题号匹配但无法计算文本相似度

    # 策略2: 无题号或题号不匹配时,依赖文本相似度
    if not query or not text:
        return 0.0

    # 文本相似度 (30%)
    similarity = SequenceMatcher(None, query[:120], text[:120]).ratio()
    containment = 1.0 if ((query[:10] and query[:10] in text)
                          or (text[:10] and text[:10] in query)) else 0.0
    text_score = max(similarity, containment)

    # 题号匹配 (70%)
    if item_number is not None and block_number is not None:
        number_bonus = 1.0 if item_number == block_number else 0.0
    else:
        number_bonus = 0.0  # 无法提取题号,降级为纯文本匹配

    # 加权: 题号70% + 文本30% (原来是 题号60% + 文本40%)
    return 0.30 * text_score + 0.70 * number_bonus


def infer_item_type(text: str, declared: str = "") -> str:
    """Infer broad answer topology from script-independent layout tokens."""
    raw, kind = str(text or ""), str(declared or "").casefold()
    if len(_OPTION_LABEL.findall(raw)) >= 2:
        return "choice"
    # A literal underline is a fill response even when an upstream model calls
    # it choice (for example "填序号").  An empty choice bracket alone is not
    # a fill blank: image-based options may have no OCR-visible A/B/C labels.
    if _PRINTED_RULE_TEXT.search(raw):
        return "fill"
    if any(token in kind for token in _GRID_TYPES):
        return "grid"
    if any(token in kind for token in _CHOICE_TYPES):
        return "choice"
    if _BLANK_TEXT.search(raw) and _CHOICE_STEM_TEXT.search(raw):
        return "choice"
    if _BLANK_TEXT.search(raw):
        return "fill"
    if any(token in kind for token in _FILL_TYPES):
        return "fill"
    if any(token in kind for token in _LARGE_TYPES):
        return "large_writing"
    return kind or "other"


@dataclass
class SlotCandidate:
    bbox: List[int]
    slot_type: str
    confidence: float
    expected_hint: Optional[str] = None
    evidence: str = ""
    component_count: int = 1


class MultiSlotTopologyEngine:
    MIN_CANDIDATE_CONFIDENCE = 0.55  # Lowered from 0.62 to allow more fill-blank candidates

    def __init__(self, policy=None):
        self.policy = policy

    @staticmethod
    def _effective_kind(item: ExamItem) -> str:
        return infer_item_type(item.question_text, item.item_type)

    @classmethod
    def _is_large_writing_item(cls, item):
        return cls._effective_kind(item) == "large_writing"

    @staticmethod
    def _local_align_reference(student_crop: Any, reference_crop: Any) -> Tuple[Any, Dict[str, Any]]:
        """Refine residual alignment inside one answer search window."""
        import cv2
        import numpy as np
        if (student_crop is None or reference_crop is None
                or student_crop.shape[:2] != reference_crop.shape[:2]
                or min(student_crop.shape[:2]) < 16):
            return reference_crop, {"status": "SKIPPED"}
        student_gray = cv2.cvtColor(student_crop, cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(reference_crop, cv2.COLOR_BGR2GRAY)
        student_edges = cv2.Canny(student_gray, 60, 180).astype(np.float32)
        reference_edges = cv2.Canny(reference_gray, 60, 180).astype(np.float32)
        try:
            (dx, dy), response = cv2.phaseCorrelate(reference_edges, student_edges)
        except cv2.error:
            return reference_crop, {"status": "FAILED"}
        height, width = student_gray.shape[:2]
        limit_x, limit_y = max(3.0, width * .12), max(3.0, height * .12)
        if response < .03 or abs(dx) > limit_x or abs(dy) > limit_y:
            return reference_crop, {
                "status": "REJECTED", "response": round(float(response), 4),
                "shift": [round(float(dx), 2), round(float(dy), 2)],
            }
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        aligned = cv2.warpAffine(
            reference_crop, matrix, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned, {
            "status": "ALIGNED", "response": round(float(response), 4),
            "shift": [round(float(dx), 2), round(float(dy), 2)],
        }

    @classmethod
    def _residual_candidates(cls, item: ExamItem, image: Any, reference_image: Any,
                             inherited_slots: Sequence[Slot], page: Page,
                             reference_kind: str = "teacher",
                             reference_confidence: Optional[float] = None) -> List[SlotCandidate]:
        """Discover student ink near topology hints using aligned image residuals."""
        import cv2
        from .ink_separation import NoBlankInkSeparator
        if (reference_image is None or image is None
                or reference_image.shape[:2] != image.shape[:2]):
            return []
        height, width = image.shape[:2]
        kind = cls._effective_kind(item)
        result: List[SlotCandidate] = []
        references = list(inherited_slots)
        region_fallback = not references
        if region_fallback:
            search_regions = list(cls._regions(item))
            if item.stem_region is not None:
                stem = item.stem_region
                left, top, right, bottom = [
                    int(round(value)) for value in stem.bbox[:4]
                ]
                if kind in {"choice", "fill", "grid"}:
                    vertical_before = max(16, int(.015 * height))
                    vertical_after = max(30, int(.035 * height))
                else:
                    vertical_before = max(12, int(.01 * height))
                    vertical_after = max(120, int(.22 * height))
                stem_search = PageRegion(
                    stem.page_index, stem.page_file,
                    [max(0, left-20), max(0, top-vertical_before),
                     min(width, right+20), min(height, bottom+vertical_after)],
                    stem.confidence, stem.ocr_text,
                )
                stem_center = (top+bottom) / 2.0
                nearby = any(
                    region.page_index == stem.page_index
                    and abs((region.bbox[1]+region.bbox[3])/2.0-stem_center)
                    <= .35 * height
                    for region in search_regions if len(region.bbox) >= 4
                )
                if not nearby:
                    search_regions.append(stem_search)
            for region in search_regions:
                if region.page_index == page.index and len(region.bbox) >= 4:
                    references.append(Slot(
                        len(references)+1, "search_region", item.item_id,
                        xyxy_to_yxyx(region.bbox), page.index,
                        _expected_text(item, len(references)),
                    ))
        target_count, _ = cls._resolve_slot_count(item, inherited_slots)
        for reference in references:
            if reference.page_index != page.index or len(reference.expected_bbox) < 4:
                continue
            y1, x1, y2, x2 = [int(round(value)) for value in reference.expected_bbox[:4]]
            box_h, box_w = max(1, y2-y1), max(1, x2-x1)
            pad_y = max(12, min(80, int(round(box_h * .20))))
            pad_x = max(16, min(100, int(round(box_w * .15))))
            sy1, sx1 = max(0, y1-pad_y), max(0, x1-pad_x)
            sy2, sx2 = min(height, y2+pad_y), min(width, x2+pad_x)
            student_crop = image[sy1:sy2, sx1:sx2]
            reference_crop = reference_image[sy1:sy2, sx1:sx2]
            if student_crop.size == 0 or reference_crop.size == 0:
                continue
            aligned_reference, local_alignment = cls._local_align_reference(
                student_crop, reference_crop
            )
            separation = NoBlankInkSeparator.separate(
                student_crop, aligned_reference, None, reference_kind,
                reference_confidence,
            )
            def components_from(mask):
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1,
                )
                count, _, stats, centroids = cv2.connectedComponentsWithStats(
                    mask, connectivity=8
                )
                values = []
                for label in range(1, count):
                    bx, by, bw, bh, area = [int(value) for value in stats[label]]
                    if area < 8 or bw < 2 or bh < 3:
                        continue
                    if bw > .92 * (sx2-sx1) and bh > .92 * (sy2-sy1):
                        continue
                    values.append((bx, by, bw, bh, area, centroids[label]))
                return values

            components = components_from(separation["handwriting_mask"])
            single_page_fallback = False
            if not components:
                fallback_separation = NoBlankInkSeparator.separate(
                    student_crop, None, None, "single_page", None,
                )
                components = components_from(
                    fallback_separation["handwriting_mask"]
                )
                single_page_fallback = bool(components)
            if not components:
                continue
            if region_fallback and target_count and target_count > 1:
                primitives = [SlotCandidate(
                    [sy1+value[1], sx1+value[0],
                     sy1+value[1]+value[3], sx1+value[0]+value[2]],
                    "residual_writing", .76,
                    None, "residual_component", 1,
                ) for value in components]
                grouped = cls._partition_underlines(
                    primitives, target_count, page, page.ocr,
                    [region for region in cls._regions(item)
                     if region.page_index == page.index],
                )
                confidence = (.64 if single_page_fallback else
                              .84 if reference_kind == "cohort" else .76)
                result.extend(SlotCandidate(
                    group.bbox, "residual_writing", confidence,
                    _expected_text(item, index),
                    "single_page_anchor_residual_fallback" if single_page_fallback
                    else "cohort_residual_local_registration" if reference_kind == "cohort"
                    else "teacher_residual_local_registration",
                    group.component_count,
                ) for index, group in enumerate(grouped))
                continue
            if kind == "choice":
                # A choice mark is compact; selecting the strongest component
                # prevents unrelated workings in a broad question ROI from
                # turning into a coarse answer box.
                components = [max(
                    components,
                    key=lambda value: value[4] / max(1.0, value[2] * value[3]),
                )]
            lx1 = min(value[0] for value in components)
            ly1 = min(value[1] for value in components)
            lx2 = max(value[0] + value[2] for value in components)
            ly2 = max(value[1] + value[3] for value in components)
            padding = 5
            candidate_box = [
                max(0, sy1 + ly1-padding), max(0, sx1 + lx1-padding),
                min(height, sy1 + ly2+padding), min(width, sx1 + lx2+padding),
            ]
            valid, _ = cls._valid_box(candidate_box, item, page)
            if not valid:
                continue
            base_confidence = (.64 if single_page_fallback else
                               .82 if reference_kind == "cohort" else .74)
            if local_alignment.get("status") == "ALIGNED":
                base_confidence += .04
            result.append(SlotCandidate(
                candidate_box,
                "choice_mark" if kind == "choice" else "residual_writing",
                min(.90, base_confidence), reference.expected_text,
                "single_page_anchor_residual_fallback" if single_page_fallback
                else "cohort_residual_local_registration" if reference_kind == "cohort"
                else "teacher_residual_local_registration",
                len(components),
            ))
        return result

    @staticmethod
    def _anonymous_anchor_count(text: str, policy=None) -> Optional[int]:
        """Infer answer-point count from repeated, language-neutral form cues."""
        raw = str(text or "")
        blank_pattern = (_BLANK_TEXT if not policy or not policy.parameter(
            "exclude_formula_parentheses", False
        ) else re.compile(r"_{2,}|\.{4,}|…{2,}"))
        delimited_matches = [
            match.group(0) for match in _DELIMITED_LABEL_ANCHOR.finditer(raw)
        ]
        if policy and policy.parameter("exclude_formula_parentheses", False):
            delimited_matches = [
                value for value in delimited_matches
                if not re.search(r"[0-9()+*/^<>²³{}\\]", value)
            ]
        counts = [
            len(_QUOTED_LABEL_ANCHOR.findall(raw)),
            len(delimited_matches),
            len(blank_pattern.findall(raw)),
        ]
        result = max(counts, default=0)
        maximum = policy.parameter("max_anonymous_anchor_count", None) if policy else None
        if maximum is not None and result > int(maximum):
            return None
        return result if result >= 2 else None

    @classmethod
    def _resolve_slot_count(cls, item: ExamItem,
                            inherited_slots: Sequence[Slot] = (), policy=None):
        """Resolve cardinality without borrowing any teacher coordinates."""
        try:
            explicit = int(item.expected_slot_count or 0)
        except (TypeError, ValueError):
            explicit = 0
        if explicit > 0:
            return explicit, item.slot_count_source or "vlm"

        if cls._effective_kind(item) == "large_writing":
            return 1, "logical_free_response"
        if cls._effective_kind(item) == "choice":
            return 1, "logical_choice_response"
        answer_count = _teacher_answer_count(item)
        if answer_count:
            return answer_count, "teacher_standard_answer"

        anonymous = cls._anonymous_anchor_count(item.question_text, policy)
        if (anonymous and policy
                and infer_item_type(item.question_text, item.item_type)
                not in {"fill", "grid"}
                and len(_QUOTED_LABEL_ANCHOR.findall(
                    str(item.question_text or "")
                )) < 2):
            # In free-form mathematics, repeated equals signs and formula
            # punctuation are working, not independent answer cavities.
            anonymous = None
        if anonymous:
            return anonymous, "anonymous_layout_anchor"

        confirmed = [slot for slot in inherited_slots
                     if slot.audit.get("cardinality_confirmed")]
        if confirmed and len(confirmed) == len(inherited_slots):
            return len(confirmed), "teacher_confirmed_topology"

        # NEW: 集成基数共识服务 - 解决 localized_slot_count = 0 的问题
        # 当前端检测器找到候选框但VLM不可用时，仍然应该生成槽位
        if hasattr(item, 'cardinality_evidence') and item.cardinality_evidence:
            from exam_pipeline.cardinality import SlotCardinalityConsensusService
            consensus_service = SlotCardinalityConsensusService()
            # 使用共识服务的三方证据解析
            # 注意：这里需要完整的ExamPackage上下文，简化版仅用于填补空缺
            layout_boxes = item.cardinality_evidence.get('layout_boxes', 0)
            vlm_boxes = item.cardinality_evidence.get('vlm_boxes', 0)
            if layout_boxes > 0 or vlm_boxes > 0:
                consensus_count = max(layout_boxes, vlm_boxes)
                if consensus_count > 0:
                    return consensus_count, "cardinality_consensus"

        return None, "geometry_estimate"

    @staticmethod
    def _nearby_anchor_strength(candidate: SlotCandidate,
                                blocks: Sequence[OCRBlock], page: Page) -> float:
        """Return a geometry-only boundary strength immediately before a line."""
        y1, x1, y2, _ = candidate.bbox
        page_width = float(page.width or 1654)
        strength = 0.0
        for block in blocks:
            if len(block.bbox) < 4:
                continue
            block_text = str(block.text or "").strip()
            structural_text = (bool(_QUOTED_LABEL_ANCHOR.search(block_text))
                               or bool(re.search(r"[:：=]\s*$", block_text)))
            if not structural_text:
                continue
            bx1, by1, bx2, by2 = [float(value) for value in block.bbox[:4]]
            overlap = max(0.0, min(y2, by2) - max(y1, by1))
            if overlap / max(1.0, min(y2-y1, by2-by1)) < 0.35:
                continue
            gap = x1 - bx2
            if (-0.02 * page_width <= gap <= 0.06 * page_width
                    and bx2 - bx1 <= 0.35 * page_width):
                strength = max(strength, 0.92)
        return strength

    @classmethod
    def _merge_transition_cost(cls, left: SlotCandidate, right: SlotCandidate,
                               page: Page, blocks: Sequence[OCRBlock],
                               layout_left: float, layout_right: float) -> float:
        """Cost of treating two consecutive physical rules as one answer."""
        if cls._nearby_anchor_strength(right, blocks, page) >= 0.9:
            return 4.0
        height = max(1.0, (left.bbox[2]-left.bbox[0] + right.bbox[2]-right.bbox[0]) / 2.0)
        center_delta = abs((left.bbox[0]+left.bbox[2]) / 2.0
                           - (right.bbox[0]+right.bbox[2]) / 2.0)
        layout_width = max(1.0, layout_right-layout_left)
        if center_delta <= 0.65 * height:
            gap = max(0.0, right.bbox[1]-left.bbox[3])
            return 0.10 + gap / max(1.0, 0.25 * layout_width)

        vertical_gap = max(0.0, right.bbox[0]-left.bbox[2])
        wraps = (left.bbox[1] >= layout_left + 0.45 * layout_width
                 and right.bbox[1] <= layout_left + 0.25 * layout_width)
        if wraps:
            return 0.05 + vertical_gap / max(1.0, 0.15 * float(page.height or 2338))
        return 0.70 + vertical_gap / max(1.0, 0.08 * float(page.height or 2338))

    @classmethod
    def _partition_underlines(cls, candidates: Sequence[SlotCandidate], groups: int,
                              page: Page, blocks: Sequence[OCRBlock],
                              regions: Sequence[PageRegion]) -> List[SlotCandidate]:
        """Optimal contiguous K-partition of physical rules using geometry DP."""
        ordered = sorted(candidates, key=lambda value: (value.bbox[0], value.bbox[1]))
        count = len(ordered)
        if groups <= 0 or groups >= count:
            return list(ordered)
        candidate_left = min(float(value.bbox[1]) for value in ordered)
        candidate_right = max(float(value.bbox[3]) for value in ordered)
        layout_left, layout_right = candidate_left, candidate_right
        if regions:
            region_left = min(float(region.bbox[0]) for region in regions)
            region_right = max(float(region.bbox[2]) for region in regions)
            # Grounding can occasionally span both page columns.  Such a broad
            # ROI must not define line-wrap geometry for a single-column answer.
            if region_right-region_left <= 1.6 * max(1.0, candidate_right-candidate_left):
                layout_left, layout_right = region_left, region_right
        transitions = [cls._merge_transition_cost(
            ordered[index], ordered[index+1], page, blocks, layout_left, layout_right
        ) for index in range(count-1)]

        def group_cost(start: int, end: int) -> float:
            joins = end-start
            return sum(transitions[start:end]) + 0.03 * joins * joins

        infinity = float("inf")
        dp = [[infinity] * (count+1) for _ in range(groups+1)]
        previous = [[-1] * (count+1) for _ in range(groups+1)]
        dp[0][0] = 0.0
        for group in range(1, groups+1):
            for end in range(group, count+1):
                for start in range(group-1, end):
                    cost = dp[group-1][start] + group_cost(start, end-1)
                    if cost < dp[group][end]:
                        dp[group][end] = cost
                        previous[group][end] = start
        ranges = []
        group, end = groups, count
        while group:
            start = previous[group][end]
            ranges.append((start, end))
            group, end = group-1, start
        ranges.reverse()
        result = []
        for start, end in ranges:
            members = ordered[start:end]
            if len(members) == 1:
                result.append(members[0])
                continue
            result.append(SlotCandidate(
                [min(value.bbox[0] for value in members),
                 min(value.bbox[1] for value in members),
                 max(value.bbox[2] for value in members),
                 max(value.bbox[3] for value in members)],
                "multiline_answer_area",
                max(value.confidence for value in members), None,
                "anonymous_anchor_geometric_dp", len(members),
            ))
        return result

    @classmethod
    def _reconcile_teacher_cardinality(cls, item: ExamItem, slots: Sequence[Slot],
                                       page_map: Dict[int, Page]) -> List[Slot]:
        """Make physical teacher regions obey the semantic answer-point count."""
        target, source = cls._resolve_slot_count(item)
        # The policy-dependent count has already been materialized on the item
        # by the semantic tree; avoid treating raw geometry as a new count.
        try:
            explicit = int(item.expected_slot_count or 0)
        except (TypeError, ValueError):
            explicit = 0
        target = explicit or target
        if not target or len(slots) <= target:
            return list(slots)
        pages = {slot.page_index for slot in slots}
        if len(pages) != 1:
            kept = sorted(
                slots,
                key=lambda slot: float((slot.audit or {}).get("candidate_confidence", .5)),
                reverse=True,
            )[:target]
            kept.sort(key=lambda slot: (slot.page_index, slot.expected_bbox[0], slot.expected_bbox[1]))
            return kept
        page_index = next(iter(pages))
        page = page_map.get(page_index)
        if page is None:
            return list(slots)[:target]
        candidates = [SlotCandidate(
            list(slot.expected_bbox), slot.slot_type,
            float((slot.audit or {}).get("candidate_confidence", .7)),
            slot.expected_text, "teacher_semantic_cardinality", 1,
        ) for slot in slots]
        groups = cls._partition_underlines(
            candidates, target, page, page.ocr,
            [region for region in cls._regions(item) if region.page_index == page_index],
        )
        result = []
        for index, group in enumerate(groups, 1):
            slot = Slot(
                index, group.slot_type, item.item_id, list(group.bbox), page_index,
                _expected_text(item, index-1) or group.expected_hint,
            )
            slot.audit = {
                "candidate_confidence": round(group.confidence, 4),
                "evidence": "teacher_semantic_cardinality_dp",
                "component_count": group.component_count,
                "target_slot_count": target,
                "slot_count_source": item.slot_count_source or source,
                "cardinality_confirmed": True,
            }
            result.append(slot)
        return result

    @classmethod
    def _valid_box(cls, box, item: ExamItem, page: Page,
                   question_corridor: Optional[Sequence[int]] = None):
        if len(box) < 4 or box[2] <= box[0] or box[3] <= box[1]:
            return False, "invalid yxyx bbox"
        height, width = float(page.height or 2338), float(page.width or 1654)
        if box[0] < 0 or box[1] < 0 or box[2] > height or box[3] > width:
            return False, "bbox outside page"
        kind = cls._effective_kind(item)
        box_height, box_width = float(box[2] - box[0]), float(box[3] - box[1])
        width_ratio = box_width / max(1.0, width)
        area_ratio = box_width * box_height / max(1.0, width * height)
        if kind == "choice":
            max_width_ratio, max_area_ratio = 0.35, 0.10
        elif kind == "fill":
            max_width_ratio, max_area_ratio = 0.58, 0.20
        elif cls._is_large_writing_item(item):
            max_width_ratio, max_area_ratio = 0.94, 0.68
        else:
            max_width_ratio, max_area_ratio = 0.82, 0.45
        if width_ratio > max_width_ratio:
            return False, f"slot width ratio {width_ratio:.3f} exceeds {max_width_ratio:.2f}"
        if area_ratio > max_area_ratio:
            return False, f"slot area ratio {area_ratio:.3f} exceeds {max_area_ratio:.2f}"
        if question_corridor and len(question_corridor) >= 4:
            center_y, center_x = (box[0]+box[2])/2.0, (box[1]+box[3])/2.0
            if not (question_corridor[0] <= center_y <= question_corridor[2]
                    and question_corridor[1] <= center_x <= question_corridor[3]):
                return False, "slot center is outside same-column question corridor"
        elif item.roi_patch and len(item.roi_patch.bbox) >= 4:
            roi = item.roi_patch.bbox
            intersection = max(0.0, min(box[2], roi[2]) - max(box[0], roi[0])) * max(
                0.0, min(box[3], roi[3]) - max(box[1], roi[1]))
            if intersection / max(1.0, box_width * box_height) < 0.90:
                return False, "slot is not contained in its question ROI"
        parent_regions = ([] if question_corridor else [
            region for region in cls._regions(item)
            if region.page_index == page.index and len(region.bbox) >= 4
        ])
        if parent_regions and item.stem_region is not None:
            stem = item.stem_region
            stem_center = (stem.bbox[1]+stem.bbox[3]) / 2.0
            nearby_regions = [
                region for region in parent_regions
                if abs((region.bbox[1]+region.bbox[3])/2.0-stem_center)
                <= .35 * height
            ]
            # Grossly mismatched OCR regions must not veto a candidate found
            # inside the independently grounded stem search corridor.
            parent_regions = nearby_regions
        if parent_regions:
            containment = max(
                max(0.0, min(box[2], region.bbox[3]) - max(box[0], region.bbox[1]))
                * max(0.0, min(box[3], region.bbox[2]) - max(box[1], region.bbox[0]))
                / max(1.0, box_width * box_height)
                for region in parent_regions
            )
            if containment < 0.70:
                return False, "slot is outside its parent item region"
        diagram_boxes = []
        for diagram in item.diagrams:
            if len(diagram.bbox) >= 4:
                diagram_boxes.append(xyxy_to_yxyx(diagram.bbox))
        if item.tri_target:
            diagram_boxes.extend(
                xyxy_to_yxyx(value) for value in item.tri_target.diagram_boxes
                if len(value) >= 4
            )
        for diagram in diagram_boxes:
            overlap = max(0.0, min(box[2], diagram[2])-max(box[0], diagram[0])) * max(
                0.0, min(box[3], diagram[3])-max(box[1], diagram[1])
            )
            if overlap / max(1.0, box_width*box_height) >= .30:
                return False, "slot overlaps diagram mask"
        return True, ""

    @staticmethod
    def _slot_match_cost(candidate: Slot, reference: Slot, page_map: Dict[int, Page]) -> float:
        """Language-free cost used for monotonic student/teacher assignment."""
        confidence = float((candidate.audit or {}).get("candidate_confidence", 0.5))
        if candidate.page_index != reference.page_index:
            return 2.0 + (1.0 - confidence)
        page = page_map.get(candidate.page_index)
        height = float((page.height if page else None) or 2338)
        width = float((page.width if page else None) or 1654)
        if len(candidate.expected_bbox or []) != 4 or len(reference.expected_bbox or []) != 4:
            return 1.0 - confidence
        cy = (candidate.expected_bbox[0] + candidate.expected_bbox[2]) / 2.0
        cx = (candidate.expected_bbox[1] + candidate.expected_bbox[3]) / 2.0
        ry = (reference.expected_bbox[0] + reference.expected_bbox[2]) / 2.0
        rx = (reference.expected_bbox[1] + reference.expected_bbox[3]) / 2.0
        geometry = abs(cy-ry) / max(1.0, height) + abs(cx-rx) / max(1.0, width)
        before = [b for b in (page.ocr if page else []) if len(b.bbox) == 4
                  and b.bbox[2] <= candidate.expected_bbox[1]+8
                  and abs((b.bbox[1]+b.bbox[3])/2-cy) <= max(18,candidate.expected_bbox[2]-candidate.expected_bbox[0])]
        observed = max(before,key=lambda b:b.bbox[2]).text[-48:] if before else ""
        semantic = (1-SequenceMatcher(None,_normalized(observed),_normalized(reference.anchor_before)).ratio()
                    if observed and reference.anchor_before else 0)
        return geometry + .4*semantic + 0.30 * (1.0 - confidence)

    @classmethod
    def _match_to_references(cls, candidates: Sequence[Slot], references: Sequence[Slot],
                             page_map: Dict[int, Page]) -> Dict[int, Slot]:
        """Minimum-cost monotonic assignment; keys are reference indexes."""
        ordered_candidates = ordered_slots(candidates)
        ordered_refs = sorted(references, key=lambda value: value.slot_idx)
        n, k = len(ordered_candidates), len(ordered_refs)
        if not n or not k:
            return {}
        infinity = float("inf")
        if n >= k:
            dp = [[infinity] * (k+1) for _ in range(n+1)]
            take = [[False] * (k+1) for _ in range(n+1)]
            for i in range(n+1):
                dp[i][0] = 0.0
            for i in range(1, n+1):
                for j in range(1, min(i, k)+1):
                    skipped = dp[i-1][j]
                    matched = dp[i-1][j-1] + cls._slot_match_cost(
                        ordered_candidates[i-1], ordered_refs[j-1], page_map
                    )
                    if matched <= skipped:
                        dp[i][j], take[i][j] = matched, True
                    else:
                        dp[i][j] = skipped
            result, i, j = {}, n, k
            while j:
                if take[i][j]:
                    result[j-1] = ordered_candidates[i-1]
                    i, j = i-1, j-1
                else:
                    i -= 1
            return result

        # Fewer candidates than expected positions: align every observed slot
        # and leave the unmatched reference indexes explicit for HITL.
        dp = [[infinity] * (k+1) for _ in range(n+1)]
        take = [[False] * (k+1) for _ in range(n+1)]
        for j in range(k+1):
            dp[0][j] = 0.0
        for i in range(1, n+1):
            for j in range(1, k+1):
                skipped_ref = dp[i][j-1]
                matched = dp[i-1][j-1] + cls._slot_match_cost(
                    ordered_candidates[i-1], ordered_refs[j-1], page_map
                )
                if matched <= skipped_ref:
                    dp[i][j], take[i][j] = matched, True
                else:
                    dp[i][j] = skipped_ref
        result, i, j = {}, n, k
        while i and j:
            if take[i][j]:
                result[j-1] = ordered_candidates[i-1]
                i, j = i-1, j-1
            else:
                j -= 1
        return result

    @classmethod
    def _lock_student_cardinality(cls, item: ExamItem, candidates: Sequence[Slot],
                                  inherited_slots: Sequence[Slot],
                                  page_map: Dict[int, Page], policy=None,
                                  question_corridor: Optional[Sequence[int]] = None) -> List[Slot]:
        target, source = cls._resolve_slot_count(item, inherited_slots, policy)
        if not target:
            return list(candidates)
        references = sorted(list(inherited_slots), key=lambda value: value.slot_idx)[:target]
        if len(references) < target:
            # No coordinate is invented here. Synthetic references are only
            # used for ordering; missing slots will be explicit HITL records.
            regions = cls._regions(item)
            stem = item.stem_region
            if stem is not None:
                page = page_map.get(stem.page_index)
                page_height = float((page.height if page else None) or 2338)
                stem_center = (stem.bbox[1]+stem.bbox[3]) / 2.0
                if not regions or all(
                    region.page_index != stem.page_index
                    or abs((region.bbox[1]+region.bbox[3])/2.0-stem_center)
                    > .35 * page_height
                    for region in regions
                ):
                    regions = [stem]
            for index in range(len(references), target):
                region = regions[min(index, len(regions)-1)] if regions else None
                if region:
                    left, top, right, bottom = [int(round(v)) for v in region.bbox[:4]]
                    box, page_index = [top, left, bottom, right], region.page_index
                else:
                    page_index = next(iter(page_map), 1)
                    page = page_map.get(page_index)
                    box = [0, 0, int((page.height if page else None) or 2338),
                           int((page.width if page else None) or 1654)]
                references.append(Slot(index+1, "unknown", item.item_id, box, page_index,
                                       _expected_text(item, index)))
        assignments = cls._match_to_references(candidates, references, page_map)
        locked: List[Slot] = []
        for index, reference in enumerate(references):
            slot = assignments.get(index)
            if slot is None:
                # A question/search corridor is not a slot coordinate. Missing
                # positions remain coordinate-free until student-local ink or
                # HITL supplies a real box.
                search_corridor = (list(question_corridor)
                                   if question_corridor and len(question_corridor) >= 4
                                   else None)
                regions = [region for region in cls._regions(item)
                           if region.page_index == reference.page_index]
                if search_corridor is None and regions:
                    search_corridor = xyxy_to_yxyx(regions[0].bbox)
                slot = Slot(index+1, reference.slot_type, item.item_id,
                            [], reference.page_index,
                            _expected_text(item, index) or reference.expected_text,
                            status="MISSING_SLOT", geometry_status="MISSING",
                            content_status="NOT_EVALUATED", review_status="NEED_REVIEW")
                slot.audit = {
                    "topology_source": "missing_placeholder",
                    "coordinate_role": "search_hint_only",
                    "target_slot_count": target,
                    "slot_count_source": source,
                    "reason": "no student-local candidate matched this expected slot",
                    "search_corridor_yxyx": search_corridor,
                    "coordinate_absent": True,
                }
            else:
                slot.slot_idx = index+1
                slot.expected_text = _expected_text(item, index) or reference.expected_text
                slot.audit.update({
                    "topology_source": "student_self",
                    "cardinality_locked": True,
                    "matched_reference_index": index+1,
                    "target_slot_count": target,
                    "slot_count_source": source,
                })
            locked.append(slot)
        return locked

    def _bracket_candidates(self, item: ExamItem, blocks: Sequence[OCRBlock], region: PageRegion):
        candidates: List[SlotCandidate] = []
        kind = self._effective_kind(item)
        answers = {_normalized(value) for value in _answer_parts(item)}
        left, top, right, bottom = [float(value) for value in region.bbox]
        visible_blocks = [block for block in blocks if len(block.bbox) >= 4
                          and block.bbox[2] >= left and block.bbox[0] <= right
                          and block.bbox[3] >= top and block.bbox[1] <= bottom]
        answer_like_count = 0
        for block in visible_blocks:
            text = block.text or ""
            for match in _BRACKET.finditer(text):
                inner = match.group(1).strip()
                if inner and re.search(r"[,，=+*/^<>]|[a-zA-Z].*\d|\d.*[a-zA-Z]", inner):
                    continue
                context = text[max(0, match.start()-2):match.start()] + text[match.end():match.end()+2]
                structural = bool(re.fullmatch(r"(?:\d{1,3}|[ivxlcdm]{1,6})", inner, re.I)) and match.start() <= 4
                formula = bool(re.search(r"[=+*/^<>]", inner + context)) or bool(
                    re.fullmatch(r"[A-Za-z0-9_]", text[max(0, match.start()-1):match.start()])
                )
                if not _SCORE.fullmatch(inner) and not structural and not formula:
                    answer_like_count += 1
        for block in visible_blocks:
            text = block.text or ""
            matches = list(_BRACKET.finditer(text))
            for match in matches:
                inner = match.group(1).strip()
                normalized_inner = _normalized(inner)
                if inner and re.search(r"[,，=+*/^<>]|[a-zA-Z].*\d|\d.*[a-zA-Z]", inner):
                    continue
                if _SCORE.fullmatch(inner):
                    continue
                structural = bool(re.fullmatch(r"(?:\d{1,3}|[ivxlcdm]{1,6})", inner, re.I)) and match.start() <= 4
                expected_match = bool(normalized_inner and normalized_inner in answers)
                if structural and not expected_match:
                    continue
                blank = not inner or not _normalized(inner.replace("_", ""))
                option_label = bool(re.fullmatch(r"[A-ZＡ-Ｚ]", inner, re.I))
                left_context = text[max(0, match.start()-2):match.start()]
                right_context = text[match.end():match.end()+2]
                math_expression = bool(re.search(r"[=+*/^<>]", inner))
                function_or_formula = bool(re.fullmatch(r"[A-Za-z0-9_]", left_context[-1:])) or bool(
                    re.search(r"[=+*/^<>]", left_context + right_context)
                )

                # Enhanced scoring for multi-blank fill items
                expected_count = len(_answer_parts(item))
                total_brackets = sum(len(list(_BRACKET.finditer(b.text or "")))
                                    for b in visible_blocks)

                if blank:
                    confidence, evidence = 0.98, "empty_bracket_cavity"
                elif expected_match and not function_or_formula:
                    confidence, evidence = 0.65, "short_response_candidate"
                elif kind == "choice" and option_label:
                    confidence, evidence = 0.90, "choice_label_in_brackets"
                elif kind == "fill" and len(inner) <= 16 and not math_expression and not function_or_formula:
                    # Boost confidence for fill blanks
                    confidence, evidence = 0.82, "short_content_in_declared_blank"
                elif kind == "fill" and expected_count > 1 and total_brackets >= expected_count:
                    # Multi-blank fill: accept multiple brackets
                    confidence, evidence = 0.78, "multi_blank_sequence"
                elif (answer_like_count >= 2 and len(inner) <= 12
                      and not math_expression and not function_or_formula):
                    confidence, evidence = 0.68, "repeated_short_bracket_answers"
                else:
                    confidence, evidence = 0.35, "ambiguous_parenthetical"
                if confidence < self.MIN_CANDIDATE_CONFIDENCE:
                    continue
                char_width = max(1.0, (block.bbox[2] - block.bbox[0]) / max(1, len(text)))
                xmin = block.bbox[0] + (match.start() + 0.65) * char_width
                xmax = block.bbox[0] + max(match.end() - 0.65, match.start() + 1.35) * char_width
                pad_y = max(4.0, (block.bbox[3] - block.bbox[1]) * 0.18)
                box = [int(max(0, block.bbox[1]-pad_y)), int(max(0, xmin)),
                       int(block.bbox[3]+pad_y), int(min(right, xmax))]
                candidates.append(SlotCandidate(box, "bracket_blank", confidence,
                                                inner or None, evidence))
        return candidates

    def _bracket_slots(self, item: ExamItem, blocks: Sequence[OCRBlock], region: PageRegion):
        return [(candidate.bbox, candidate.expected_hint)
                for candidate in self._bracket_candidates(item, blocks, region)]

    @classmethod
    def _split_choice_cavity(cls, item: ExamItem, page: Page,
                             region: PageRegion) -> List[SlotCandidate]:
        """Recover a choice cavity when OCR splits ``(C)`` across blocks."""
        left, top, right, bottom = [float(value) for value in region.bbox]
        blocks = [block for block in page.ocr if len(block.bbox) >= 4
                  and left <= (block.bbox[0]+block.bbox[2])/2 <= right
                  and top <= (block.bbox[1]+block.bbox[3])/2 <= bottom]
        openings = [block for block in blocks
                    if re.search(r"[（(]\s*$", str(block.text or ""))]
        if not openings:
            return []
        stem_y = ((item.stem_region.bbox[1]+item.stem_region.bbox[3])/2
                  if item.stem_region else top)
        opening = min(openings, key=lambda block: abs(
            (block.bbox[1]+block.bbox[3])/2-stem_y
        ))
        y1, y2 = int(opening.bbox[1]), int(opening.bbox[3])
        x1 = int(opening.bbox[2])
        same_line = sorted([
            block for block in blocks if block.bbox[0] >= x1
            and abs((block.bbox[1]+block.bbox[3])/2-(y1+y2)/2)
            <= max(18, (y2-y1)*.8)
        ], key=lambda block: block.bbox[0])
        answer = next((block for block in same_line
                       if re.fullmatch(r"\s*[A-HＡ-Ｈ]\s*[）)]?\s*",
                                       str(block.text or ""))), None)
        closing = next((block for block in same_line
                        if re.match(r"^\s*[）)]", str(block.text or ""))), None)
        x2 = int(closing.bbox[0]) if closing else (
            int(answer.bbox[2])+12 if answer else x1+max(45, int((page.width or 1654)*.04))
        )
        hint = re.sub(r"[^A-HＡ-Ｈ]", "", str(answer.text or ""), flags=re.I) if answer else None
        return [SlotCandidate(
            [max(0, y1-8), max(0, x1),
             min(int(page.height or 2338), y2+8), min(int(page.width or 1654), x2)],
            "bracket_blank", .99, hint or None, "split_ocr_choice_cavity",
        )]

    @staticmethod
    def _merge_overlapping_slots(slots, iou_threshold=0.3):
        """Merge overlapping slot boxes."""
        if not slots:
            return []

        slots = sorted(slots, key=lambda b: (b[0], b[1]))
        merged = [slots[0]]

        for current in slots[1:]:
            last = merged[-1]

            # Calculate IOU
            y1_i = max(last[0], current[0])
            x1_i = max(last[1], current[1])
            y2_i = min(last[2], current[2])
            x2_i = min(last[3], current[3])

            inter_area = max(0, y2_i - y1_i) * max(0, x2_i - x1_i)
            last_area = (last[2] - last[0]) * (last[3] - last[1])
            current_area = (current[2] - current[0]) * (current[3] - current[1])
            union_area = last_area + current_area - inter_area

            iou = inter_area / max(1, union_area)

            if iou > iou_threshold:
                # Merge: take bounding box
                merged[-1] = [
                    min(last[0], current[0]),
                    min(last[1], current[1]),
                    max(last[2], current[2]),
                    max(last[3], current[3])
                ]
            else:
                merged.append(current)

        return merged

    @staticmethod
    def _underline_slots(image: Any, region: PageRegion):
        import cv2
        left, top, right, bottom = [int(round(v)) for v in region.bbox]
        crop = image[max(0, top):max(top, bottom), max(0, left):max(left, right)]
        if crop is None or crop.size == 0:
            return []
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Enhanced contrast for thin line detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Multi-scale detection for different line widths
        all_slots = []
        kernel_sizes = [
            int(round(crop.shape[1] * 0.03)),  # Short lines
            int(round(crop.shape[1] * 0.05)),  # Medium lines
            int(round(crop.shape[1] * 0.08)),  # Long lines
        ]

        for kernel_width in kernel_sizes:
            kernel_width = max(10, min(80, kernel_width))
            lines = cv2.morphologyEx(
                binary, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
            )
            contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            minimum_width = kernel_width * 1.2
            vertical_padding = max(20, int(round(image.shape[0] * 0.025)))

            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                max_height = max(8, int(round(image.shape[0] * 0.008)))

                if width >= minimum_width and height <= max_height and width <= crop.shape[1] * 0.95:
                    slot_box = [
                        max(0, top+y-vertical_padding),
                        max(0, left+x),
                        min(image.shape[0], top+y+height+vertical_padding),
                        min(image.shape[1], left+x+width)
                    ]
                    all_slots.append(slot_box)

        # Merge overlapping slots
        merged = MultiSlotTopologyEngine._merge_overlapping_slots(all_slots, iou_threshold=0.3)
        return sorted(merged, key=lambda box: (box[0], box[1]))

    @staticmethod
    def _grid_slots(image: Any, region: PageRegion):
        import cv2
        left, top, right, bottom = [int(round(v)) for v in region.bbox]
        crop = image[max(0, top):max(top, bottom), max(0, left):max(left, right)]
        if crop is None or crop.size == 0:
            return []
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)[1]
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        slots = []
        scale = max(1.0, min(image.shape[:2]) / 1000.0)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            rectangularity = cv2.contourArea(contour) / max(1, width * height)
            if (18*scale <= width <= 180*scale and 18*scale <= height <= 180*scale
                    and 0.65 <= width/max(1, height) <= 1.35
                    and len(polygon) == 4 and rectangularity >= 0.55):
                slots.append([top+y, left+x, top+y+height, left+x+width])
        return sorted(slots, key=lambda box: (box[0], box[1]))

    @classmethod
    def _regions(cls, item: ExamItem):
        source = (item.student_regions or item.answer_regions or
                  item.blank_regions or item.option_regions or item.writing_regions)
        result, seen = [], set()
        for region in source:
            key = (region.page_index, tuple(region.bbox[:4]))
            if key not in seen:
                seen.add(key); result.append(region)
        return result

    @classmethod
    def _student_bands(cls, package: ExamPackage,
                       pages: Sequence[Page]) -> Dict[str, Tuple[int, List[int]]]:
        """Re-ground every item on the student page using the student's own OCR.

        The teacher owns question identity (text, answer count); the student
        page owns every coordinate.  Returns ``item_id -> (page_index, yxyx
        band)`` where a band runs from the item's own printed anchor down to
        the next anchor in the same physical column.  Anchors are assigned
        greedily and uniquely so two items cannot claim one printed line.

        Multi-level fallback for robustness:
        1. Text-anchor scoring (primary)
        2. Geometric estimation from neighboring anchors (secondary)
        3. Vertical page partitioning (tertiary, guaranteed)
        """

        def _detect_column_bounds(entries, width):
            """Detect actual content column boundaries instead of fixed proportions."""
            if not entries:
                return 45, width - 45

            # Collect left and right boundaries from all anchor boxes
            lefts = [box[0] for _, box in entries]
            rights = [box[2] for _, box in entries]

            if len(lefts) < 2:
                # Fallback for single item: use the box itself with padding
                return max(20, int(lefts[0] - width * 0.02)), min(width - 20, int(rights[0] + width * 0.02))

            # Use 5th and 95th percentile as column boundaries to avoid outliers
            lefts_sorted = sorted(lefts)
            rights_sorted = sorted(rights)
            left_bound = lefts_sorted[max(0, len(lefts_sorted) // 20)]
            right_bound = rights_sorted[min(len(rights_sorted) - 1, len(rights_sorted) * 19 // 20)]

            # Add reasonable padding
            return max(20, int(left_bound - width * 0.02)), min(width - 20, int(right_bound + width * 0.02))

        page_map = {page.index: page for page in pages}
        logical = [item for section in package.sections
                   for question in section.questions for item in question.items]

        # Level 1: Text-based anchor scoring (existing logic)
        scored = []
        for logical_index, item in enumerate(logical):
            for page in pages:
                for block_index, block in enumerate(page.ocr):
                    if len(block.bbox) < 4:
                        continue
                    score = _student_anchor_score(item, block)
                    # P2 修复：进一步降低阈值从 0.35 到 0.28
                    # 对于题号匹配的情况，0.70 的基础分已经足够可靠
                    # 对于纯文本匹配，0.30 的分数也能覆盖合理的相似度
                    # 这样可以提高召回率，减少 Q5、Q9 等题目的匹配失败
                    if score >= 0.28:
                        scored.append((score, logical_index, page.index, block_index))
        scored.sort(key=lambda value: (-value[0], value[1], value[2], value[3]))

        # Greedy assignment
        anchors: Dict[int, Tuple[int, List[float]]] = {}
        used_items, used_blocks = set(), set()
        for _score, logical_index, page_index, block_index in scored:
            if logical_index in used_items or (page_index, block_index) in used_blocks:
                continue
            used_items.add(logical_index)
            used_blocks.add((page_index, block_index))
            block = page_map[page_index].ocr[block_index]
            anchors[logical_index] = (page_index, [float(v) for v in block.bbox[:4]])

        by_page: Dict[int, List[Tuple[str, List[float]]]] = {}
        for logical_index, (page_index, box) in anchors.items():
            by_page.setdefault(page_index, []).append((logical[logical_index].item_id, box))
        result: Dict[str, Tuple[int, List[int]]] = {}
        for page_index, entries in by_page.items():
            page = page_map.get(page_index)
            if page is None:
                continue
            height, width = int(page.height or 2338), int(page.width or 1654)
            narrow = [pair for pair in entries if pair[1][2]-pair[1][0] <= width*.62]
            left = [pair for pair in narrow if (pair[1][0]+pair[1][2])/2 < width*.46]
            right = [pair for pair in narrow if (pair[1][0]+pair[1][2])/2 > width*.54]
            two_column = len(left) >= 2 and len(right) >= 2
            groups: Dict[str, List[Tuple[str, List[float]]]] = {}
            for item_id, box in entries:
                center = (box[0]+box[2])/2
                column = ("left" if center < width*.5 else "right") if two_column else "full"
                groups.setdefault(column, []).append((item_id, box))
            for column, values in groups.items():
                values.sort(key=lambda pair: (pair[1][1], pair[1][0]))
                if column == "left":
                    x1, x2 = _detect_column_bounds(values, width)
                    x2 = min(x2, int(width * .58))  # Cap at left column boundary
                elif column == "right":
                    x1, x2 = _detect_column_bounds(values, width)
                    x1 = max(x1, int(width * .42))  # Floor at right column boundary
                else:
                    x1, x2 = _detect_column_bounds(values, width)
                for index, (item_id, box) in enumerate(values):
                    top = int(max(0, box[1]-6))
                    bottom = (int(values[index+1][1][1])-8
                              if index+1 < len(values) else height-20)
                    if bottom > top:
                        result[item_id] = [top, x1, min(height, bottom), x2]
        return {item_id: (anchors[index][0], band)
                for index, item in enumerate(logical)
                for item_id, band in [(item.item_id, result.get(item.item_id))]
                if index in anchors and band}

    def _autonomous_ink_candidates(self, item: ExamItem, image: Any,
                                   reference_image: Any, page: Page,
                                   band: Sequence[int], target_count: Optional[int],
                                   reference_kind: str = "teacher",
                                   reference_confidence: Optional[float] = None
                                   ) -> List[SlotCandidate]:
        """Locate student answer ink inside a student-grounded band.

        No teacher coordinate reaches this path: the band comes from the
        student's own printed anchor and the ink comes from the student's own
        pixels.  Printed body text recognised inside the band is masked out so
        a stem or option line can never be promoted to an answer box.
        """
        import cv2
        from .ink_separation import NoBlankInkSeparator
        if image is None or len(band) < 4:
            return []
        height, width = image.shape[:2]
        by1, bx1 = max(0, int(band[0])), max(0, int(band[1]))
        by2, bx2 = min(height, int(band[2])), min(width, int(band[3]))
        if by2-by1 < 12 or bx2-bx1 < 24:
            return []
        crop = image[by1:by2, bx1:bx2]
        if crop is None or crop.size == 0:
            return []
        reference_crop = None
        if (reference_image is not None
                and reference_image.shape[:2] == image.shape[:2]):
            reference_crop = reference_image[by1:by2, bx1:bx2]
        separation = NoBlankInkSeparator.separate(
            crop, reference_crop, self.policy, reference_kind, reference_confidence,
        )
        mask = separation.get("handwriting_mask")
        if mask is None:
            return []
        mask = mask.copy()
        # Printed stem/option text inside the band is never an answer.  Without
        # a cohort template the single-page classifier can retain printed
        # glyphs, so drop every block whose text belongs to the item's own body.
        body = _normalized(item.question_text)
        # Build 4-gram set for fuzzy matching
        body_words = set(body[i:i+4] for i in range(len(body)-3)) if len(body) >= 4 else set()

        for block in page.ocr:
            if len(block.bbox) < 4:
                continue
            text = _normalized(block.text)
            if len(text) < 3:
                continue

            # Two-tier matching: exact substring OR 4-gram overlap
            exact_match = len(text) >= 4 and body and text in body

            # Fuzzy match via 4-gram overlap
            fuzzy_match = False
            if not exact_match and body_words:
                text_words = set(text[i:i+4] for i in range(len(text)-3)) if len(text) >= 4 else {text}
                overlap_ratio = len(body_words & text_words) / max(1, len(text_words))
                # Overlap > 40% indicates this is likely question text
                fuzzy_match = overlap_ratio > 0.40

            if not exact_match and not fuzzy_match:
                continue

            left, top, right, bottom = [int(round(value)) for value in block.bbox[:4]]
            y1, y2 = max(0, top-by1-2), min(mask.shape[0], bottom-by1+2)
            x1, x2 = max(0, left-bx1-2), min(mask.shape[1], right-bx1+2)
            if y2 > y1 and x2 > x1:
                mask[y1:y2, x1:x2] = 0
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1,
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        crop_h, crop_w = mask.shape[:2]
        components = []
        for label in range(1, count):
            cx, cy, cw, ch, area = [int(value) for value in stats[label]]
            if area < 12 or cw < 3 or ch < 5:
                continue
            if cw >= .55 * crop_w and ch <= 4:
                continue          # surviving printed rule, not a glyph
            if cw > .92 * crop_w and ch > .92 * crop_h:
                continue
            components.append((cx, cy, cw, ch, area))
        if not components:
            return []
        components.sort(key=lambda value: (value[1], value[0]))
        lines: List[List[Tuple[int, int, int, int, int]]] = []
        for component in components:
            for line in lines:
                top = min(value[1] for value in line)
                bottom = max(value[1]+value[3] for value in line)
                overlap = min(bottom, component[1]+component[3]) - max(top, component[1])
                if overlap > .40 * max(1, min(bottom-top, component[3])):
                    line.append(component)
                    break
            else:
                lines.append([component])
        padding = 4
        primitives = [SlotCandidate(
            [max(0, by1+min(value[1] for value in line)-padding),
             max(0, bx1+min(value[0] for value in line)-padding),
             min(height, by1+max(value[1]+value[3] for value in line)+padding),
             min(width, bx1+max(value[0]+value[2] for value in line)+padding)],
            "student_ink_answer", .0, None, "student_native_ink_band", len(line),
        ) for line in lines]
        primitives.sort(key=lambda value: (value.bbox[0], value.bbox[1]))
        if target_count and len(primitives) > target_count:
            primitives = self._partition_underlines(
                primitives, target_count, page, page.ocr, [],
            )
        confidence = .72 if reference_crop is not None else .66
        return [SlotCandidate(
            candidate.bbox, "student_ink_answer", confidence,
            _expected_text(item, index), "student_native_ink_band",
            candidate.component_count,
        ) for index, candidate in enumerate(primitives)]

    def _discover_free_response(self, item, page, image, reference_image, regions,
                                reference_kind, reference_confidence):
        from .ink_separation import NoBlankInkSeparator
        import cv2
        import numpy as np
        fragments = []
        if item.quality.get("geometry_status") == "UNRESOLVED_SUBITEM_BOUNDARY":
            return []
        for region in regions:
            x1,y1,x2,y2 = [int(v) for v in region.bbox]
            x1,y1 = max(0,x1),max(0,y1)
            x2,y2 = min(image.shape[1],x2),min(image.shape[0],y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2,x1:x2]
            ref = reference_image[y1:y2,x1:x2] if reference_image is not None else None
            separation = NoBlankInkSeparator.separate(crop,ref,self.policy,reference_kind,reference_confidence)
            mask = separation["handwriting_mask"]
            # Exclude known prompt lines; do not treat their parentheses as answers.
            for block in page.ocr:
                if len(block.bbox) != 4 or not block.text.strip():
                    continue
                if len(block.text.strip()) >= 10 and block.text.strip() in item.question_text:
                    bx1,by1,bx2,by2 = [int(v) for v in block.bbox]
                    mask[max(0,by1-y1):max(0,min(y2,by2)-y1),
                         max(0,bx1-x1):max(0,min(x2,bx2)-x1)] = 0
            joined = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,21),np.uint8))
            count,_,stats,_ = cv2.connectedComponentsWithStats(joined,8)
            for x,y,w,h,area in stats[1:]:
                if area >= 35 and h >= 8 and w >= 8:
                    fragments.append([int(y1+y),int(x1+x),int(y1+y+h),int(x1+x+w)])
        if not fragments:
            return []
        box=[min(b[0] for b in fragments),min(b[1] for b in fragments),
             max(b[2] for b in fragments),max(b[3] for b in fragments)]
        slot=Slot(1,"free_response",item.item_id,box,page.index)
        slot.audit={"evidence":"free_response_components", "candidate_confidence":.65,
                    "physical_fragments_yxyx":fragments, "coordinate_authority":"student_page_only"}
        item.expected_slot_count = 1
        item.slot_count_source = "logical_free_response"
        return [slot]

    def discover_item(self, item: ExamItem, page: Page, image: Any,
                      inherited_slots: Sequence[Slot] = (), reference_image: Any = None,
                      reference_kind: str = "teacher",
                      reference_confidence: Optional[float] = None,
                      question_corridor: Optional[Sequence[int]] = None,
                      student_band: Optional[Sequence[int]] = None) -> List[Slot]:
        if question_corridor and len(question_corridor) >= 4:
            y1, x1, y2, x2 = question_corridor
            regions = [PageRegion(page.index, page.path, [x1, y1, x2, y2], 1.0,
                                  "same_column_question_corridor")]
        else:
            regions = [region for region in self._regions(item)
                       if region.page_index == page.index]
        kind = self._effective_kind(item)
        if kind == "large_writing":
            # A free response has one logical answer, not a slot per formula
            # parenthesis, underline, or line of working.
            return self._discover_free_response(item, page, image, reference_image, regions,
                                                reference_kind, reference_confidence)
        target_count, count_source = self._resolve_slot_count(
            item, inherited_slots, self.policy
        )
        has_option_layout = len(_OPTION_LABEL.findall(str(item.question_text or ""))) >= 2
        candidates: List[SlotCandidate] = []
        for region in regions:
            candidates.extend(self._bracket_candidates(item, page.ocr, region))
            if kind == "choice":
                candidates.extend(self._split_choice_cavity(item, page, region))
                # Rulers, charts and option illustrations contain many
                # rectangles. They are never response boxes when a choice
                # response cavity is expected.
                continue
            grid_boxes = self._grid_slots(image, region)
            rectangular_topology = kind == "grid" or len(grid_boxes) >= 3
            if rectangular_topology:
                candidates.extend(SlotCandidate(
                    box, "tianzige", 0.84 if kind == "grid" else 0.70, None,
                    "rectangular_cell_cluster",
                ) for box in grid_boxes)
            elif kind == "choice" and grid_boxes:
                rectangular_topology = True
                candidates.extend(SlotCandidate(
                    box, "choice_box", 0.68, None, "rectangular_choice_mark",
                ) for box in grid_boxes)
            # Printed rules inside option text are not answer lines. A stale
            # declared "choice" label may still be overridden when the item
            # has no physical A/B/C/D option layout (e.g. multiple long blanks).
            if not rectangular_topology and not (kind == "choice" and has_option_layout):
                for box in self._underline_slots(image, region):
                    ratio = (box[3] - box[1]) / max(1.0, float(page.width or image.shape[1]))
                    candidates.append(SlotCandidate(
                        box, "underline_blank", min(0.92, 0.70 + ratio), None,
                        "horizontal_answer_rule",
                    ))

        # Remove cross-column rules and implausibly broad primitives before
        # cardinality reconciliation.  Otherwise a rejected primitive could be
        # fused into a valid answer group and consume one of the target groups.
        candidates = [candidate for candidate in candidates
                      if candidate.confidence >= self.MIN_CANDIDATE_CONFIDENCE
                      and self._valid_box(candidate.bbox, item, page,
                                         question_corridor)[0]]
        if kind == "choice":
            response_cavities = [candidate for candidate in candidates
                                 if candidate.slot_type == "bracket_blank"]
            if response_cavities:
                candidates = response_cavities

        # A physical writing line is not automatically a semantic answer
        # point. Subjective responses commonly span several ruled lines but
        # still represent one answer. Preserve explicit bracket cavities; in
        # their absence merge all detected writing lines into one area unless
        # the item has independent evidence of multiple answers.
        explicit_blanks = len(_BLANK_TEXT.findall(str(item.question_text or "")))
        allows_multiple = (kind in {"fill", "grid"} or len(_answer_parts(item)) > 1
                           or explicit_blanks > 1)
        if candidates and kind != "choice" and not allows_multiple:
            bracket_candidates = [candidate for candidate in candidates
                                  if candidate.slot_type == "bracket_blank"]
            underline_candidates = [candidate for candidate in candidates
                                    if candidate.slot_type == "underline_blank"]
            if bracket_candidates:
                candidates = bracket_candidates
            elif underline_candidates:
                merged_box = [
                    min(candidate.bbox[0] for candidate in underline_candidates),
                    min(candidate.bbox[1] for candidate in underline_candidates),
                    max(candidate.bbox[2] for candidate in underline_candidates),
                    max(candidate.bbox[3] for candidate in underline_candidates),
                ]
                candidates = [SlotCandidate(
                    merged_box, "writing_area",
                    max(candidate.confidence for candidate in underline_candidates),
                    _expected_text(item, 0), "merged_multiline_writing",
                    len(underline_candidates),
                )]

        # Reconcile physical rules with semantic answer-point cardinality.  The
        # DP groups only student-local candidates and therefore never imports a
        # teacher bbox. Brackets/grids remain stronger explicit cavities.
        if target_count and candidates:
            underlines = [candidate for candidate in candidates
                          if candidate.slot_type == "underline_blank"]
            fixed = [candidate for candidate in candidates
                     if candidate.slot_type != "underline_blank"]
            underline_groups = target_count - len(fixed)
            if underlines and underline_groups <= 0:
                candidates = fixed
            elif underlines and underline_groups < len(underlines):
                candidates = fixed + self._partition_underlines(
                    underlines, underline_groups, page, page.ocr, regions
                )

        if not candidates and reference_image is not None:
            candidates.extend(self._residual_candidates(
                item, image, reference_image, inherited_slots, page,
                reference_kind, reference_confidence,
            ))

        # NEW: 多空填空槽位拓扑增强
        # 当现有检测器找到的候选框数量少于预期时，使用拓扑增强策略
        # 注意：_partition_underlines 已经在1331行被调用，这里只是确保它被正确触发
        # 实际的拓扑增强已经内置在现有逻辑中，无需额外调用

        # Autonomous student-native discovery.  The printed detectors above and
        # the residual path both search inside teacher-inherited regions, so a
        # student who answered outside them yields nothing.  This band is
        # grounded on the student's own printed anchor, so ink found here needs
        # no teacher coordinate at all.  It only supplements: explicit brackets,
        # grids and rules already found on this page always win.
        if student_band and len(student_band) >= 4 and (
                not candidates or (target_count and len(candidates) < target_count)):
            candidates.extend(self._autonomous_ink_candidates(
                item, image, reference_image, page, student_band, target_count,
                reference_kind, reference_confidence,
            ))

        accepted: List[SlotCandidate] = []
        for candidate in sorted(candidates, key=lambda value: value.confidence, reverse=True):
            # A student-native candidate is bounded by its own student band, not
            # by the teacher-inherited corridor that would veto it.
            corridor = (student_band if candidate.evidence == "student_native_ink_band"
                        else question_corridor)
            valid, _ = self._valid_box(candidate.bbox, item, page, corridor)
            if (candidate.confidence >= self.MIN_CANDIDATE_CONFIDENCE and valid
                    and not any(_iou(candidate.bbox, old.bbox) > 0.72 for old in accepted)):
                accepted.append(candidate)
        accepted.sort(key=lambda value: (value.bbox[0], value.bbox[1]))
        slots = []
        for index, candidate in enumerate(accepted):
            slot = Slot(index+1, candidate.slot_type, item.item_id, candidate.bbox, page.index,
                        _expected_text(item, index) or candidate.expected_hint)
            slot.audit = {"candidate_confidence": round(candidate.confidence, 4),
                          "evidence": candidate.evidence,
                          "component_count": candidate.component_count,
                          "detector": ("student_native_ink_band_v1"
                                       if candidate.evidence == "student_native_ink_band"
                                       else "local_residual_geometric_dp_v4"),
                          "coordinate_authority": (
                              "student_page_only"
                              if candidate.evidence == "student_native_ink_band"
                              else "student_page_within_inherited_region"),
                          "target_slot_count": target_count,
                          "slot_count_source": count_source,
                          "knowledge_rules": (
                              self.policy.rule_ids("slot") if self.policy else []
                          )}
            slots.append(slot)
        return slots

    def enrich_package(self, package: ExamPackage, pages: Sequence[Page],
                       reference_pages: Sequence[Page] = (),
                       reference_kind: str = "teacher",
                       reference_confidence: Optional[float] = None):
        import cv2
        page_map = {page.index: page for page in pages}
        question_corridors = QuestionLayoutService.corridors(package, page_map)
        reference_page_map = {page.index: page for page in reference_pages}
        cache: Dict[int, Any] = {}
        reference_cache: Dict[int, Any] = {}
        items = slots = rejected = student_self_discovered = 0
        residual_discovered = teacher_cardinality_merged = autonomous_discovered = 0
        student_mode = str(package.document_type or "").lower() == "student"
        student_bands: Dict[str, Tuple[int, List[int]]] = {}
        if student_mode:
            package.warnings.append(
                "学生卷 Slot 采用自定位策略：继承教师题目层级与答案；教师/共识坐标仅作"
                "局部搜索提示，最终框由学生卷横线、括号、方格或新增墨迹证据生成"
            )
            # Re-ground every item on the student's own printed anchors. When the
            # inherited regions miss the student's answer entirely, this band is
            # the only coordinate source that owes nothing to the teacher page.
            student_bands = self._student_bands(package, pages)
            package.warnings.append(
                "学生卷自主定位：{}/{} 个小题已在学生卷自身版式上重新锚定搜索带".format(
                    len(student_bands),
                    sum(len(question.items) for section in package.sections
                        for question in section.questions),
                )
            )
        occupied: Dict[int, List[Any]] = {}
        for section in package.sections:
            for question in section.questions:
                for item in question.items:
                    question_corridor = question_corridors.get(item.item_id)
                    if question_corridor:
                        item.quality["question_corridor_yxyx"] = list(question_corridor)
                    indexes = sorted({region.page_index for region in self._regions(item)})
                    if not indexes and item.stem_region is not None:
                        indexes = [item.stem_region.page_index]
                    if item.is_cross_page or len(indexes) > 1:
                        item.cross_page_status = "NEEDS_NEXT_PAGE_MERGE"
                    rejection_reasons = []
                    inherited_slots = list(item.slots)
                    if student_mode or not item.slots:
                        if student_mode:
                            item.slots = []
                        found = []
                        for index in indexes:
                            page = page_map.get(index)
                            if page:
                                cache.setdefault(index, cv2.imread(page.path))
                                if cache[index] is not None:
                                    reference_page = reference_page_map.get(index)
                                    if reference_page is not None:
                                        reference_cache.setdefault(
                                            index, cv2.imread(reference_page.path)
                                        )
                                    band_page, band = student_bands.get(
                                        item.item_id, (None, None)
                                    )
                                    found.extend(self.discover_item(
                                        item, page, cache[index], inherited_slots,
                                        reference_cache.get(index), reference_kind,
                                        reference_confidence, question_corridor,
                                        band if band_page == index else None,
                                    ))
                        if student_mode and not self._is_large_writing_item(item):
                            found = self._lock_student_cardinality(
                                item, found, inherited_slots, page_map, self.policy,
                                question_corridor,
                            )
                        for index, slot in enumerate(found, 1):
                            slot.slot_idx = index
                            observed_hint = slot.expected_text
                            if student_mode and slot.audit.get("topology_source") == "missing_placeholder":
                                slot.audit["expected_source"] = (
                                    "teacher_truth" if slot.expected_text else "missing"
                                )
                            elif student_mode:
                                inherited_truth = (inherited_slots[index-1].expected_text
                                                   if index <= len(inherited_slots) else None)
                                slot.expected_text = _expected_text(item, index-1) or inherited_truth
                                slot.audit.update({
                                    "topology_source": "student_self",
                                    "expected_source": ("teacher_truth" if slot.expected_text else "missing"),
                                })
                                if observed_hint and not slot.expected_text:
                                    slot.audit["student_observed_hint"] = observed_hint
                            else:
                                slot.audit.update({
                                    "topology_source": "teacher",
                                    "expected_source": ("teacher_truth_or_ocr" if slot.expected_text else "missing"),
                                })
                        item.slots = ordered_slots(found)
                        for local_index, current_slot in enumerate(item.slots, 1):
                            current_slot.slot_idx = local_index
                        if not item.expected_slot_count and found:
                            resolved_count, resolved_source = self._resolve_slot_count(
                                item, inherited_slots, self.policy
                            )
                            if resolved_count:
                                item.expected_slot_count = resolved_count
                                item.slot_count_source = resolved_source
                        if student_mode:
                            student_self_discovered += sum(
                                slot.audit.get("topology_source") == "student_self" for slot in found
                            )
                            residual_discovered += sum(
                                "residual_local_registration" in str(
                                    slot.audit.get("evidence", "")
                                ) for slot in found
                            )
                            autonomous_discovered += sum(
                                slot.audit.get("evidence") == "student_native_ink_band"
                                for slot in found
                            )
                        if not found and self._effective_kind(item) in {"choice", "fill", "grid"}:
                            package.warnings.append(
                                f"{item.item_id} 未探测到可信物理槽位，禁止粗题框回退，已进入 HITL"
                            )
                    if not student_mode and item.slots:
                        before_cardinality = len(item.slots)
                        item.slots = self._reconcile_teacher_cardinality(
                            item, item.slots, page_map
                        )
                        teacher_cardinality_merged += max(
                            0, before_cardinality-len(item.slots)
                        )
                    item_band_page, item_band = student_bands.get(
                        item.item_id, (None, None)
                    )
                    accepted_slots = []
                    for slot in item.slots:
                        page = page_map.get(slot.page_index)
                        if page is None:
                            rejection_reasons.append(f"slot {slot.slot_idx}: missing page")
                            continue
                        placeholder = slot.audit.get("topology_source") == "missing_placeholder"
                        # A student-native box is bounded by the student band it
                        # was found in. Re-checking it against the teacher-derived
                        # corridor would discard exactly the answers this detector
                        # exists to recover.
                        slot_corridor = question_corridor
                        if (slot.audit.get("coordinate_authority") == "student_page_only"
                                and item_band and item_band_page == slot.page_index):
                            slot_corridor = item_band
                        valid, reason = self._valid_box(
                            slot.expected_bbox, item, page, slot_corridor
                        )
                        duplicate = None if placeholder else next((
                            owner for owner, old in occupied.get(slot.page_index, [])
                            if owner != item.item_id and _iou(slot.expected_bbox, old) > 0.85
                        ), None)
                        if duplicate:
                            valid, reason = False, f"duplicates item {duplicate}"
                        if not valid and not placeholder:
                            rejected += 1
                            rejection_reasons.append(f"slot {slot.slot_idx}: {reason}")
                            continue
                        accepted_slots.append(slot)
                        if not placeholder:
                            occupied.setdefault(slot.page_index, []).append((item.item_id, slot.expected_bbox))
                    item.slots = accepted_slots
                    from .semantic_slots import bind_semantic_slots
                    bind_semantic_slots(item, page_map)

                    # P1 修复：将槽位检测找到的答题区域回写到 student_regions
                    # 这样 student_answer OCR 就会从正确的区域提取
                    if student_mode and accepted_slots:
                        # 按页面分组槽位
                        slots_by_page: Dict[int, List[List[float]]] = {}
                        for slot in accepted_slots:
                            if slot.expected_bbox and len(slot.expected_bbox) >= 4:
                                slots_by_page.setdefault(slot.page_index, []).append(
                                    [float(v) for v in slot.expected_bbox]
                                )

                        # 为每个页面创建一个包含所有槽位的区域
                        new_student_regions = []
                        for page_idx in sorted(slots_by_page.keys()):
                            bboxes = slots_by_page[page_idx]
                            if bboxes:
                                # 计算所有槽位的并集区域
                                y1 = min(bbox[0] for bbox in bboxes)
                                x1 = min(bbox[1] for bbox in bboxes)
                                y2 = max(bbox[2] for bbox in bboxes)
                                x2 = max(bbox[3] for bbox in bboxes)

                                page = page_map.get(page_idx)
                                if page:
                                    new_student_regions.append(
                                        PageRegion(
                                            page_idx, page.path,
                                            [x1, y1, x2, y2],
                                            0.95,  # 高置信度，因为来自槽位检测
                                            "slot_derived_answer_region"
                                        )
                                    )

                        # 回写到 student_regions
                        if new_student_regions:
                            item.student_regions = new_student_regions
                            item.quality["student_regions_source"] = "slot_derived"

                    if rejection_reasons:
                        package.warnings.append(
                            f"{item.item_id} 槽位预检未通过，已移除异常框并进入 HITL："
                            + "; ".join(rejection_reasons)
                        )
                    items += 1
                    slots += len(item.slots)
        return {"items": items, "slots": slots, "rejected_slots": rejected,
                "student_self_discovered": student_self_discovered,
                "residual_discovered": residual_discovered,
                "autonomous_discovered": autonomous_discovered,
                "student_bands_grounded": len(student_bands),
                "teacher_cardinality_merged": teacher_cardinality_merged,
                "student_slot_policy": "self" if student_mode else "teacher",
                "cardinality_policy": "teacher_vlm_hard_lock",
                "question_corridor_policy": "shared_same_column_reading_order_v1",
                "whole_question_fallback": False,
                "detector": "local_residual_geometric_dp_v5"}


__all__ = ["MultiSlotTopologyEngine", "SlotCandidate", "infer_item_type"]
