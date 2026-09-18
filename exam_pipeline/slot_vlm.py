"""Vision review and bounded automatic repair of final slot coordinates.

Geometry remains the primary locator. A VLM proposal is committed only after
local hard gates and a second visual verification. Original coordinates are
always retained in the audit trail.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .contracts import ExamPackage, Page
from .io_utils import atomic_write_json
from .layout import QuestionLayoutService
from .roi import xyxy_to_yxyx

LOGGER = logging.getLogger("exam_pipeline.slot_vlm")

SLOT_REVIEW_SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["verdicts"],
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["candidate_id", "decision", "confidence", "reason_codes",
                     "corrected_bbox_normalized", "observed_answer"],
        "properties": {
            "candidate_id": {"type": "string"},
            "decision": {"type": "string", "enum": ["ACCEPT", "REPAIR", "REJECT"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason_codes": {"type": "array", "items": {"type": "string"}},
            "corrected_bbox_normalized": {
                "type": "array", "items": {"type": "integer", "minimum": 0,
                                               "maximum": 1000}, "maxItems": 4
            },
            "observed_answer": {"type": "string"},
        },
    }}},
}

SLOT_RECHECK_SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["verdicts"],
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["candidate_id", "decision", "confidence", "reason_codes"],
        "properties": {
            "candidate_id": {"type": "string"},
            "decision": {"type": "string", "enum": ["ACCEPT", "REJECT"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason_codes": {"type": "array", "items": {"type": "string"}},
        },
    }}},
}


def _box(value: Any) -> Optional[List[int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None


def _valid(value: Any, height: int, width: int) -> bool:
    value = _box(value)
    return bool(value and 0 <= value[0] < value[2] <= height
                and 0 <= value[1] < value[3] <= width)


def _area(value: Sequence[int]) -> int:
    return max(0, value[2] - value[0]) * max(0, value[3] - value[1])


def _intersection(left: Sequence[int], right: Sequence[int]) -> int:
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1]))


def _stem_box(value: Any) -> Optional[List[int]]:
    value = _box(value)
    return [value[1], value[0], value[3], value[2]] if value else None


def _to_normalized(value: Any, roi: Sequence[int]) -> Optional[List[int]]:
    value = _box(value)
    if not value or len(roi) != 4:
        return None
    height, width = roi[2]-roi[0], roi[3]-roi[1]
    if height <= 0 or width <= 0:
        return None
    result = [
        round((value[0]-roi[0])*1000/height),
        round((value[1]-roi[1])*1000/width),
        round((value[2]-roi[0])*1000/height),
        round((value[3]-roi[1])*1000/width),
    ]
    return result if 0 <= result[0] < result[2] <= 1000 and 0 <= result[1] < result[3] <= 1000 else None


def _from_normalized(value: Any, roi: Sequence[int]) -> Optional[List[int]]:
    value = _box(value)
    if (not value or len(roi) != 4 or not
            (0 <= value[0] < value[2] <= 1000 and 0 <= value[1] < value[3] <= 1000)):
        return None
    height, width = roi[2]-roi[0], roi[3]-roi[1]
    return [
        roi[0] + round(value[0]*height/1000),
        roi[1] + round(value[1]*width/1000),
        roi[0] + round(value[2]*height/1000),
        roi[1] + round(value[3]*width/1000),
    ]


def _to_local(value: Any, roi: Sequence[int]) -> Optional[List[int]]:
    value = _box(value)
    if not value:
        return None
    return [value[0]-roi[0], value[1]-roi[1],
            value[2]-roi[0], value[3]-roi[1]]


class SlotCoordinateVisionVerifier:
    """Review coordinates and automatically repair safely bounded failures."""

    def __init__(self, request: Callable[[str, Sequence[Path], Dict[str, Any]], Dict[str, Any]],
                 model: str, minimum_confidence: float = .70,
                 auto_repair: bool = True):
        self.request = request
        self.model = str(model)
        self.minimum_confidence = float(minimum_confidence)
        self.auto_repair = bool(auto_repair)

    @staticmethod
    def _prompt(candidates: Sequence[Dict[str, Any]], height: int, width: int) -> str:
        return (
            "你看到的是单道题目的 ROI，不是整页。红框及标签是候选槽位。"
            "如果图中没有红框，表示本地未找到坐标，请直接定位该题唯一应答位置。"
            "正确返回 ACCEPT；错误但能明确定位时返回 REPAIR，并给出"
            " corrected_bbox_normalized=[y1,x1,y2,x2]；四个值都是相对当前 ROI 的"
            "0到1000整数，左上角为0，右下角为1000；不能可靠定位返回 REJECT。"
            "修正框只能紧密覆盖一个学生答案、横线、括号或合理作答区。"
            "不得覆盖整段题干、选项正文、多个空、相邻题、知识点标题或大幅插图。"
            "occupied_sibling_bboxes_normalized 是同一小题其他槽位已经占用的区域，"
            "橙框也表示这些区域；当前槽位严禁与它们重叠或重复框同一个答案。"
            "选择题只框括号内手写选项或括号应答区。空答也应框准确应答位置。"
            "候选清单中 requires_repair=true 表示本地硬门禁已经判定原框不安全，"
            "此时禁止 ACCEPT，必须 REPAIR 或 REJECT。"
            "ROI 像素 height={} width={}，但禁止返回像素坐标。ACCEPT/REJECT 的"
            " corrected_bbox_normalized"
            " 返回空数组；没有可见答案时 observed_answer 返回空字符串。"
            "每个 candidate_id 必须返回。"
            "只输出符合 Schema 的 JSON。\n候选：\n{}"
        ).format(height, width, json.dumps(list(candidates), ensure_ascii=False))

    @staticmethod
    def _recheck_prompt(candidates: Sequence[Dict[str, Any]], height: int, width: int) -> str:
        return (
            "你是槽位修正终审员。绿色框是上一轮修正坐标。只判断它是否紧密覆盖"
            "对应 candidate_id 的单个答案或应答空。覆盖题干、多个空、小题、相邻题"
            "或插图必须 REJECT，正确才 ACCEPT。不得再修改坐标。"
            "这是同一个题目 ROI，height={} width={}。逐项返回符合 Schema 的 JSON。\n修正候选：\n{}"
        ).format(height, width, json.dumps(list(candidates), ensure_ascii=False))

    @staticmethod
    def _annotate(image: Any, candidates: Sequence[Dict[str, Any]], path: Path,
                  color: Tuple[int, int, int] = (0, 0, 255), prefix: str = "S",
                  context: Sequence[Dict[str, Any]] = ()) -> None:
        import cv2
        canvas = image.copy()
        for index, candidate in enumerate(context, 1):
            y1, x1, y2, x2 = candidate["bbox"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 140, 255), 3)
            cv2.putText(canvas, "O{} occupied".format(index),
                        (max(4, x1), max(24, y1-6)), cv2.FONT_HERSHEY_SIMPLEX,
                        .58, (0, 140, 255), 2, cv2.LINE_AA)
        for index, candidate in enumerate(candidates, 1):
            y1, x1, y2, x2 = candidate["bbox"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 5)
            cv2.putText(canvas, "{}{} {}".format(prefix, index, candidate["candidate_id"]),
                        (max(4, x1), max(28, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        .72, color, 2, cv2.LINE_AA)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
            raise RuntimeError("无法写入槽位视觉验收标注图: {}".format(path))

    @staticmethod
    def _hard_gate(value: Any, candidate: Dict[str, Any], height: int, width: int,
                   check_text: bool = False) -> Dict[str, Any]:
        value = _box(value)
        if not value or not _valid(value, height, width):
            return {"accepted": False, "bbox": value,
                    "reason_codes": ["invalid_or_out_of_bounds"]}
        y1, x1, y2, x2 = value
        ratio = _area(value) / max(1.0, float(height * width))
        reasons: List[str] = []
        if ratio > (.035 if candidate.get("item_type") == "choice" else .12):
            reasons.append("box_area_too_large")
        if candidate.get("item_type") == "choice":
            if x2 - x1 > .22 * width or y2 - y1 > .12 * height:
                reasons.append("choice_box_not_local")
        elif y2 - y1 > .32 * height:
            reasons.append("answer_box_too_tall")
        corridor = candidate.get("question_corridor") or []
        if len(corridor) == 4:
            center_y, center_x = (y1+y2)/2.0, (x1+x2)/2.0
            if not (corridor[0] <= center_y <= corridor[2]
                    and corridor[1] <= center_x <= corridor[3]):
                reasons.append("outside_question_corridor")
        stem = candidate.get("stem_bbox_yxyx") or []
        if len(stem) == 4 and _area(stem) and _intersection(value, stem) / float(_area(stem)) > .62:
            reasons.append("covers_printed_stem")
        for other in candidate.get("other_stem_boxes", []):
            if _area(other) and _intersection(value, other) / float(_area(other)) > .45:
                reasons.append("crosses_other_question_stem")
                break
        for diagram in candidate.get("diagram_boxes_yxyx", []):
            if _area(value) and _intersection(value, diagram) / float(_area(value)) >= .30:
                reasons.append("overlaps_diagram_mask")
                break
        for sibling in candidate.get("sibling_slot_boxes", []):
            if (_area(value) and _area(sibling)
                    and _intersection(value, sibling)
                    / float(min(_area(value), _area(sibling))) >= .55):
                reasons.append("duplicates_sibling_slot")
                break
        if check_text and len(str(candidate.get("recognized_text") or "")) > 80 and ratio > .008:
            reasons.append("ocr_text_crosses_multiple_lines")
        return {"accepted": not reasons, "bbox": value, "area_ratio": round(ratio, 6),
                "reason_codes": list(dict.fromkeys(reasons))}

    @staticmethod
    def _commit(slot: Any, corrected: Sequence[int], record: Dict[str, Any]) -> None:
        old_expected = list(slot.expected_bbox)
        old_handwriting = list(slot.handwriting_bbox) if slot.handwriting_bbox else None
        old_text = str(slot.recognized_text or "")
        slot.expected_bbox = list(corrected)
        slot.handwriting_bbox = list(corrected) if slot.has_ink else None
        slot.geometry_status = "ALIGNED"
        slot.status = "CONVERGED_SUCCESS" if slot.has_ink else "BLANK_UNANSWERED"
        slot.review_status = "AUTO_REPAIRED"
        if old_text:
            slot.recognized_text = ""
            slot.content_status = "OCR_UNCERTAIN"
        elif not slot.has_ink:
            slot.content_status = "BLANK"
        slot.audit["vision_coordinate_repair"] = {
            "status": "AUTO_REPAIRED", "original_expected_bbox": old_expected,
            "original_handwriting_bbox": old_handwriting,
            "corrected_bbox": list(corrected), "previous_recognized_text": old_text,
            "corrected_bbox_normalized": record.get("proposed_bbox_normalized", []),
            "source_roi_bbox_yxyx": record.get("roi_bbox_yxyx", []),
            "coordinate_protocol": "item_roi_normalized_0_1000",
            "observed_answer": record.get("observed_answer", ""),
            "model": record["model"], "confidence": record["repair_confidence"],
            "policy": "model_proposal_local_hard_gate_second_visual_recheck",
        }

    def verify_package(self, package: ExamPackage, pages: Sequence[Page],
                       audit_dir: Path) -> Dict[str, Any]:
        import cv2
        page_map = {page.index: page for page in pages}
        images = {page.index: cv2.imread(page.path) for page in pages}
        corridors = QuestionLayoutService.corridors(package, page_map)
        stems: Dict[int, List[Tuple[str, List[int]]]] = {}
        for section in package.sections:
            for question in section.questions:
                for item in question.items:
                    if item.stem_region:
                        value = _stem_box(item.stem_region.bbox)
                        if value:
                            stems.setdefault(item.stem_region.page_index, []).append((item.item_id, value))

        jobs: List[Dict[str, Any]] = []
        for section in package.sections:
            for question in section.questions:
                for item in question.items:
                    stem = _stem_box(item.stem_region.bbox) if item.stem_region else None
                    for slot in item.slots:
                        page, image = page_map.get(slot.page_index), images.get(slot.page_index)
                        if not page or image is None:
                            continue
                        height, width = image.shape[:2]
                        original = _box(slot.handwriting_bbox or slot.expected_bbox)
                        corridor = corridors.get(item.item_id) or [0, 0, height, width]
                        roi = [max(0, int(corridor[0])), max(0, int(corridor[1])),
                               min(height, int(corridor[2])), min(width, int(corridor[3]))]
                        if not _valid(roi, height, width):
                            roi = [0, 0, height, width]
                        candidate_id = "{}__slot_{}".format(item.item_id, slot.slot_idx)
                        diagram_boxes = [xyxy_to_yxyx(diagram.bbox) for diagram in item.diagrams
                                         if len(diagram.bbox) >= 4]
                        if item.tri_target:
                            diagram_boxes.extend(
                                xyxy_to_yxyx(value) for value in item.tri_target.diagram_boxes
                                if len(value) >= 4
                            )
                        candidate = {
                            "candidate_id": candidate_id,
                            "original_bbox": original, "item_type": item.item_type,
                            "question_text": str(item.question_text or question.question_title or "")[:300],
                            "recognized_text": str(slot.recognized_text or "")[:200],
                            "deterministic_geometry_status": slot.geometry_status,
                            "box_kind": "handwriting" if slot.handwriting_bbox else "expected",
                            "requires_repair": not _valid(original, height, width),
                            "stem_bbox_yxyx": stem,
                            "question_corridor": corridor,
                            "roi_bbox_yxyx": roi,
                            "diagram_boxes_yxyx": diagram_boxes,
                            "other_stem_boxes": [box for key, box in stems.get(slot.page_index, [])
                                                 if key != item.item_id],
                            "sibling_slot_boxes": [
                                _box(other.handwriting_bbox or other.expected_bbox)
                                for other in item.slots if other is not slot
                                and other.page_index == slot.page_index
                                and _box(other.handwriting_bbox or other.expected_bbox)
                            ],
                        }
                        precheck = self._hard_gate(original, candidate, height, width,
                                                   check_text=True)
                        candidate["requires_repair"] = not precheck["accepted"]
                        candidate["local_precheck_reason_codes"] = precheck["reason_codes"]
                        candidate["current_bbox_normalized"] = (
                            _to_normalized(original, roi) or []
                        )
                        candidate["stem_bbox_normalized"] = _to_normalized(stem, roi) or []
                        candidate["diagram_boxes_normalized"] = [
                            value for value in (
                                _to_normalized(box, roi) for box in diagram_boxes
                            ) if value
                        ]
                        candidate["occupied_sibling_bboxes_normalized"] = [
                            value for value in (
                                _to_normalized(box, roi)
                                for box in candidate["sibling_slot_boxes"]
                            ) if value
                        ]
                        jobs.append({"page_index": slot.page_index, "slot": slot,
                                     "candidate": candidate, "image": image})

        summary = {"enabled": True, "provider": "doubao", "model": self.model,
                   "auto_repair": self.auto_repair, "pages_requested": 0,
                   "roi_requests": 0,
                   "repair_recheck_pages": 0, "candidates": len(jobs),
                   "accepted": 0, "repaired": 0, "rejected": 0, "uncertain": 0,
                   "unreviewed": len(jobs), "hard_gate_rejected": 0, "errors": []}
        page_audits: Dict[str, Any] = {}
        summary["candidates"] = len(jobs)
        summary["pages_requested"] = len({job["page_index"] for job in jobs})
        for job in jobs:
            page_index = job["page_index"]
            image = job["image"]
            slot = job["slot"]
            candidate = job["candidate"]
            height, width = image.shape[:2]
            roi = candidate["roi_bbox_yxyx"]
            crop = image[roi[0]:roi[2], roi[1]:roi[3]].copy()
            local = _to_local(candidate["original_bbox"], roi)
            safe_id = candidate["candidate_id"].replace("/", "_")
            annotated = Path(audit_dir) / "page_{:02d}_{}_roi_review.jpg".format(
                page_index, safe_id
            )
            annotations = ([{"candidate_id": candidate["candidate_id"], "bbox": local}]
                           if _valid(local, crop.shape[0], crop.shape[1]) else [])
            sibling_annotations = [
                {"candidate_id": "occupied", "bbox": value}
                for value in (
                    _to_local(box, roi) for box in candidate["sibling_slot_boxes"]
                ) if _valid(value, crop.shape[0], crop.shape[1])
            ]
            self._annotate(crop, annotations, annotated,
                           context=sibling_annotations)
            model_candidate = {key: value for key, value in candidate.items() if key not in {
                "original_bbox", "stem_bbox_yxyx", "question_corridor", "roi_bbox_yxyx",
                "diagram_boxes_yxyx", "other_stem_boxes", "sibling_slot_boxes",
            }}
            summary["roi_requests"] += 1
            try:
                response = self.request(self._prompt(
                    [model_candidate], crop.shape[0], crop.shape[1]
                ),
                                        [annotated], SLOT_REVIEW_SCHEMA)
                raw_verdicts = response.get("verdicts", []) if isinstance(response, dict) else []
            except Exception as exc:
                LOGGER.warning("slot_vlm_roi_failed page=%s candidate=%s error=%s",
                               page_index, candidate["candidate_id"], exc)
                error = {"page": page_index, "candidate_id": candidate["candidate_id"],
                         "error": "{}: {}".format(type(exc).__name__, exc)}
                summary["errors"].append(error)
                page_audits.setdefault(str(page_index), {"verdicts": []})["verdicts"].append(error)
                continue
            raw_map = {str(item.get("candidate_id") or ""): item for item in raw_verdicts
                       if isinstance(item, dict)}
            raw = raw_map.get(candidate["candidate_id"])
            if not raw:
                continue
            initial_decision = str(raw.get("decision") or "").upper()
            decision = initial_decision
            confidence = max(0., min(1., float(raw.get("confidence") or 0.)))
            reasons = [str(item) for item in raw.get("reason_codes", [])]
            current_gate = self._hard_gate(candidate["original_bbox"], candidate,
                                           height, width, check_text=True)
            if decision == "ACCEPT" and not current_gate["accepted"]:
                decision = "REJECT"; reasons += current_gate["reason_codes"]
                summary["hard_gate_rejected"] += 1
            record = {
                "candidate_id": candidate["candidate_id"],
                "initial_decision": initial_decision, "decision": decision,
                "confidence": confidence, "reason_codes": list(dict.fromkeys(reasons)),
                "original_bbox": candidate["original_bbox"],
                "current_bbox_normalized": candidate["current_bbox_normalized"],
                "roi_bbox_yxyx": roi, "current_hard_gate": current_gate,
                "model": self.model,
                "observed_answer": str(raw.get("observed_answer") or "")[:100],
                "policy": "item_roi_normalized_bounded_auto_repair_two_pass",
            }
            proposed_normalized = _box(raw.get("corrected_bbox_normalized"))
            proposed = _from_normalized(proposed_normalized, roi)
            repaired = False
            repaired_image = None
            if (self.auto_repair and confidence >= self.minimum_confidence and proposed
                    and decision in {"REPAIR", "REJECT"}):
                gate = self._hard_gate(proposed, candidate, height, width)
                record.update({"proposed_bbox_normalized": proposed_normalized,
                               "proposed_bbox": proposed, "proposal_hard_gate": gate})
                if gate["accepted"]:
                    repaired_image = Path(audit_dir) / "page_{:02d}_{}_roi_repaired.jpg".format(
                        page_index, safe_id
                    )
                    local_proposed = _to_local(proposed, roi)
                    self._annotate(crop, [{"candidate_id": candidate["candidate_id"],
                                           "bbox": local_proposed}],
                                   repaired_image, (0, 180, 0), "R",
                                   context=sibling_annotations)
                    summary["repair_recheck_pages"] += 1
                    try:
                        recheck_response = self.request(self._recheck_prompt(
                            [{"candidate_id": candidate["candidate_id"],
                              "bbox_normalized": proposed_normalized}],
                            crop.shape[0], crop.shape[1]), [repaired_image], SLOT_RECHECK_SCHEMA)
                        recheck = next((value for value in recheck_response.get("verdicts", [])
                                        if value.get("candidate_id") == candidate["candidate_id"]), None)
                    except Exception as exc:
                        recheck = None
                        summary["errors"].append({"page": page_index,
                                                  "candidate_id": candidate["candidate_id"],
                                                  "stage": "repair_recheck",
                                                  "error": "{}: {}".format(type(exc).__name__, exc)})
                    if recheck:
                        final_decision = str(recheck.get("decision") or "").upper()
                        final_confidence = max(0., min(1., float(recheck.get("confidence") or 0.)))
                        record["repair_recheck"] = {
                            "decision": final_decision, "confidence": final_confidence,
                            "reason_codes": recheck.get("reason_codes", []),
                        }
                        if final_decision == "ACCEPT" and final_confidence >= self.minimum_confidence:
                            record["repair_confidence"] = min(confidence, final_confidence)
                            self._commit(slot, proposed, record)
                            record.update({"decision": "REPAIRED", "final_bbox": proposed})
                            summary["repaired"] += 1; repaired = True
                else:
                    record["reason_codes"] = list(dict.fromkeys(
                        reasons + gate["reason_codes"]
                    ))
                    summary["hard_gate_rejected"] += 1
            if not repaired and (record["decision"] == "ACCEPT"
                                 and confidence >= self.minimum_confidence
                                 and current_gate["accepted"]):
                summary["accepted"] += 1
            elif not repaired and confidence < self.minimum_confidence:
                slot.review_status = "NEED_REVIEW"; summary["uncertain"] += 1
            elif not repaired:
                slot.geometry_status = "FAILED"; slot.review_status = "NEED_REVIEW"
                slot.status = "ANOMALY_ESCALATED"; summary["rejected"] += 1
            slot.audit["vision_coordinate_review"] = record
            summary["unreviewed"] -= 1
            page_record = page_audits.setdefault(str(page_index), {
                "roi_policy": "same_column_question_corridor",
                "candidate_count": 0, "reviewed_count": 0, "verdicts": [],
            })
            page_record["candidate_count"] += 1
            page_record["reviewed_count"] += 1
            page_record["verdicts"].append({
                **record, "annotated_image": str(annotated),
                "repaired_image": str(repaired_image) if repaired_image else None,
            })
        result = {"summary": summary, "pages": page_audits, "bbox_format": "yxyx",
                  "coordinate_protocol": "item_roi_normalized_0_1000",
                  "policy": "item_roi_model_proposal_local_hard_gate_second_visual_recheck"}
        atomic_write_json(Path(audit_dir) / "slot_vlm_audit.json", result)
        return result


__all__ = ["SLOT_REVIEW_SCHEMA", "SLOT_RECHECK_SCHEMA", "SlotCoordinateVisionVerifier"]
