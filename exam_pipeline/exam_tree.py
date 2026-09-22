"""Versioned, validated and reusable logical exam tree."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .contracts import (
    ExamItem, ExamPackage, ExamQuestion, ExamSection, PageRegion, Slot,
)
from .io_utils import atomic_write_json
from .slots import infer_item_type


SCHEMA_VERSION = "exam_tree.v1"
ALLOWED_ITEM_TYPES = {
    "choice", "fill", "grid", "large_writing", "calculation", "proof",
    "drawing", "reading", "essay", "solve", "other",
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _canonical_payload(tree: Dict[str, Any]) -> Dict[str, Any]:
    payload = copy.deepcopy(tree)
    # Identity is based on the reusable logical contract, not on validation
    # timestamps or runtime provenance. Equivalent trees remain comparable
    # across reruns and machines.
    for key in ("fingerprint", "validation", "created_at", "updated_at",
                "state", "provenance"):
        payload.pop(key, None)
    return payload


def tree_fingerprint(tree: Dict[str, Any]) -> str:
    encoded = json.dumps(
        _canonical_payload(tree), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _anchor(item: ExamItem) -> Optional[Dict[str, Any]]:
    region = item.stem_region
    if region is None:
        regions = item.answer_regions or item.student_regions
        region = regions[0] if regions else None
    if region is None or len(region.bbox) < 4:
        return None
    return {
        "page_index": int(region.page_index),
        "bbox": [float(value) for value in region.bbox[:4]],
        "bbox_format": "xyxy",
        "confidence": region.confidence,
    }


_ANSWER_CUE = re.compile(r"_{2,}|\.{3,}|…{2,}|（\s*）|\(\s*\)|\[\s*\]")


def _semantic_contexts(text: str, count: int) -> List[Tuple[str, str]]:
    """Return stable text surrounding each answer point.

    A blueprint must remain useful when concrete coordinates are regenerated
    on another paper, so it stores semantic neighbours rather than copying a
    teacher/student box as truth.
    """
    raw = str(text or "").strip()
    matches = list(_ANSWER_CUE.finditer(raw))
    contexts: List[Tuple[str, str]] = []
    for match in matches[:count]:
        contexts.append((raw[max(0, match.start()-48):match.start()].strip(),
                         raw[match.end():match.end()+48].strip()))
    while len(contexts) < count:
        # OCR sometimes drops the actual underline/bracket. Retain a bounded
        # question-text anchor and let the evidence gate decide whether this
        # synthesized point is safe to lock.
        contexts.append((raw[-96:] if raw else "", ""))
    return contexts


def _slot_blueprint(item: ExamItem, expected_count: int) -> List[Dict[str, Any]]:
    slots = sorted(item.slots, key=lambda value: value.slot_idx)
    if item.semantic_slot_plan:
        slots = [next((s for s in slots if s.semantic_id == entry["slot_id"]), None)
                 for entry in item.semantic_slot_plan]
    contexts = _semantic_contexts(item.question_text, expected_count)
    anchor = _anchor(item) or {}
    evidence = copy.deepcopy(item.cardinality_evidence or {})
    result = []
    for local_index in range(1, expected_count + 1):
        slot = slots[local_index-1] if local_index <= len(slots) else None
        before, after = contexts[local_index-1]
        if slot:
            before = slot.anchor_before or before
            after = slot.anchor_after or after
        slot_type = str((slot.slot_type if slot else "semantic_answer_point") or
                        "semantic_answer_point")
        result.append({
            # slot_idx is local to an ExamItem. Upstream splitting can move a
            # global slot into a child item where it must restart from one.
            "slot_idx": local_index,
            "slot_id": f"{item.item_id}:slot:{local_index}",
            "slot_type": slot_type,
            "cue_type": slot_type,
            "answer_point": copy.deepcopy(
                (slot.expected_text if slot else None) or f"answer_point_{local_index}"
            ),
            "expected_text": copy.deepcopy(slot.expected_text if slot else None),
            "page_index": int((slot.page_index if slot else None) or
                              anchor.get("page_index") or 1),
            "anchor_before": before,
            "anchor_after": after,
            "relative_order": local_index,
            "confidence": float(min(1.0, max(0.0,
                (slot.audit.get("candidate_confidence", item.confidence)
                 if slot else item.confidence) or 0.0))),
            "evidence": evidence,
        })
    return result


class ExamTreeService:
    """Compile, validate, persist and apply a logical tree to ExamPackages."""

    @staticmethod
    def compile(package: ExamPackage, revision: int = 1,
                provenance: Optional[Dict[str, Any]] = None,
                production_strict: bool = False) -> Dict[str, Any]:
        sections = []
        for section_order, section in enumerate(package.sections, 1):
            questions = []
            for question_order, question in enumerate(section.questions, 1):
                items = []
                for item_order, item in enumerate(question.items, 1):
                    slot_count = item.expected_slot_count or len(item.slots) or None
                    blueprint_count = int(slot_count or 0)
                    items.append({
                        "item_id": str(item.item_id),
                        "item_name": str(item.item_name or item.item_id),
                        "order": item_order,
                        "item_type": infer_item_type(item.question_text, item.item_type),
                        "question_text": str(item.question_text or ""),
                        "standard_answer": copy.deepcopy(item.standard_answer),
                        "item_score": item.item_score,
                        "rubric": item.rubric,
                        "expected_slot_count": int(slot_count) if slot_count else None,
                        # This is deliberately separate from the logical
                        # blueprint count.  Production validation must prove
                        # that every teacher answer point was localized; a
                        # semantic placeholder is not a coordinate.
                        "semantic_slot_plan": copy.deepcopy(item.semantic_slot_plan),
                        "slot_semantics_audit": copy.deepcopy(item.slot_semantics_audit),
                        "structure_references": copy.deepcopy(
                            item.quality.get("structure_references") or []
                        ),
                        "localized_slot_count": len({s.semantic_id or str(s.slot_idx) for s in item.slots
                                                     if len(s.expected_bbox or []) == 4}),
                        "physical_region_count": len(item.slots),
                        "slot_count_source": str(item.slot_count_source or (
                            "teacher_topology" if item.slots else "unknown"
                        )),
                        "anchor": _anchor(item),
                        "slot_blueprint": _slot_blueprint(item, blueprint_count),
                        "cardinality_evidence": copy.deepcopy(
                            item.cardinality_evidence or {}
                        ),
                    })
                questions.append({
                    "question_id": str(question.question_id),
                    "question_num": int(question.question_num),
                    "question_title": str(question.question_title or ""),
                    "order": question_order,
                    "question_score": question.question_score,
                    "items": items,
                })
            sections.append({
                "section_id": str(section.section_id),
                "section_title": str(section.section_title or ""),
                "order": section_order,
                "section_score": section.section_score,
                "questions": questions,
            })

        # 自动去重：移除重复的 item_id，优先保留有内容的项目
        if package.structure_audit.get("topology_source") != "vlm":
            sections = ExamTreeService._deduplicate_items(sections)
        tree = {
            "schema_version": SCHEMA_VERSION,
            "tree_id": f"{package.subject}:exam_tree",
            "subject": package.subject,
            "source_exam_id": package.exam_id,
            "revision": max(1, int(revision)),
            "state": "DRAFT",
            "created_at": _now(),
            "updated_at": _now(),
            "canonical_canvas": {"width": 1654, "height": 2338},
            "production_policy": {
                "strict": bool(production_strict),
                "required_cardinality_sources": ["available_independent_evidence"],
            },
            "sections": sections,
            "provenance": provenance or {
                "kind": "teacher_extraction",
                "golden_source": package.golden_source,
            },
        }
        return ExamTreeService.finalize(tree)

    @staticmethod
    def _deduplicate_items(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        自动去重：移除重复的 item_id，优先保留有内容的项目。

        评分标准（分数越高越优先保留）：
        1. 有题目文本（question_text 非空且不是 "None"）：+100
        2. 有标准答案（standard_answer 非空）：+50
        3. 有槽位（expected_slot_count > 0）：+30
        4. 有锚点（anchor 非空）：+20
        5. 不是 composite 类型：+10
        """
        seen_item_ids: Dict[str, Tuple[int, int, int, Dict[str, Any]]] = {}  # item_id -> (score, section_idx, question_idx, item)

        def score_item(item: Dict[str, Any]) -> int:
            """计算项目的质量分数"""
            score = 0
            question_text = str(item.get("question_text", "")).strip()
            if question_text and question_text.lower() not in ("", "none", "null"):
                score += 100
            if item.get("standard_answer"):
                score += 50
            expected_count = item.get("expected_slot_count")
            if expected_count is not None and expected_count > 0:
                score += 30
            if item.get("anchor"):
                score += 20
            if item.get("item_type", "") != "composite":
                score += 10
            return score

        # 第一遍：找出所有重复的 item_id 并记录最优项目
        for section_idx, section in enumerate(sections):
            for question_idx, question in enumerate(section.get("questions", [])):
                for item in question.get("items", []):
                    item_id = str(item.get("item_id", ""))
                    if not item_id:
                        continue

                    current_score = score_item(item)

                    if item_id in seen_item_ids:
                        existing_score, _, _, _ = seen_item_ids[item_id]
                        if current_score > existing_score:
                            # 当前项目更优，替换
                            seen_item_ids[item_id] = (current_score, section_idx, question_idx, item)
                    else:
                        seen_item_ids[item_id] = (current_score, section_idx, question_idx, item)

        # 第二遍：移除重复项目，只保留最优的
        deduplicated_sections = []
        for section_idx, section in enumerate(sections):
            new_questions = []
            for question_idx, question in enumerate(section.get("questions", [])):
                new_items = []
                for item in question.get("items", []):
                    item_id = str(item.get("item_id", ""))
                    if not item_id:
                        new_items.append(item)
                        continue

                    # 检查这个项目是否是最优的
                    best_score, best_section_idx, best_question_idx, best_item = seen_item_ids[item_id]
                    if (section_idx == best_section_idx and
                        question_idx == best_question_idx and
                        item is best_item):
                        # 这是最优项目，保留
                        new_items.append(item)
                    # 否则跳过这个重复项目

                if new_items:
                    new_question = dict(question)
                    new_question["items"] = new_items
                    new_questions.append(new_question)

            if new_questions:
                new_section = dict(section)
                new_section["questions"] = new_questions
                deduplicated_sections.append(new_section)

        return deduplicated_sections

    @staticmethod
    def iter_items(tree: Dict[str, Any]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
        for section in tree.get("sections", []) or []:
            for question in section.get("questions", []) or []:
                for item in question.get("items", []) or []:
                    yield section, question, item

    @staticmethod
    def validate(tree: Dict[str, Any], production: Optional[bool] = None) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []
        if tree.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        if not str(tree.get("subject") or "").strip():
            errors.append("subject is required")
        sections = tree.get("sections")
        if not isinstance(sections, list) or not sections:
            errors.append("sections must be a non-empty array")
            sections = []

        strict = (bool((tree.get("production_policy") or {}).get("strict"))
                  if production is None else bool(production))
        coverage = (tree.get("provenance") or {}).get("structure_coverage") or {}
        if strict and (not coverage or coverage.get("status") != "COMPLETE"):
            errors.append("structure coverage unresolved")
        elif coverage and coverage.get("status") != "COMPLETE":
            warnings.append("structure coverage unresolved")
        required_blueprint_fields = {
            "slot_idx", "slot_id", "slot_type", "cue_type", "answer_point",
            "page_index", "anchor_before", "anchor_after", "relative_order",
            "confidence", "evidence",
        }
        section_ids, question_ids, item_ids = set(), set(), set()
        canvas = tree.get("canonical_canvas") or {}
        width, height = float(canvas.get("width") or 1654), float(canvas.get("height") or 2338)
        question_count = item_count = slot_count = 0
        for section_index, section in enumerate(sections, 1):
            sid = str(section.get("section_id") or "")
            if not sid:
                errors.append(f"section[{section_index}] missing section_id")
            elif sid in section_ids:
                errors.append(f"duplicate section_id: {sid}")
            section_ids.add(sid)
            questions = section.get("questions")
            if not isinstance(questions, list) or not questions:
                errors.append(f"section {sid or section_index} has no questions")
                continue
            numbers = []
            for question in questions:
                question_count += 1
                qid = str(question.get("question_id") or "")
                if not qid:
                    errors.append(f"section {sid} contains a question without question_id")
                elif qid in question_ids:
                    errors.append(f"duplicate question_id: {qid}")
                question_ids.add(qid)
                try:
                    number = int(question.get("question_num"))
                    numbers.append(number)
                    if number < 1:
                        errors.append(f"question {qid} has invalid question_num")
                except (TypeError, ValueError):
                    errors.append(f"question {qid or '?'} has invalid question_num")
                items = question.get("items")
                if not isinstance(items, list) or not items:
                    errors.append(f"question {qid or '?'} has no items")
                    continue
                for item in items:
                    item_count += 1
                    iid = str(item.get("item_id") or "")
                    if not iid:
                        errors.append(f"question {qid} contains an item without item_id")
                    elif iid in item_ids:
                        errors.append(f"duplicate item_id: {iid}")
                    item_ids.add(iid)
                    kind = str(item.get("item_type") or "other")
                    if kind not in ALLOWED_ITEM_TYPES:
                        warnings.append(f"item {iid} has non-standard item_type: {kind}")
                    expected = item.get("expected_slot_count")
                    if expected is not None:
                        try:
                            expected = int(expected)
                            if expected < 1 or expected > 100:
                                errors.append(f"item {iid} expected_slot_count outside 1..100")
                        except (TypeError, ValueError):
                            errors.append(f"item {iid} expected_slot_count is not an integer")
                    blueprint = item.get("slot_blueprint") or []
                    slot_count += len(blueprint)
                    indexes = [entry.get("slot_idx") for entry in blueprint]
                    if indexes and indexes != list(range(1, len(indexes)+1)):
                        errors.append(f"item {iid} slot_blueprint indexes must be contiguous from 1")
                    if expected and expected != len(blueprint):
                        message = (f"item {iid} expects {expected} slots but blueprint "
                                   f"contains {len(blueprint)}")
                        (errors if strict else warnings).append(message)
                    if strict:
                        localized = item.get("localized_slot_count")
                        if not isinstance(localized, int) or localized < 0:
                            errors.append(
                                f"item {iid} missing localized_slot_count"
                            )
                        elif expected and localized != expected:
                            errors.append(
                                f"item {iid} expects {expected} slots but only "
                                f"{localized} teacher slots have concrete coordinates"
                            )
                    if strict and not blueprint:
                        errors.append(f"item {iid} has empty slot_blueprint")
                    if strict:
                        for entry in blueprint:
                            missing = sorted(required_blueprint_fields - set(entry))
                            if missing:
                                errors.append(
                                    f"item {iid} slot {entry.get('slot_idx', '?')} "
                                    f"blueprint missing fields: {','.join(missing)}"
                                )
                            if not (str(entry.get("anchor_before") or "").strip()
                                    or str(entry.get("anchor_after") or "").strip()):
                                errors.append(
                                    f"item {iid} slot {entry.get('slot_idx', '?')} "
                                    "blueprint has no semantic anchor"
                                )
                        card = item.get("cardinality_evidence") or {}
                        if card.get("decision") == "CONFLICT":
                            errors.append(f"item {iid} slot-count evidence conflicts")
                    anchor = item.get("anchor")
                    if strict and not anchor:
                        errors.append(f"item {iid} missing anchor")
                    if anchor:
                        box = anchor.get("bbox") or []
                        if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                            errors.append(f"item {iid} has invalid anchor bbox")
                        elif box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
                            errors.append(f"item {iid} anchor bbox is outside canonical canvas")
                        if int(anchor.get("page_index") or 0) < 1:
                            errors.append(f"item {iid} anchor page_index must be positive")
            if len(numbers) != len(set(numbers)):
                warnings.append(f"section {sid} contains repeated question numbers")
            if numbers and numbers != sorted(numbers):
                warnings.append(f"section {sid} question numbers are not in reading order")

        readiness = []
        if not coverage or coverage.get("status") != "COMPLETE":
            readiness.append("STRUCTURE_INCOMPLETE")
        for _, _, item in ExamTreeService.iter_items(tree):
            if item.get("slot_semantics_audit") and item["slot_semantics_audit"].get("status") != "ACCEPTED":
                readiness.append("SLOT_SEMANTICS_UNRESOLVED:" + item["item_id"])
            if not item.get("slot_blueprint"):
                readiness.append("SLOTS_UNRESOLVED:" + item["item_id"])
            if (item.get("cardinality_evidence") or {}).get("decision") == "CONFLICT":
                readiness.append("SLOT_COUNT_CONFLICT:" + item["item_id"])
        report = {
            "ready_to_lock": not errors and not readiness,
            "readiness_issues": readiness,
            "status": "VALID" if not errors else "INVALID",
            "production_strict": strict,
            "errors": list(dict.fromkeys(errors)),
            "warnings": list(dict.fromkeys(warnings)),
            "counts": {
                "sections": len(sections), "questions": question_count,
                "items": item_count, "slot_blueprints": slot_count,
            },
            "validated_at": _now(),
        }
        return report

    @staticmethod
    def finalize(tree: Dict[str, Any], lock: bool = False) -> Dict[str, Any]:
        result = copy.deepcopy(tree)
        result["schema_version"] = result.get("schema_version") or SCHEMA_VERSION
        result["updated_at"] = _now()
        report = ExamTreeService.validate(result)
        if lock and report["errors"]:
            raise ValueError("ExamTree 校验失败: " + "; ".join(report["errors"]))
        result["state"] = "LOCKED" if lock and report["ready_to_lock"] else "DRAFT"
        result["validation"] = report
        result["fingerprint"] = tree_fingerprint(result)
        return result

    @staticmethod
    def load(path: Path, expected_subject: Optional[str] = None,
             require_valid: bool = True,
             require_production: bool = False,
             allow_valid_draft: bool = False) -> Dict[str, Any]:
        tree = json.loads(Path(path).read_text(encoding="utf-8"))
        if expected_subject and str(tree.get("subject")) != str(expected_subject):
            raise ValueError(
                f"ExamTree 学科不匹配: expected={expected_subject}, actual={tree.get('subject')}"
            )
        if require_valid and tree.get("state") != "LOCKED" and not allow_valid_draft:
            raise ValueError("生产 ExamTree 必须先使用 exam_tree_tool.py lock 锁定")
        declared_fingerprint = str(tree.get("fingerprint") or "")
        actual_fingerprint = tree_fingerprint(tree)
        if (require_valid and declared_fingerprint
                and declared_fingerprint != actual_fingerprint):
            raise ValueError("ExamTree 指纹不匹配，文件可能在锁定后被修改")
        report = ExamTreeService.validate(tree)
        if require_valid and report["errors"]:
            raise ValueError("ExamTree 校验失败: " + "; ".join(report["errors"]))
        if require_production:
            tree.setdefault("production_policy", {})["strict"] = True
        tree = ExamTreeService.finalize(
            tree, lock=require_valid and not allow_valid_draft)
        return tree

    @staticmethod
    def save(tree: Dict[str, Any], path: Path, lock: bool = True) -> Dict[str, Any]:
        finalized = ExamTreeService.finalize(tree, lock=lock)
        atomic_write_json(Path(path), finalized)
        return finalized

    @staticmethod
    def apply_to_package(tree: Dict[str, Any], package: ExamPackage) -> ExamPackage:
        report = ExamTreeService.validate(tree)
        if report["errors"]:
            raise ValueError("不能应用无效 ExamTree: " + "; ".join(report["errors"]))
        if str(tree.get("subject")) != str(package.subject):
            raise ValueError("ExamTree 与 ExamPackage 学科不一致")

        existing = {
            item.item_id: item for section in package.sections
            for question in section.questions for item in question.items
        }
        page_files = package.page_files
        rebuilt_sections: List[ExamSection] = []
        for section_data in sorted(tree.get("sections", []), key=lambda value: value.get("order", 0)):
            questions: List[ExamQuestion] = []
            for question_data in sorted(section_data.get("questions", []), key=lambda value: value.get("order", 0)):
                items: List[ExamItem] = []
                for item_data in sorted(question_data.get("items", []), key=lambda value: value.get("order", 0)):
                    iid = str(item_data["item_id"])
                    item = copy.deepcopy(existing.get(iid)) if iid in existing else ExamItem(
                        iid, str(item_data.get("item_name") or iid)
                    )
                    item.item_name = str(item_data.get("item_name") or iid)
                    item.item_type = str(item_data.get("item_type") or "other")
                    item.question_text = str(item_data.get("question_text") or "")
                    item.standard_answer = copy.deepcopy(item_data.get("standard_answer"))
                    item.item_score = item_data.get("item_score")
                    item.rubric = item_data.get("rubric")
                    item.semantic_slot_plan = copy.deepcopy(item_data.get("semantic_slot_plan") or [])
                    item.slot_semantics_audit = copy.deepcopy(item_data.get("slot_semantics_audit") or {})
                    item.quality["structure_references"] = copy.deepcopy(
                        item_data.get("structure_references") or []
                    )
                    item.expected_slot_count = item_data.get("expected_slot_count")
                    item.slot_count_source = str(item_data.get("slot_count_source") or "exam_tree")
                    item.cardinality_evidence = copy.deepcopy(
                        item_data.get("cardinality_evidence") or {}
                    )
                    anchor = item_data.get("anchor")
                    if anchor and len(anchor.get("bbox") or []) == 4:
                        page_index = int(anchor.get("page_index", 1))
                        page_file = page_files[page_index-1] if 1 <= page_index <= len(page_files) else ""
                        item.stem_region = PageRegion(
                            page_index, page_file, list(anchor["bbox"]),
                            anchor.get("confidence"), item.question_text,
                        )
                    blueprint = item_data.get("slot_blueprint") or []
                    old_slots = sorted(item.slots, key=lambda slot: slot.slot_idx)
                    if item.semantic_slot_plan and old_slots:
                        item.slots = old_slots
                    elif blueprint and item.item_type == "large_writing" and old_slots:
                        for old_slot in old_slots:
                            old_slot.semantic_id = str(blueprint[0]["slot_id"])
                            old_slot.anchor_before = str(blueprint[0].get("anchor_before") or "")
                            old_slot.anchor_after = str(blueprint[0].get("anchor_after") or "")
                    elif blueprint:
                        # Preserve geometry only when the blueprint still maps
                        # one-to-one. A human cardinality edit clears stale
                        # boxes so the detector must localize the new topology.
                        # Match by reading order, not by the upstream slot_idx:
                        # splitting q1 slot 6 into q1_6 makes its local index 1.
                        ordered_blueprint = sorted(
                            blueprint, key=lambda entry: int(entry["slot_idx"])
                        )
                        if len(ordered_blueprint) == len(old_slots):
                            new_slots = []
                            for entry, old_slot in zip(ordered_blueprint, old_slots):
                                index = int(entry["slot_idx"])
                                slot = copy.deepcopy(old_slot)
                                slot.slot_idx = index
                                slot.parent_item_id = iid
                                slot.slot_type = str(entry.get("slot_type") or slot.slot_type)
                                slot.expected_text = copy.deepcopy(entry.get("expected_text"))
                                slot.semantic_id = str(entry.get("slot_id") or "")
                                slot.anchor_before = str(entry.get("anchor_before") or "")
                                slot.anchor_after = str(entry.get("anchor_after") or "")
                                slot.audit["cardinality_confirmed"] = True
                                slot.audit["topology_source"] = "exam_tree"
                                new_slots.append(slot)
                            item.slots = new_slots
                        elif not old_slots and len(ordered_blueprint) > 0:
                            # First-time application: reconstruct slots from
                            # cardinality evidence when available. The blueprint
                            # itself only holds semantic metadata; geometry lives
                            # in the audit trail.
                            cardinality_evidence = item_data.get("cardinality_evidence") or {}
                            layout_source = cardinality_evidence.get("sources", {}).get("layout", {})
                            layout_boxes = layout_source.get("boxes") or []
                            if len(layout_boxes) == len(ordered_blueprint):
                                new_slots = []
                                for entry, bbox in zip(ordered_blueprint, layout_boxes):
                                    index = int(entry["slot_idx"])
                                    page_idx = int(entry.get("page_index") or 1)
                                    page_file = page_files[page_idx-1] if 1 <= page_idx <= len(page_files) else ""
                                    slot = Slot(
                                        slot_idx=index,
                                        slot_type=str(entry.get("slot_type") or "semantic_answer_point"),
                                        parent_item_id=iid, expected_bbox=list(bbox),
                                        page_index=page_idx,
                                    )
                                    slot.expected_text = copy.deepcopy(entry.get("expected_text"))
                                    slot.semantic_id = str(entry.get("slot_id") or "")
                                    slot.anchor_before = str(entry.get("anchor_before") or "")
                                    slot.anchor_after = str(entry.get("anchor_after") or "")
                                    slot.audit["cardinality_confirmed"] = True
                                    slot.audit["topology_source"] = "exam_tree_layout_evidence"
                                    slot.audit["promoted_from"] = "cardinality_consensus_layout"
                                    new_slots.append(slot)
                                item.slots = new_slots
                            else:
                                item.slots = []
                        else:
                            item.slots = []
                    items.append(item)
                questions.append(ExamQuestion(
                    str(question_data["question_id"]), int(question_data["question_num"]),
                    str(question_data.get("question_title") or ""), items,
                    question_data.get("question_score"),
                ))
            rebuilt_sections.append(ExamSection(
                str(section_data["section_id"]),
                str(section_data.get("section_title") or ""), questions,
                section_data.get("section_score"),
            ))
        package.sections = rebuilt_sections
        package.topology_locked = tree.get("state") == "LOCKED"
        package.exam_tree_id = str(tree.get("tree_id") or "")
        package.exam_tree_revision = int(tree.get("revision") or 1)
        package.exam_tree_fingerprint = str(tree.get("fingerprint") or tree_fingerprint(tree))
        return package


def resolve_override(root: Optional[Path], subject: str) -> Optional[Path]:
    if root is None:
        return None
    root = Path(root)
    if root.is_file():
        return root
    safe = "".join(character if character.isalnum() or character in "_.-" else "_"
                   for character in subject)
    candidate = root / f"{safe}.json"
    return candidate if candidate.exists() else None


__all__ = [
    "ALLOWED_ITEM_TYPES", "ExamTreeService", "SCHEMA_VERSION",
    "resolve_override", "tree_fingerprint",
]
