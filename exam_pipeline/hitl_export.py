"""Export extraction results for the protected HITL workbench."""
import copy
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from .io_utils import atomic_write_json


def _safe(value):
    return re.sub(r"[^\w.-]+", "_", str(value).strip()).strip("._") or "unknown"


def _resolve(value, extraction_dir):
    raw = Path(value)
    for path in ([raw] if raw.is_absolute() else [Path(extraction_dir)/raw, Path(extraction_dir).parent/raw, Path.cwd()/raw]):
        if path.exists(): return path.resolve()
    raise FileNotFoundError(f"无法定位导出源文件: {value}")


def _yxyx(box):
    return [box[1], box[0], box[3], box[2]] if isinstance(box, (list, tuple)) and len(box) >= 4 else box


def _expand_slots(result, role, page_names):
    count = 0
    for section in result.get("sections", []):
        for question in section.get("questions", []):
            expanded = []
            for item in question.get("items", []):
                slots = item.get("slots") or []
                if not slots:
                    # ``answer_regions`` are question search corridors, not
                    # final answer geometry.
                    # Preserve them as diagnostics but never let the protected
                    # workbench render them as a slot-sized editable target.
                    item["search_regions"] = copy.deepcopy(
                        item.get("answer_regions") or item.get("student_regions") or []
                    )
                    if item.get("answer_bbox"):
                        item["search_bbox"] = copy.deepcopy(item["answer_bbox"])
                    item["answer_bbox"] = []
                    item["answer_boxes"] = []
                    item["answer_regions"] = []
                    item["student_regions"] = []
                    item["expected_bbox"] = []
                    item["handwriting_bbox"] = []
                    item["slot_status"] = "MISSING_SLOT_COORDINATES"
                    item["geometry_status"] = "ANOMALY_ESCALATED"
                    item["review_status"] = "HITL_REQUIRED"
                    expanded.append(item); continue
                for slot in slots:
                    clone = copy.deepcopy(item); index = int(slot.get("slot_idx", 1)); status = slot.get("status", "PENDING")
                    geometry_status = slot.get("geometry_status", "PENDING")
                    content_status = slot.get("content_status", "PENDING")
                    review_status = slot.get("review_status", "PENDING")
                    page_index = int(slot.get("page_index", 1)); expected = copy.deepcopy(slot.get("expected_bbox") or [])
                    handwriting = copy.deepcopy(slot.get("handwriting_bbox") or []); active = handwriting or expected
                    page_file = page_names[page_index-1] if 1 <= page_index <= len(page_names) else ""
                    region = {"page_index": page_index, "page_file": page_file, "bbox": active,
                              "confidence": item.get("confidence"), "ocr_text": slot.get("recognized_text", "")}
                    clone.update({"item_id": f"{item.get('item_id')}__slot_{index}", "parent_item_id": item.get("item_id"),
                                  "item_name": f"{item.get('item_name') or item.get('item_id')} · 槽位 {index} [{status}]",
                                  "slot_idx": index, "slot_type": slot.get("slot_type"), "slot_status": status,
                                  "geometry_status": geometry_status,
                                  "content_status": content_status,
                                  "review_status": review_status,
                                  "expected_bbox": expected, "handwriting_bbox": handwriting,
                                  "recognized_text": slot.get("recognized_text", ""), "answer_bbox": active,
                                  "answer_boxes": [active] if active else [], "answer_regions": [region] if active else [],
                                  "student_regions": [region] if role == "student" and active else [], "slots": [copy.deepcopy(slot)]})
                    expanded.append(clone); count += 1
            question["items"] = expanded
    return count


def _convert_package(package: Dict[str, Any], role: str, page_names: Sequence[str]):
    result = copy.deepcopy(package); convert = str(result.get("bbox_format", "xyxy")).lower() != "yxyx"
    for page in result.get("ocr", []):
        for block in page.get("blocks", []):
            if convert: block["bbox"] = _yxyx(block.get("bbox"))
    for section in result.get("sections", []):
        for question in section.get("questions", []):
            for item in question.get("items", []):
                for key in ("answer_regions", "student_regions", "option_regions", "blank_regions", "writing_regions"):
                    for region in item.get(key, []) or []:
                        if convert: region["bbox"] = _yxyx(region.get("bbox"))
                        index = int(region.get("page_index", 1));
                        if 1 <= index <= len(page_names): region["page_file"] = page_names[index-1]
                stem = item.get("stem_region")
                if isinstance(stem, dict):
                    if convert: stem["bbox"] = _yxyx(stem.get("bbox"))
                    index = int(stem.get("page_index", 1));
                    if 1 <= index <= len(page_names): stem["page_file"] = page_names[index-1]
                for diagram in item.get("diagrams", []) or []:
                    if convert: diagram["bbox"] = _yxyx(diagram.get("bbox"))
                regions = item.get("student_regions") or item.get("answer_regions") or []
                if not item.get("answer_bbox") and regions: item["answer_bbox"] = regions[0].get("bbox", [])
                item["answer_boxes"] = [r.get("bbox", []) for r in regions if r.get("bbox")]
    result["metadata"] = {**(result.get("metadata") or {}), "is_teacher_golden": role == "teacher",
                          "total_pages": len(page_names), "quality": result.get("quality", {})}
    result["page_files"] = list(page_names); result["workbench_slot_items"] = _expand_slots(result, role, page_names)
    result["source_bbox_format"] = result.get("bbox_format", "xyxy"); result["bbox_format"] = "yxyx"
    result["hitl_adapter_version"] = "2.1"
    return result


class HITLExporter:
    def export(self, extraction_dir: Path, workspace_dir: Path, subject: Optional[str] = None,
               batch_id: Optional[str] = None, overwrite: bool = False):
        extraction_dir, workspace_dir = Path(extraction_dir).resolve(), Path(workspace_dir).resolve()
        manifest = json.loads((extraction_dir/"manifest.json").read_text(encoding="utf-8"))
        docs = [d for d in manifest.get("documents", []) if not subject or d.get("subject") == subject]
        subjects = sorted({str(d.get("subject", "综合")) for d in docs})
        if not subjects: raise ValueError("没有可导出的文档，请检查 --subject 和提取目录")
        batches = []
        for subject_name in subjects:
            current = [d for d in docs if str(d.get("subject", "综合")) == subject_name]
            bid = _safe(batch_id or subject_name); batch = workspace_dir/"exams"/bid
            if batch.exists() and any(batch.iterdir()) and not overwrite: raise FileExistsError(f"批次已存在: {batch}；确认覆盖时使用 --overwrite")
            if batch.exists() and overwrite: shutil.rmtree(str(batch))
            norm, structured = batch/"normalized", batch/"structured"; students = {}
            normalized_manifest = {"batch_id": bid, "subject": subject_name, "teacher": {"pages": []}, "students": {}}
            for doc in current:
                source_json = _resolve(doc.get("output", ""), extraction_dir); package = json.loads(source_json.read_text(encoding="utf-8"))
                role = doc.get("role", package.get("document_type", "student")); sid = _safe(doc.get("student_id") or package.get("student_id") or source_json.stem)
                pages = [_resolve(value, extraction_dir) for value in (doc.get("normalized_pages") or package.get("page_files") or [])]
                if role == "teacher":
                    names = [f"teacher/page_{i:02d}.jpg" for i in range(1, len(pages)+1)]; out = structured/"teacher_exam_model.json"
                else:
                    names = [f"students/{sid}/page_{i:02d}.jpg" for i in range(1, len(pages)+1)]; out = structured/"students"/f"{sid}_evaluation.json"
                records = []
                for i, (source, name) in enumerate(zip(pages, names), 1):
                    destination = norm/name; destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(str(source), str(destination))
                    records.append({"page_index": i, "original_filename": source.name, "normalized_path": name})
                atomic_write_json(out, _convert_package(package, role, names))
                if role == "teacher": normalized_manifest["teacher"] = {"pages": records}
                else: normalized_manifest["students"][sid] = {"pages": records}; students[sid] = str(out)
            atomic_write_json(norm/"manifest.json", normalized_manifest)
            atomic_write_json(structured/"structuring_summary.json", {"batch_id": bid, "students_json": students})
            batches.append({"batch_id": bid, "subject": subject_name, "batch_dir": str(batch), "students": sorted(students)})
        index = {"adapter": "intelligent-grading-system.hitl_export", "created_at": datetime.now().astimezone().isoformat(),
                 "source": str(extraction_dir), "workspace": str(workspace_dir), "batches": batches}
        atomic_write_json(workspace_dir/"hitl_export_manifest.json", index)
        return index


__all__ = ["HITLExporter"]
