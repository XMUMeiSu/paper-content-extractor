"""Deterministic refinement from coarse VLM questions to physical sub-items."""
import copy
import re
from typing import Any, Dict, List, Sequence
from .contracts import ExamItem, ExamPackage, Page, PageRegion

_SUB_MARKER = re.compile(r"[（(]\s*(\d{1,2})\s*[）)]\s*", re.MULTILINE)
_SCORE = re.compile(r"[（(]\s*(\d+(?:\.\d+)?)\s*分\s*[）)]")
_ANSWER_CUE = re.compile(r"_{2,}|\.{4,}|…{2,}|[（(]\s*[）)]|\[\s*\]")


def _answer_for_marker(answer: Any, marker: int) -> Any:
    if isinstance(answer, dict):
        for key in (str(marker), marker, f"({marker})", f"（{marker}）"):
            if key in answer:
                return copy.deepcopy(answer[key])
        return None
    if isinstance(answer, (list, tuple)):
        return copy.deepcopy(answer[marker-1]) if marker <= len(answer) else None
    return copy.deepcopy(answer) if marker == 1 else None


def _split_regions(regions: Sequence[PageRegion], count: int, pages: Sequence[Page]):
    result = [[] for _ in range(count)]
    page_map = {p.index: p for p in pages}
    for region in regions:
        if len(region.bbox) < 4:
            continue
        left, top, right, bottom = [float(v) for v in region.bbox[:4]]
        bottom = min(bottom, float((page_map.get(region.page_index).height
                                    if page_map.get(region.page_index) else None) or 2338))
        if bottom <= top:
            continue
        step = (bottom-top) / count
        for index in range(count):
            y1, y2 = top + index*step, bottom if index+1 == count else top + (index+1)*step
            result[index].append(PageRegion(region.page_index, region.page_file,
                                            [left, round(y1, 2), right, round(y2, 2)],
                                            region.confidence, ""))
    return result


def _physical_marker_regions(markers, pages: Sequence[Page], source_regions):
    """Locate numbered child lines instead of assuming equal-height children."""
    preferred_pages = {region.page_index for region in source_regions}
    anchors = []
    used = set()
    for marker in markers:
        pattern = re.compile(rf"[（(]\s*{marker}\s*[）)]")
        choices = []
        for page in pages:
            if preferred_pages and page.index not in preferred_pages:
                continue
            for block_index, block in enumerate(page.ocr):
                inside = any(r.page_index == page.index and len(r.bbox) == 4
                             and r.bbox[0] <= (block.bbox[0]+block.bbox[2])/2 <= r.bbox[2]
                             and r.bbox[1] <= (block.bbox[1]+block.bbox[3])/2 <= r.bbox[3]
                             for r in source_regions) if len(block.bbox) == 4 else False
                if inside and pattern.search(block.text or ""):
                    key = (page.index, block_index)
                    if key not in used:
                        choices.append((page, block_index, block))
        if not choices:
            return None
        page, block_index, block = min(choices, key=lambda value: (value[0].index, value[2].bbox[1]))
        used.add((page.index, block_index))
        anchors.append((page, block))
    physical_order = [(page.index, float(block.bbox[1])) for page, block in anchors]
    if physical_order != sorted(physical_order):
        return None
    result = []
    for index, (page, anchor) in enumerate(anchors):
        page_width, page_height = float(page.width or 1654), float(page.height or 2338)
        next_anchor = anchors[index + 1] if index + 1 < len(anchors) else None
        if next_anchor and next_anchor[0].index == page.index:
            bottom = (float(anchor.bbox[3]) + float(next_anchor[1].bbox[1])) / 2.0
        else:
            source_bottoms = [float(region.bbox[3]) for region in source_regions
                              if region.page_index == page.index and len(region.bbox) >= 4
                              and float(region.bbox[3]) > float(anchor.bbox[3])]
            bottom = min(source_bottoms, default=page_height - 20.0)
        same_band = [block for block in page.ocr if len(block.bbox) >= 4
                     and float(block.bbox[1]) >= float(anchor.bbox[1]) - 6
                     and float(block.bbox[3]) <= bottom + 6]
        left = max(0.0, min((float(block.bbox[0]) for block in same_band), default=float(anchor.bbox[0])) - 24)
        right = min(page_width, max((float(block.bbox[2]) for block in same_band), default=float(anchor.bbox[2])) + 24)
        result.append([PageRegion(page.index, page.path,
                                  [round(left, 2), round(float(anchor.bbox[1]), 2),
                                   round(right, 2), round(bottom, 2)],
                                  anchor.confidence, anchor.text)])
    return result


class FineGrainedItemSplitter:
    def split_item(self, item: ExamItem, question_id: str, question_num: int,
                   pages: Sequence[Page]) -> List[ExamItem]:
        text = str(item.question_text or "")
        matches = list(_SUB_MARKER.finditer(text))
        # Ignore a repeated numbering sequence in handwritten working after
        # the printed prompts; retain the first increasing sequence.
        sequence = []
        for match in matches:
            value = int(match.group(1))
            if sequence and value <= int(sequence[-1].group(1)):
                break
            sequence.append(match)
        matches = sequence
        markers = [int(match.group(1)) for match in matches]
        if len(matches) < 2 or len(set(markers)) != len(markers) or any(
                right <= left for left, right in zip(markers, markers[1:])):
            return [item]
        source_regions = item.student_regions or item.answer_regions
        physical = _physical_marker_regions(markers, pages, source_regions)
        # A semantic split must not invent equal-height physical answers.
        bands = physical or [copy.deepcopy(source_regions) for _ in matches]
        segments = []
        for index, match in enumerate(matches):
            end = matches[index+1].start() if index+1 < len(matches) else len(text)
            segments.append(text[match.start():end].strip())
        # A coarse VLM item can legitimately report the aggregate number of
        # answer points (for example q11 has 8 blanks across five sub-items).
        # Broadcasting that aggregate to every child turns 8 into 40.  When
        # the printed cues account for the aggregate exactly, deterministically
        # distribute the count to the physical children instead.
        cue_counts = [len(_ANSWER_CUE.findall(segment)) for segment in segments]
        parent_count = int(item.expected_slot_count or 0)
        distribute_cues = bool(
            parent_count > 0
            and all(count > 0 for count in cue_counts)
            and sum(cue_counts) == parent_count
        )
        children = []
        for index, (match, marker) in enumerate(zip(matches, markers)):
            segment = segments[index]
            child = copy.deepcopy(item)
            child.item_id = f"{question_id}_{marker}"
            child.item_name = f"{question_num}.({marker})"
            child.question_text = text[:matches[0].start()].strip() + "\n" + segment
            child.item_type = "solve"
            child.standard_answer = _answer_for_marker(item.standard_answer, marker)
            score = _SCORE.search(segment)
            child.item_score = float(score.group(1)) if score else None
            child.slots = []; child.roi_patch = None; child.tri_target = None; child.quality = {}
            child.expected_slot_count = None
            child.slot_count_source = ""
            if not physical:
                child.quality["geometry_status"] = "UNRESOLVED_SUBITEM_BOUNDARY"
            child.answer_regions = copy.deepcopy(bands[index])
            if physical and bands[index]:
                child.stem_region = copy.deepcopy(bands[index][0])
            child.student_regions = copy.deepcopy(bands[index]) if item.student_regions else []
            child.option_regions = []
            child.blank_regions = copy.deepcopy(bands[index]) if item.blank_regions else []
            child.writing_regions = copy.deepcopy(bands[index]) if item.writing_regions else []
            if distribute_cues:
                child.expected_slot_count = cue_counts[index]
                child.slot_count_source = (
                    f"{item.slot_count_source or 'semantic'}+subitem_printed_cues"
                )
            children.append(child)
        return children

    def enrich_package(self, package: ExamPackage, pages: Sequence[Page]) -> Dict[str, int]:
        original = final = split = 0
        for section in package.sections:
            for question in section.questions:
                refined = []
                for item in question.items:
                    original += 1
                    children = self.split_item(item, question.question_id, question.question_num, pages)
                    split += int(len(children) > 1)
                    refined.extend(children)
                question.items = refined
                final += len(refined)
        if split:
            package.warnings.append(f"细粒度小题拆分：{split} 个粗粒度 Item 拆为 {final-original+split} 个子 Item")
        return {"original_items": original, "final_items": final, "coarse_items_split": split}


__all__ = ["FineGrainedItemSplitter"]
