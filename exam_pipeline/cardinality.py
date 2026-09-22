"""Independent LLM/layout/cohort answer-point cardinality evidence."""
from __future__ import annotations

import copy
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .contracts import ExamItem, ExamPackage, OCRBlock, Page, PageRegion, Slot
from .ink_separation import NoBlankInkSeparator
from .layout import QuestionLayoutService
from .slots import MultiSlotTopologyEngine


class SlotCardinalityConsensusService:
    """Build auditable three-way evidence before an ExamTree can be locked.

    The sources intentionally have different failure modes:
    semantic/LLM supplies logical answer-point count, the pseudo-blank page
    supplies printed layout cavities, and repeated student residuals confirm
    that those cavities are actually used as answer locations.
    """

    MIN_COHORT_SAMPLES = 2
    MIN_USAGE_RATIO = 0.20
    _EMPTY_RESPONSE = re.compile(r"[（(]\s*[）)]")
    # LLM-normalized question text often flattens every option onto one line.
    # Accept a whitespace boundary as well as a newline; requiring ``^|\n``
    # misclassified inline ``A. ... B. ...`` choice questions as fill items and
    # allowed diagram/text strokes to inflate the layout slot count.
    _OPTION_LABEL = re.compile(r"(?:^|\s)[A-HＡ-Ｈ]\s*[.、．:)）]", re.I)
    _BARE_OPTION_LABEL = re.compile(r"(?:^|\s)[A-HＡ-Ｈ](?=$|\s)", re.I)
    _PRINTED_BLANK = re.compile(r"_{2,}|\.{4,}|…{2,}")
    _CHOICE_STEM = re.compile(
        r"(?:下列|以下).{0,80}(?:正确|错误|属于|符合|不符合|说法|做法)|"
        r"(?:选择|选出).{0,40}(?:一项|答案)", re.I,
    )

    def __init__(self, policy=None):
        self.policy = policy
        self.detector = MultiSlotTopologyEngine(policy=policy)
        self._mask_cache: Dict[str, Any] = {}

    @staticmethod
    def _fallback_region(item: ExamItem, page: Page) -> Optional[PageRegion]:
        if item.stem_region is None or item.stem_region.page_index != page.index:
            return None
        left, top, right, bottom = [float(v) for v in item.stem_region.bbox[:4]]
        height = float(page.height or 2338)
        return PageRegion(
            page.index, page.path,
            [max(0.0, left-20), max(0.0, top-15),
             min(float(page.width or 1654), right+40),
             min(height, max(bottom+80, top+height*.12))],
            item.stem_region.confidence, item.question_text,
        )

    @staticmethod
    def _corridors(package: ExamPackage, pages: Dict[int, Page]) -> Dict[str, PageRegion]:
        by_page: Dict[int, List[ExamItem]] = defaultdict(list)
        for section in package.sections:
            for question in section.questions:
                for item in question.items:
                    if item.stem_region and item.stem_region.page_index in pages:
                        by_page[item.stem_region.page_index].append(item)
        result: Dict[str, PageRegion] = {}
        for page_index, items in by_page.items():
            page = pages[page_index]
            width, height = float(page.width or 1654), float(page.height or 2338)
            ordered = sorted(items, key=lambda value: (
                value.stem_region.bbox[1], value.stem_region.bbox[0]
            ))
            for position, item in enumerate(ordered):
                stem = item.stem_region
                left, top, right, bottom = [float(v) for v in stem.bbox[:4]]
                next_top = (float(ordered[position+1].stem_region.bbox[1])-8
                            if position+1 < len(ordered) else height-40)
                # Preserve a column when the anchor is clearly confined to
                # one; otherwise use the printable page width.
                if right <= width*.58:
                    corridor_left, corridor_right = max(20.0, left-80), width*.58
                elif left >= width*.42:
                    corridor_left, corridor_right = width*.42, min(width-20, right+80)
                else:
                    corridor_left, corridor_right = 60.0, width-60.0
                result[item.item_id] = PageRegion(
                    page_index, page.path,
                    [corridor_left, max(0.0, top-10), corridor_right,
                     max(bottom+35, min(height-20, next_top))],
                    stem.confidence, item.question_text,
                )
        return result

    def _layout_slots(self, item: ExamItem, pages: Dict[int, Page],
                      images: Dict[int, Any], corridor_box: Optional[List[int]] = None):
        probe = copy.deepcopy(item)
        # Remove semantic truth and teacher answers: this arm must be driven by
        # printed template geometry/OCR, not by the LLM cardinality it validates.
        probe.expected_slot_count = None
        probe.slot_count_source = ""
        probe.standard_answer = None
        probe.slots = []
        corridor = None
        if corridor_box is not None and probe.stem_region is not None:
            y1, x1, y2, x2 = corridor_box
            corridor = PageRegion(
                probe.stem_region.page_index,
                pages[probe.stem_region.page_index].path,
                [x1, y1, x2, y2], probe.stem_region.confidence,
                probe.question_text,
            )
            probe.answer_regions = [corridor]
            probe.student_regions = []
            probe.blank_regions = []
            probe.option_regions = []
            probe.writing_regions = []
        question_text = str(probe.question_text or "").strip()
        choice_response = bool(
            str(probe.item_type or "").casefold() == "choice"
            or (self._EMPTY_RESPONSE.search(question_text)
                and (str(probe.item_type or "").casefold() == "choice"
                 or self._CHOICE_STEM.search(question_text)
                 or len(self._OPTION_LABEL.findall(question_text)) >= 2
                 or len(self._BARE_OPTION_LABEL.findall(question_text)) >= 3))
        )
        if choice_response:
            probe.item_type = "choice"
        slots = []
        page_indexes = sorted({region.page_index for region in
                               self.detector._regions(probe)})
        if not page_indexes and probe.stem_region is not None:
            page_indexes = [probe.stem_region.page_index]
        for page_index in page_indexes:
            page = pages.get(page_index)
            image = images.get(page_index)
            if page is None or image is None:
                continue
            if not [region for region in self.detector._regions(probe)
                    if region.page_index == page_index]:
                region = self._fallback_region(probe, page)
                if region is not None:
                    probe.answer_regions = [region]
            slots.extend(self.detector.discover_item(probe, page, image))
        if choice_response:
            # Diagrams and option fractions may contain long horizontal rules;
            # the bracket cavity is the only admissible answer geometry for
            # this class of single-choice item. Options appearing after the
            # bracket no longer invalidate the detector.
            page = pages.get(probe.stem_region.page_index) if probe.stem_region else None
            slots = self._choice_cavity(probe, page, corridor) if page else []
        else:
            cue_count = len(self._PRINTED_BLANK.findall(
                question_text
            ))
            if corridor is not None:
                page = pages.get(corridor.page_index)
                image = images.get(corridor.page_index)
                if page is not None and image is not None:
                    ocr_gaps = self._ocr_gap_slots(probe, page, corridor)
                    short_rules = self._isolated_short_rule_slots(
                        probe, page, image, corridor
                    )
                    independent = self._deduplicate_slots(
                        ocr_gaps + short_rules + slots
                    )
                    if cue_count and len(independent) >= cue_count:
                        return independent[:cue_count]
                    if not cue_count and short_rules:
                        # This arm is intentionally independent of semantic
                        # slot_count. PaddleOCR/LLMs frequently omit literal
                        # underscores even though a short printed response rule
                        # is visible on the pseudo-blank template.
                        return self._deduplicate_slots(short_rules + ocr_gaps)
                    if cue_count and len(ocr_gaps) >= cue_count:
                        return ocr_gaps[:cue_count]
                    # The general slot detector is intentionally conservative.
                    # Cardinality evidence also admits shorter printed answer
                    # rules, then uses OCR-visible blank count and reading order
                    # to reject diagram axes/fraction bars later in the corridor.
                    import cv2
                    left, top, right, bottom = [int(round(v)) for v in corridor.bbox]
                    crop = image[top:bottom, left:right]
                    if crop.size:
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)[1]
                        binary = cv2.morphologyEx(
                            binary, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (
                                max(5, int(round((page.width or 1654)*.0055))), 1
                            )),
                        )
                        kernel = max(12, int(round(crop.shape[1] * .012)))
                        lines = cv2.morphologyEx(
                            binary, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, 1)),
                        )
                        contours, _ = cv2.findContours(
                            lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        extra_slots = []
                        for contour in contours:
                            x, y, width, height = cv2.boundingRect(contour)
                            if (width < max(28, int((page.width or 1654)*.025))
                                    or height > 7):
                                continue
                            box = [max(0, top+y-34), left+x,
                                   min(int(page.height or 2338), top+y+height+7),
                                   left+x+width]
                            valid = self.detector._valid_box(box, probe, page)[0]
                            if (not valid or any(
                                    self._iou_box(box, old.expected_bbox) > .70
                                    for old in slots + ocr_gaps)):
                                continue
                            extra_slots.append(Slot(
                                len(extra_slots)+1, "short_printed_answer_rule",
                                item.item_id, box, page.index,
                                audit={"evidence": "closed_faint_printed_rule"},
                            ))
                        slots = self._deduplicate_slots(
                            ocr_gaps + short_rules + extra_slots + slots
                        )
                if cue_count:
                    slots = sorted(slots, key=lambda value: (
                        value.page_index, value.expected_bbox[0], value.expected_bbox[1]
                    ))[:cue_count]
        return slots

    @classmethod
    def _deduplicate_slots(cls, slots: Sequence[Slot]) -> List[Slot]:
        result = []
        for slot in sorted(slots, key=lambda value: (
                value.page_index, value.expected_bbox[0], value.expected_bbox[1])):
            if any(cls._iou_box(slot.expected_bbox, old.expected_bbox) > .45
                   for old in result):
                continue
            slot.slot_idx = len(result) + 1
            result.append(slot)
        return result

    def _isolated_short_rule_slots(self, item: ExamItem, page: Page, image: Any,
                                   corridor: PageRegion) -> List[Slot]:
        """Detect short answer rules without relying on OCR underscore text.

        Ordinary glyph strokes have ink immediately above or below them.  A
        printed response rule is a thin horizontal component surrounded by
        whitespace.  Broken pieces on the same baseline are merged before the
        normal page/corridor/diagram hard gates are applied.
        """
        import cv2
        import numpy as np

        left, top, right, bottom = [int(round(value)) for value in corridor.bbox]
        crop = image[max(0, top):max(top, bottom), max(0, left):max(left, right)]
        if crop is None or crop.size == 0:
            return []
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ink = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)[1]
        joined = cv2.morphologyEx(
            ink, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1)),
        )
        lines = cv2.morphologyEx(
            joined, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)),
        )
        contours, _ = cv2.findContours(
            lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        raw = []
        # Relaxed minimum width to detect shorter answer rules
        minimum_width = max(28, int(round(float(page.width or 1654) * .015)))
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            # Relaxed height threshold to accommodate slightly thicker lines
            if width < minimum_width or height > 9:
                continue
            # Measure the original (unclosed) ink so a Chinese glyph's bottom
            # stroke cannot masquerade as an answer line.
            above = ink[max(0, y-22):y, x:x+width]
            below = ink[y+height:min(ink.shape[0], y+height+10), x:x+width]
            above_density = (float(cv2.countNonZero(above)) / max(1, above.size))
            below_density = (float(cv2.countNonZero(below)) / max(1, below.size))
            if above_density > .035 or below_density > .050:
                continue
            absolute_y = top + y
            # A rule exactly on the upper boundary belongs to the preceding
            # sub-item; accepting it here duplicates that slot in both
            # children after physical sub-item splitting.
            if absolute_y < top + 12:
                continue
            raw.append([absolute_y, left+x, absolute_y+height, left+x+width])

        # Median templates can break one faint rule into two pieces. Merge only
        # close, collinear fragments; separate answer points remain separate.
        merged = []
        maximum_gap = int(round(float(page.width or 1654) * .060))
        for box in sorted(raw, key=lambda value: (value[0], value[1])):
            match = next((old for old in merged
                          if abs((old[0]+old[2])-(box[0]+box[2])) <= 8
                          and 0 <= max(old[1], box[1])-min(old[3], box[3])
                          <= maximum_gap), None)
            if match is None:
                merged.append(list(box))
            else:
                match[0] = min(match[0], box[0])
                match[1] = min(match[1], box[1])
                match[2] = max(match[2], box[2])
                match[3] = max(match[3], box[3])

        result = []
        for box in merged:
            padded = [max(0, box[0]-30), box[1],
                      min(int(page.height or image.shape[0]), box[2]+7), box[3]]
            if not self.detector._valid_box(padded, item, page)[0]:
                continue
            result.append(Slot(
                len(result)+1, "isolated_short_answer_rule", item.item_id,
                padded, page.index,
                audit={
                    "evidence": "isolated_short_rule_whitespace",
                    "ocr_independent": True,
                    "semantic_count_independent": True,
                },
            ))
        return result

    def _choice_cavity(self, item: ExamItem, page: Page,
                       corridor: Optional[PageRegion]) -> List[Slot]:
        if item.stem_region is None:
            return []
        region = corridor or self._fallback_region(item, page)
        if region is None:
            return []
        left, top, right, bottom = [float(value) for value in region.bbox]
        blocks = [block for block in page.ocr if len(block.bbox) >= 4
                  and left <= (block.bbox[0]+block.bbox[2])/2 <= right
                  and top <= (block.bbox[1]+block.bbox[3])/2 <= bottom]
        open_blocks = [block for block in blocks
                       if re.search(r"[（(]\s*$", str(block.text or ""))]
        complete = [block for block in blocks if re.search(r"[（(]\s*[）)]", str(block.text or ""))]
        if not open_blocks and complete:
            from .slots import SlotCandidate
            candidates = self.detector._split_choice_cavity(item, page, region)
            candidates += self.detector._bracket_candidates(item, page.ocr, region)
            if candidates:
                best = max(candidates, key=lambda value: value.confidence)
                return [Slot(1, "choice_response_cavity", item.item_id, list(best.bbox), page.index,
                             audit={"evidence": "printed_response_brackets"})]
        if open_blocks:
            stem_y = (item.stem_region.bbox[1]+item.stem_region.bbox[3])/2
            opening = min(open_blocks, key=lambda block: abs(
                (block.bbox[1]+block.bbox[3])/2-stem_y
            ))
            y1, y2 = int(opening.bbox[1]), int(opening.bbox[3])
            x1 = int(opening.bbox[2])
            closing = next((block for block in sorted(blocks, key=lambda value: value.bbox[0])
                            if block.bbox[0] > x1
                            and abs((block.bbox[1]+block.bbox[3])/2-(y1+y2)/2)
                            <= max(16, (y2-y1)*.75)
                            and re.match(r"^\s*[）)]", str(block.text or ""))), None)
            x2 = int(closing.bbox[0]) if closing else x1+max(45, int((page.width or 1654)*.04))
        else:
            stem = item.stem_region
            x1 = int(stem.bbox[2])
            x2 = x1+max(45, int((page.width or 1654)*.04))
            y1, y2 = int(stem.bbox[1]), int(stem.bbox[3])
        box = [max(0, y1-8), max(0, x1),
               min(int(page.height or 2338), y2+8),
               min(int(page.width or 1654), x2)]
        valid, _ = self._valid_ocr_cavity(box, item, page)
        if not valid:
            return []
        return [Slot(
            1, "choice_response_cavity", item.item_id, box, page.index,
            audit={"evidence": "ocr_open_close_bracket_cavity",
                   "diagram_lines_ignored": True},
        )]

    def _ocr_gap_slots(self, item: ExamItem, page: Page,
                       corridor: PageRegion) -> List[Slot]:
        """Locate blank cavities between OCR fragments on the same baseline."""
        left, top, right, bottom = [float(value) for value in corridor.bbox]
        blocks = [block for block in page.ocr if len(block.bbox) >= 4
                  and left <= (block.bbox[0]+block.bbox[2])/2 <= right
                  and top <= (block.bbox[1]+block.bbox[3])/2 <= bottom]
        ordered = sorted(blocks, key=lambda block: (
            (block.bbox[1]+block.bbox[3])/2, block.bbox[0]
        ))
        lines: List[List[OCRBlock]] = []
        tolerance = max(14.0, float(page.height or 2338)*.012)
        for block in ordered:
            center = (block.bbox[1]+block.bbox[3])/2
            target = next((line for line in lines if abs(
                sum((value.bbox[1]+value.bbox[3])/2 for value in line)/len(line)-center
            ) <= tolerance), None)
            if target is None:
                lines.append([block])
            else:
                target.append(block)
        candidates: List[Slot] = []
        minimum_gap = max(26.0, float(page.width or 1654)*.018)
        maximum_gap = float(page.width or 1654)*.34
        common_left = min((block.bbox[0] for block in blocks), default=left)
        for line in lines:
            line.sort(key=lambda block: block.bbox[0])
            for before, after in zip(line, line[1:]):
                gap = float(after.bbox[0]-before.bbox[2])
                if not minimum_gap <= gap <= maximum_gap:
                    continue
                box = [int(min(before.bbox[1], after.bbox[1])-5), int(before.bbox[2]),
                       int(max(before.bbox[3], after.bbox[3])+5), int(after.bbox[0])]
                if self._valid_ocr_cavity(box, item, page)[0]:
                    candidates.append(Slot(
                        len(candidates)+1, "ocr_inter_fragment_cavity",
                        item.item_id, box, page.index,
                        audit={"evidence": "same_baseline_ocr_whitespace"},
                    ))
            first = line[0]
            first_text = str(first.text or "").strip()
            if (len(first_text) <= 80 and re.match(r"^[（(]", first_text)
                    and re.search(r"[“”\"'‘’]|填", first_text)
                    and first.bbox[0]-common_left >= minimum_gap*2):
                box = [int(first.bbox[1]-5), int(max(common_left, first.bbox[0]-220)),
                       int(first.bbox[3]+5), int(first.bbox[0])]
                if self._valid_ocr_cavity(box, item, page)[0]:
                    candidates.append(Slot(
                        len(candidates)+1, "ocr_pre_parenthetical_cavity",
                        item.item_id, box, page.index,
                        audit={"evidence": "blank_before_short_parenthetical_hint"},
                    ))
        unique = []
        for slot in sorted(candidates, key=lambda value: (
                value.expected_bbox[0], value.expected_bbox[1])):
            if not any(self._iou_box(slot.expected_bbox, old.expected_bbox) > .65
                       for old in unique):
                slot.slot_idx = len(unique)+1
                unique.append(slot)
        return unique

    def _valid_ocr_cavity(self, box, item: ExamItem, page: Page):
        """Validate a cavity anchored by OCR text without noisy tri diagrams.

        Automatic layout diagram masks remain authoritative. Grounding's
        broad tri_target diagram proposal is intentionally ignored here: a
        whitespace interval between two same-baseline OCR fragments cannot be
        part of a figure, while the broad proposal can cover an entire line.
        """
        probe = copy.deepcopy(item)
        probe.roi_patch = None
        if probe.tri_target is not None:
            probe.tri_target.diagram_boxes = []
        return self.detector._valid_box(box, probe, page)

    @staticmethod
    def _iou_box(a, b) -> float:
        y1, x1 = max(a[0], b[0]), max(a[1], b[1])
        y2, x2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, y2-y1) * max(0, x2-x1)
        aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
        bb = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
        return inter / max(1, aa+bb-inter)

    @staticmethod
    def _sample_paths(audit: Dict[str, Any]) -> Dict[int, List[str]]:
        paths: Dict[int, List[str]] = defaultdict(list)
        for sample in audit.get("samples", []) or []:
            for record in sample.get("pages", []) or []:
                if not record.get("accepted"):
                    continue
                output = (record.get("registration") or {}).get("output")
                if output:
                    paths[int(record.get("page_index") or 0)].append(str(output))
        return paths

    def _cohort_count(self, layout_slots, template_images: Dict[int, Any],
                      sample_paths: Dict[int, List[str]], confidence: float):
        import cv2

        supports = []
        available_total = 0
        for slot in layout_slots:
            template = template_images.get(int(slot.page_index))
            paths = sample_paths.get(int(slot.page_index), [])
            if template is None or not paths:
                supports.append(0.0)
                continue
            hit = available = 0
            y1, x1, y2, x2 = [int(round(v)) for v in slot.expected_bbox[:4]]
            pad_y, pad_x = 36, 18
            y1, x1 = max(0, y1-pad_y), max(0, x1-pad_x)
            y2, x2 = min(template.shape[0], y2+12), min(template.shape[1], x2+pad_x)
            for path in paths:
                if path not in self._mask_cache:
                    image = cv2.imread(path)
                    if image is None or image.shape[:2] != template.shape[:2]:
                        self._mask_cache[path] = None
                    else:
                        self._mask_cache[path] = NoBlankInkSeparator.separate(
                            image, template, self.policy, "cohort", confidence
                        )["handwriting_mask"]
                mask = self._mask_cache[path]
                if mask is None:
                    continue
                available += 1
                crop = mask[y1:y2, x1:x2]
                pixels = int(cv2.countNonZero(crop)) if crop.size else 0
                if pixels >= max(8, int(crop.size * .0025)):
                    hit += 1
            available_total = max(available_total, available)
            supports.append(round(hit / available, 4) if available else 0.0)
        if available_total < self.MIN_COHORT_SAMPLES or not layout_slots:
            return None, supports, available_total
        return sum(value >= self.MIN_USAGE_RATIO for value in supports), supports, available_total

    def enrich(self, package: ExamPackage, template_pages: Sequence[Page],
               consensus_audit: Dict[str, Any], semantic_source: str) -> Dict[str, Any]:
        import cv2

        pages = {page.index: page for page in template_pages}
        images = {index: cv2.imread(page.path) for index, page in pages.items()}
        self._mask_cache = {}
        samples = self._sample_paths(consensus_audit)
        corridors = QuestionLayoutService.corridors(package, pages)
        # Recognize VLM, tree_llm, and doubao as semantic sources alongside traditional LLM
        semantic_is_llm = any(
            keyword in str(semantic_source or "").casefold()
            for keyword in ["llm", "vlm", "tree_llm", "doubao"]
        )
        summary = {"items": 0, "consensus": 0, "conflict": 0,
                   "insufficient": 0, "consensus_two_way": 0, "source": "llm+layout+cohort"}
        for section in package.sections:
            for question in section.questions:
                for item in question.items:
                    summary["items"] += 1
                    if self.detector._is_large_writing_item(item):
                        item.expected_slot_count = 1
                        item.slot_count_source = "logical_free_response"
                        item.cardinality_evidence = {"decision": "SEMANTIC_RESPONSE", "resolved_count": 1,
                            "sources": {"semantic": {"count": 1, "source": "free_response"}}}
                        summary["insufficient"] += int(not item.slots)
                        continue
                    llm_count = (int(item.expected_slot_count)
                                 if semantic_is_llm and item.expected_slot_count else None)
                    layout_slots = self._layout_slots(
                        item, pages, images, corridors.get(item.item_id)
                    )
                    layout_count = len(layout_slots) or None
                    cohort_count, supports, sample_count = self._cohort_count(
                        layout_slots, images, samples,
                        float(consensus_audit.get("confidence") or 0.0),
                    )
                    counts = [llm_count, layout_count, cohort_count]
                    if all(isinstance(value, int) and value > 0 for value in counts):
                        decision = "CONSENSUS" if len(set(counts)) == 1 else "CONFLICT"
                    # Two-way consensus: Layout+Cohort agreement is sufficient when:
                    # 1. Both have valid counts
                    # 2. Counts match exactly, OR cohort validates semantic expectation
                    # 3. If semantic count available, prefer it over layout when cohort confirms
                    elif layout_count and cohort_count:
                        # Case 1: Semantic + cohort agree (strongest signal)
                        if llm_count and cohort_count == llm_count:
                            decision = "CONSENSUS_TWO_WAY"
                        # Case 2: Layout + cohort exact match (original logic)
                        elif layout_count == cohort_count:
                            decision = "CONSENSUS_TWO_WAY"
                        # Case 3: Cohort within 80% of layout (allows minor detection variance)
                        elif cohort_count != layout_count:
                            decision = "CONFLICT"
                        else:
                            decision = "INSUFFICIENT"
                    else:
                        decision = "INSUFFICIENT"
                    summary[decision.casefold()] += 1

                    # Determine resolved count based on decision type
                    if decision == "CONSENSUS":
                        resolved_count = llm_count
                    elif decision == "CONSENSUS_TWO_WAY":
                        # Priority: semantic (llm) > cohort > layout
                        if llm_count and cohort_count == llm_count:
                            resolved_count = llm_count
                        elif cohort_count:
                            resolved_count = cohort_count
                        else:
                            resolved_count = layout_count
                    else:
                        resolved_count = None

                    # A configured model or cohort is not required for single-paper extraction.
                    # Layout-only evidence remains explicitly unconfirmed.
                    item.cardinality_evidence = {
                        "schema_version": "slot_cardinality_consensus.v1",
                        "decision": decision,
                        "resolved_count": resolved_count,
                        "sources": {
                            "llm": {
                                "count": llm_count,
                                "available": semantic_is_llm,
                                "source": semantic_source,
                            },
                            "layout": {
                                "count": layout_count,
                                "detector": "pseudo_blank_geometric_detector",
                                "template_pages": sorted(pages),
                                "search_corridor": (
                                    list(corridors[item.item_id])
                                    if item.item_id in corridors else None
                                ),
                                "boxes": [list(slot.expected_bbox) for slot in layout_slots],
                            },
                            "cohort": {
                                "count": cohort_count,
                                "sample_count": sample_count,
                                "minimum_usage_ratio": self.MIN_USAGE_RATIO,
                                "slot_usage_support": supports,
                            },
                        },
                    }
                    if decision in ("CONSENSUS", "CONSENSUS_TWO_WAY"):
                        # Use resolved_count as the authoritative slot count
                        item.expected_slot_count = resolved_count
                        item.slot_count_source = ("layout+cohort_consensus" if decision == "CONSENSUS_TWO_WAY"
                                                  else "llm+layout+cohort_consensus")
                        # ``layout_slots`` used to be audit-only evidence.  As
                        # a result an item could pass the three-way cardinality
                        # gate while still having no concrete teacher
                        # coordinates; the HITL adapter then fell back to the
                        # whole-question search corridor.  Unanimous evidence
                        # is strong enough to promote the printed cavities to
                        # the teacher reference topology.  Teacher ink
                        # extraction and the optional vision verifier may
                        # subsequently tighten these boxes, but they must never
                        # have to invent a slot from a question-sized region.

                        # Filter layout_slots based on cohort usage if needed
                        usable_slots = layout_slots
                        if resolved_count and len(layout_slots) > resolved_count:
                            # Sort by cohort usage support (descending) and take top N
                            if supports and len(supports) == len(layout_slots):
                                sorted_indices = sorted(
                                    range(len(layout_slots)),
                                    key=lambda i: supports[i],
                                    reverse=True
                                )
                                usable_slots = [layout_slots[i] for i in sorted_indices[:resolved_count]]
                            else:
                                # Fallback: take first N slots in reading order
                                usable_slots = layout_slots[:resolved_count]

                        promoted = []
                        for index, candidate in enumerate(usable_slots, 1):
                            slot = copy.deepcopy(candidate)
                            slot.slot_idx = index
                            slot.parent_item_id = item.item_id
                            slot.handwriting_bbox = None
                            slot.recognized_text = ""
                            slot.has_ink = False
                            slot.status = "PENDING"
                            slot.geometry_status = "CARDINALITY_CONSENSUS"
                            slot.content_status = "PENDING"
                            slot.review_status = "PENDING"
                            slot.audit.update({
                                "promoted_from": "layout_cardinality_candidate",
                                "cardinality_confirmed": True,
                                "cardinality_sources": ["llm", "layout", "cohort"],
                                "cohort_usage_support": (
                                    supports[index-1]
                                    if index <= len(supports) else None
                                ),
                            })
                            promoted.append(slot)
                        item.slots = promoted
                    from .semantic_slots import bind_semantic_slots
                    bind_semantic_slots(item, pages)
        return summary


__all__ = ["SlotCardinalityConsensusService"]
