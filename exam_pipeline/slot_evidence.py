"""OCR coordinate grounding for VLM semantic slot proposals.

The visual model decides which logical answer slots exist and supplies rough
search hints. Final public coordinates always come from OCR boxes on the
current document. No ink mask, print/handwriting classifier, connected
component detector, template difference, or stroke-boundary rule participates
in this stage.
"""
from __future__ import annotations

import re
import hashlib
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median

from .result_contract import valid_box


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersection(left, right):
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1]))


def normalized(text):
    text = unicodedata.normalize("NFKC", str(text or ""))
    return re.sub(r"[\W_]+", "", text).casefold()


def printed_text(text, item):
    value = normalized(text)
    question = normalized(item.question_text)
    return bool(value and len(value) >= 3 and (
        value in question or SequenceMatcher(None, value, question).ratio() > .82
    ))


def resembles_printed_text(text, references):
    """Return whether an OCR block belongs to any known printed prompt."""
    value = normalized(text)
    if len(value) < 3:
        return False
    return any(
        value in reference or SequenceMatcher(None, value, reference).ratio() > .82
        for reference in (normalized(candidate) for candidate in references)
        if reference
    )


def printed_option_line(text):
    """Detect an OCR option row without classifying print or handwriting."""
    value = unicodedata.normalize("NFKC", str(text or ""))
    return bool(re.match(r"^\s*[A-Ha-h]\s*[.、．:：)）]", value))


def _center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _union(boxes):
    return [min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes)]


def _axis_overlap(left, right, axis):
    start = max(left[axis], right[axis])
    end = min(left[axis + 2], right[axis + 2])
    return max(0, end - start)


def _cluster_long_answer_candidates(candidates, proposal, line_height):
    """Group OCR boxes into continuous writing blocks and rank the blocks."""
    if not candidates:
        return [], []
    remaining = set(range(len(candidates)))
    groups = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            left = candidates[current]['bbox']
            left_width = max(1, left[2] - left[0])
            left_height = max(1, left[3] - left[1])
            for other in list(remaining):
                right = candidates[other]['bbox']
                right_width = max(1, right[2] - right[0])
                right_height = max(1, right[3] - right[1])
                horizontal_gap = max(0, max(left[0], right[0]) - min(left[2], right[2]))
                vertical_gap = max(0, max(left[1], right[1]) - min(left[3], right[3]))
                x_overlap = _axis_overlap(left, right, 0) / max(1, min(left_width, right_width))
                y_overlap = _axis_overlap(left, right, 1) / max(1, min(left_height, right_height))
                same_line = y_overlap >= .25 and horizontal_gap <= line_height * 3.0
                adjacent_line = (x_overlap >= .12 and vertical_gap <= line_height * 2.4)
                aligned_working = (vertical_gap <= line_height * 1.5
                                   and abs(_center(left)[0] - _center(right)[0])
                                   <= max(left_width, right_width) * 1.2)
                if same_line or adjacent_line or aligned_working:
                    remaining.remove(other)
                    component.add(other)
                    frontier.append(other)
        groups.append(sorted(component))

    ranked = []
    px, py = _center(proposal)
    proposal_diagonal = max(1.0, ((proposal[2] - proposal[0]) ** 2
                                  + (proposal[3] - proposal[1]) ** 2) ** .5)
    for indexes in groups:
        members = [candidates[index] for index in indexes]
        box = _union([member['bbox'] for member in members])
        member_area = sum(area(member['bbox']) for member in members)
        density = min(1.0, member_area / max(1, area(box)) * 3.0)
        overlap = intersection(box, proposal) / max(1, area(box))
        cx, cy = _center(box)
        distance = ((cx - px) ** 2 + (cy - py) ** 2) ** .5 / proposal_diagonal
        score = (.38 * max(member['score'] for member in members)
                 + .24 * sum(member['score'] for member in members) / len(members)
                 + .18 * min(1.0, overlap)
                 + .12 * density
                 + .08 * max(0.0, 1.0 - distance))
        ranked.append({'bbox': box, 'member_indexes': indexes,
                       'member_boxes': [member['bbox'] for member in members],
                       'score': round(score, 4), 'proposal_overlap': round(overlap, 4),
                       'density': round(density, 4)})
    ranked.sort(key=lambda value: (-value['score'], value['bbox'][1], value['bbox'][0]))
    selected = [candidates[index] for index in ranked[0]['member_indexes']]
    selected.sort(key=lambda value: (value['bbox'][1], value['bbox'][0]))
    return selected, ranked


def _current_fingerprint(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


class LocalSlotEvidenceValidator:
    """Snap one VLM search hint to current-page OCR coordinates.

    Printed-text similarity is a ranking signal rather than a veto. This
    avoids dropping a correct short response merely because OCR merged it with
    a nearby prompt. Page identity, finite coordinates, page bounds and
    cross-slot ownership remain hard requirements.
    """

    def __init__(self, policy=None, ocr_service=None, ocr_engine='paddle',
                 language='chi_sim+eng', local_ocr_scale=2.5):
        self.policy = policy
        self.ocr_service = ocr_service
        self.ocr_engine = ocr_engine
        self.language = language
        self.local_ocr_scale = local_ocr_scale

    def validate(self, item, page, image, proposal, domain, reference=None,
                 expanded=False, foreign_texts=(), printed_exclusion_regions=()):
        del image, reference
        current_fingerprint = _current_fingerprint(page.path) if page.file_fingerprint else ""
        if (page.file_fingerprint and current_fingerprint
                and current_fingerprint != page.file_fingerprint):
            return {"status": "UNRESOLVED", "reason": "PAGE_FINGERPRINT_MISMATCH",
                    "search_bbox": list(proposal), "candidates": [],
                    "coordinate_authority": "ocr_boxes_only"}
        if (page.ocr_source_fingerprint and page.file_fingerprint
                and page.ocr_source_fingerprint != page.file_fingerprint):
            return {"status": "UNRESOLVED", "reason": "OCR_SOURCE_IMAGE_MISMATCH",
                    "search_bbox": list(proposal), "candidates": [],
                    "coordinate_authority": "ocr_boxes_only"}
        heights = [float(block.bbox[3] - block.bbox[1]) for block in page.ocr
                   if valid_box(block.bbox, page)
                   and intersection(block.bbox, domain) > 0]
        text_height = median(heights) if heights else 24.0
        padding = text_height * (2.5 if expanded else .75)
        search = [max(domain[0], int(proposal[0] - padding)),
                  max(domain[1], int(proposal[1] - padding)),
                  min(domain[2], int(proposal[2] + padding)),
                  min(domain[3], int(proposal[3] + padding))]
        if not valid_box(search, page):
            return {"status": "UNRESOLVED", "reason": "INVALID_SEARCH_REGION",
                    "search_bbox": search, "candidates": []}

        px, py = _center(proposal)
        diagonal = max(1.0, ((proposal[2] - proposal[0]) ** 2
                             + (proposal[3] - proposal[1]) ** 2) ** .5)
        proposal_ratio = area(proposal) / max(1, area(domain))
        broad_hint = proposal_ratio > .62
        kind = str(item.item_type or "").casefold()
        multi_line = any(token in kind for token in (
            "solve", "writing", "proof", "essay", "calculation",
            "解答", "证明", "作文", "计算"))
        stem = (list(item.stem_region.bbox)
                if item.stem_region and item.stem_region.page_index == page.index
                and valid_box(item.stem_region.bbox, page) else None)
        exclusions = [list(box) for box in (printed_exclusion_regions or ())
                      if valid_box(box, page)]
        stem_height = max(1.0, stem[3] - stem[1]) if stem else text_height
        block_sources = [{'block': block, 'source': 'ocr', 'original_bbox': None}
                         for block in page.ocr]
        local_ocr_attempts = []
        if exclusions:
            for block in page.ocr:
                box = [int(round(value)) for value in block.bbox]
                if not valid_box(box, page):
                    continue
                exclusion_pixels = max((intersection(box, exclusion)
                                        for exclusion in exclusions), default=0)
                domain_pixels = intersection(box, domain)
                overlap = exclusion_pixels / max(1, area(box))
                # A box spanning both a known printed region and answer domain
                # is contaminated geometry. Re-detect only this bounded crop.
                if overlap < .05 or overlap >= .95 or domain_pixels / max(1, area(box)) < .15:
                    continue
                record = {'reason': 'OCR_BOX_SPANS_PRINTED_AND_ANSWER_REGIONS',
                          'original_bbox': box, 'scale': self.local_ocr_scale,
                          'status': 'FAILED', 'boxes': []}
                try:
                    service = self.ocr_service
                    if service is None:
                        from .ocr import OCRService
                        service = OCRService()
                    method = getattr(service, 'recognize_crop_high_resolution', None)
                    refined = (method(page.path, box, padding=max(4, int(text_height * .3)),
                                      scale=self.local_ocr_scale,
                                      engine=self.ocr_engine, language=self.language)
                               if method else service.recognize_crop(
                                   page.path, box, padding=max(4, int(text_height * .3)),
                                   engine=self.ocr_engine, language=self.language))
                    for detected in refined:
                        if valid_box(detected.bbox, page):
                            block_sources.append({'block': detected,
                                                  'source': 'local_high_resolution_ocr',
                                                  'original_bbox': box})
                            record['boxes'].append([int(round(value))
                                                    for value in detected.bbox])
                    record['status'] = 'REDETECTED' if record['boxes'] else 'NO_BOXES'
                except Exception as exc:
                    record['error_type'] = type(exc).__name__
                local_ocr_attempts.append(record)
        candidates = []
        refined_originals = {tuple(entry['original_bbox']) for entry in block_sources
                             if entry['original_bbox'] is not None}
        for source_record in block_sources:
            block = source_record['block']
            box = [int(round(value)) for value in block.bbox]
            if not valid_box(box, page):
                continue
            overlap_search = intersection(box, search) / max(1, area(box))
            overlap_proposal = intersection(box, proposal) / max(1, area(box))
            overlap_domain = intersection(box, domain) / max(1, area(box))
            overlap_stem = (intersection(box, stem) / max(1, area(box))
                            if stem else 0.0)
            exclusion_overlap = max((intersection(box, exclusion) / max(1, area(box))
                                     for exclusion in exclusions), default=0.0)
            cx, cy = _center(box)
            center_distance = ((cx - px) ** 2 + (cy - py) ** 2) ** .5 / diagonal
            center_inside = (proposal[0] <= cx <= proposal[2]
                             and proposal[1] <= cy <= proposal[3])
            center_in_domain = (domain[0] <= cx <= domain[2]
                                and domain[1] <= cy <= domain[3])
            if (overlap_search < .15 or overlap_domain < .40 or not center_in_domain
                    or (not center_inside and overlap_proposal < .08)):
                continue
            is_printed = printed_text(block.text, item)
            is_foreign_printed = resembles_printed_text(block.text, foreign_texts)
            is_option = printed_option_line(block.text)
            is_marker = bool(re.fullmatch(
                r"\s*(?:第\s*\d+\s*[题问]?|\d+\s*[.、。．:：)）]|[A-Ha-h]\s*[.、．)])\s*",
                block.text or ""))
            before_anchor = bool(
                stem and not multi_line and cy < (stem[1] + stem[3]) / 2.0 - .60 * stem_height
            )
            long_partial_line = bool(
                (box[2] - box[0]) > .70 * max(1, domain[2] - domain[0])
                and overlap_proposal < .35
            )
            printed_stem_overlap = bool(
                not multi_line and overlap_stem >= .60
                and (is_printed or (box[2] - box[0]) > .65 * max(1, domain[2] - domain[0]))
            )
            printed_exclusion_overlap = exclusion_overlap >= .15
            merged_original = (source_record['source'] == 'ocr'
                               and tuple(box) in refined_originals)
            rejection_reasons = []
            if is_marker:
                rejection_reasons.append("STRUCTURAL_MARKER")
            if is_foreign_printed:
                rejection_reasons.append("OTHER_ITEM_PRINTED_TEXT")
            if is_option and "choice" not in kind:
                rejection_reasons.append("PRINTED_OPTION_FOR_OTHER_ITEM")
            if before_anchor:
                rejection_reasons.append("BEFORE_ITEM_READING_ANCHOR")
            if long_partial_line:
                rejection_reasons.append("LONG_LINE_PARTIAL_INTERSECTION")
            if printed_stem_overlap:
                rejection_reasons.append("PRINTED_STEM_OVERLAP")
            if printed_exclusion_overlap:
                rejection_reasons.append("PRINTED_EXCLUSION_OVERLAP")
            if merged_original:
                rejection_reasons.append("MERGED_PRINTED_AND_ANSWER_OCR_BOX")
            confidence = float(block.confidence) if block.confidence is not None else .55
            geometry_weight = .18 if broad_hint else .50
            rightward = max(0.0, min(1.0, (cx - domain[0]) / max(1, domain[2] - domain[0])))
            score = (geometry_weight * min(1.0, overlap_proposal)
                     + .22 * max(0.0, 1.0 - center_distance)
                     + .18 * max(0.0, min(1.0, confidence))
                     + (.10 if center_inside else 0.0)
                     + (.10 * rightward if not multi_line else 0.0)
                     - (.22 if is_printed else 0.0)
                     - (.28 if is_foreign_printed else 0.0)
                     - (.24 if is_option else 0.0)
                     - (.30 if is_marker else 0.0)
                     - (.24 if exclusion_overlap >= .72 else 0.0))
            candidates.append({
                "bbox": box,
                "text": str(block.text or ""),
                "source": "ocr",
                "coordinate_source": source_record['source'],
                "original_merged_bbox": source_record['original_bbox'],
                "confidence": block.confidence,
                "score": round(score, 4),
                "proposal_overlap": round(overlap_proposal, 4),
                "domain_overlap": round(overlap_domain, 4),
                "stem_overlap": round(overlap_stem, 4),
                "printed_exclusion_overlap": round(exclusion_overlap, 4),
                "center_distance": round(center_distance, 4),
                "printed_similarity": is_printed,
                "foreign_printed_similarity": is_foreign_printed,
                "printed_option_line": is_option,
                "structural_marker": is_marker,
                "broad_search_hint": broad_hint,
                "eligible": not rejection_reasons,
                "rejection_reasons": rejection_reasons,
            })

        candidates.sort(key=lambda value: (-value["score"], value["bbox"][1],
                                           value["bbox"][0]))
        usable = [value for value in candidates if value["eligible"]]
        if not usable:
            return {"status": "UNRESOLVED",
                    "reason": "NO_OCR_COORDINATE_EVIDENCE",
                    "search_bbox": search, "candidates": candidates,
                    "local_ocr_redetection": local_ocr_attempts,
                    "coordinate_authority": "ocr_boxes_only"}

        best = usable[0]
        selected = [best]
        clusters = []
        if multi_line:
            selected, clusters = _cluster_long_answer_candidates(
                usable, proposal, text_height)

        box = _union([value["bbox"] for value in selected])
        warnings = []
        if any(value["printed_similarity"] for value in selected):
            warnings.append("OCR_BOX_MAY_INCLUDE_PRINTED_CONTEXT")
        if best["score"] < .35:
            warnings.append("LOW_OCR_COORDINATE_SCORE")
        if broad_hint:
            warnings.append("BROAD_VLM_SEARCH_HINT_GEOMETRY_DOWNWEIGHTED")
        return {
            "status": "VERIFIED",
            "bbox": box,
            "support": "ocr_coordinates",
            "search_bbox": search,
            "candidates": candidates,
            "alternatives": [{"bbox": value["bbox"], "source": "ocr",
                              "score": value["score"]} for value in usable],
            "selected_boxes": [value["bbox"] for value in selected],
            "selected_text": [value["text"] for value in selected],
            "spatial_clusters": clusters,
            "local_ocr_redetection": local_ocr_attempts,
            "warnings": warnings,
            "coordinate_authority": "ocr_boxes_only",
        }


def assign_unique_candidates(records):
    """Maximum-weight one-to-one OCR assignment, grouped by physical page.

    Each record contains ``page_index`` and a candidate list produced by
    :class:`LocalSlotEvidenceValidator`. Dummy columns let a slot remain
    unresolved when no unique positive candidate exists.
    """
    result = {}
    pages = {}
    for position, record in enumerate(records):
        pages.setdefault(record["page_index"], []).append((position, record))
    for page_records in pages.values():
        keys = []
        by_key = {}
        for _, record in page_records:
            for candidate in record.get("candidates", []):
                if not candidate.get("eligible", not candidate.get("structural_marker")):
                    continue
                key = tuple(int(round(value)) for value in candidate["bbox"])
                if key not in by_key:
                    keys.append(key)
                    by_key[key] = len(keys) - 1
        row_count = len(page_records)
        column_count = len(keys) + row_count
        if not row_count:
            continue
        unavailable = 1_000_000.0
        costs = [[unavailable] * column_count for _ in range(row_count)]
        lookup = []
        for row, (_, record) in enumerate(page_records):
            row_lookup = {}
            for candidate in record.get("candidates", []):
                if not candidate.get("eligible", not candidate.get("structural_marker")):
                    continue
                score = float(candidate.get("score") or 0.0)
                if score <= 0:
                    continue
                key = tuple(int(round(value)) for value in candidate["bbox"])
                column = by_key[key]
                if -score < costs[row][column]:
                    costs[row][column] = -score
                    row_lookup[column] = candidate
            for dummy in range(len(keys), column_count):
                costs[row][dummy] = 0.0
            lookup.append(row_lookup)

        # Rectangular Hungarian algorithm (rows <= columns).
        u = [0.0] * (row_count + 1)
        v = [0.0] * (column_count + 1)
        p = [0] * (column_count + 1)
        way = [0] * (column_count + 1)
        for source_row in range(1, row_count + 1):
            p[0] = source_row
            minv = [float("inf")] * (column_count + 1)
            used = [False] * (column_count + 1)
            column0 = 0
            while True:
                used[column0] = True
                row0 = p[column0]
                delta = float("inf")
                column1 = 0
                for column in range(1, column_count + 1):
                    if used[column]:
                        continue
                    current = costs[row0 - 1][column - 1] - u[row0] - v[column]
                    if current < minv[column]:
                        minv[column] = current
                        way[column] = column0
                    if minv[column] < delta:
                        delta = minv[column]
                        column1 = column
                for column in range(column_count + 1):
                    if used[column]:
                        u[p[column]] += delta
                        v[column] -= delta
                    else:
                        minv[column] -= delta
                column0 = column1
                if p[column0] == 0:
                    break
            while True:
                column1 = way[column0]
                p[column0] = p[column1]
                column0 = column1
                if column0 == 0:
                    break
        assignment = [None] * row_count
        for column in range(1, column_count + 1):
            if p[column]:
                assignment[p[column] - 1] = column - 1
        for row, (position, _) in enumerate(page_records):
            column = assignment[row]
            result[position] = lookup[row].get(column) if column is not None else None
    return result


__all__ = ["LocalSlotEvidenceValidator", "area", "intersection",
           "normalized", "printed_text", "resembles_printed_text",
           "printed_option_line", "assign_unique_candidates"]
