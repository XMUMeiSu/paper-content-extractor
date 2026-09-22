"""Evidence-grounded question contexts, independent of answer text and teacher boxes."""
import re
import unicodedata
from difflib import SequenceMatcher
from statistics import median
from .contracts import PageRegion
from .reading_order import column_groups
from .structure_recovery import anchors


LONG_FORM_TOKENS = (
    'large_writing', 'solve', 'writing', 'proof', 'essay', 'calculation',
    '解答', '证明', '作文', '计算',
)


def is_long_form_item(item):
    """Return whether an item needs a below-stem answer corridor.

    The explicit ``large_writing`` type is included because it is the
    canonical type emitted by the structure tree for Chinese free-response
    subquestions.  Keeping this in one helper avoids the old drift where
    different stages treated that type as a short answer.
    """
    value = str(getattr(item, 'item_type', '') or '').casefold()
    return any(token in value for token in LONG_FORM_TOKENS)


def _valid_xyxy(box, page):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    except (TypeError, ValueError):
        return False
    width, height = page.width or 1654, page.height or 2338
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def answer_domain_for_context(item, context, page, stem_box=None):
    """Derive a deterministic answer search domain from a question context.

    ``context['bbox']`` identifies the printed question.  ``limits`` is the
    wider same-column reading corridor.  Long-form answers start after the
    printed context and may continue to the corridor boundary; short answers
    stay near the question row and receive a small OCR-height margin.
    """
    context_box = list(context.get('bbox') or [])
    if not _valid_xyxy(context_box, page):
        return None
    width, height = page.width or 1654, page.height or 2338
    limits = list(context.get('limits') or context_box)
    if not _valid_xyxy(limits, page):
        limits = context_box
    x1, y1, x2, y2 = [int(round(float(value))) for value in context_box]
    lx1, ly1, lx2, ly2 = [int(round(float(value))) for value in limits]
    # ``context_box`` may extend through the whole same-question corridor.
    # Once OCR has matched the printed stem, the answer corridor must begin
    # immediately below that measured stem, otherwise the context's bottom
    # would hide the student's answer area.
    anchor_box = context.get('anchor_bbox')
    stem_bottom = (int(round(float(stem_box[3])))
                   if _valid_xyxy(stem_box, page)
                   else int(round(float(anchor_box[3])))
                   if _valid_xyxy(anchor_box, page) else y2)

    if is_long_form_item(item):
        # Keep the question's full printed width but begin below all printed
        # context.  This is the independent answer area for free responses.
        return [max(0, lx1), min(height - 1, stem_bottom),
                min(width, lx2), min(height, max(stem_bottom + 1, ly2))]

    # Short answers usually share the question row.  A modest OCR-derived
    # margin captures a handwritten glyph that sits just above/below the
    # printed baseline without opening the whole next-question corridor.
    heights = []
    for block in page.ocr:
        if _valid_xyxy(block.bbox, page):
            bx1, by1, bx2, by2 = block.bbox
            if max(0, min(bx2, x2) - max(bx1, x1)) > 0 and max(0, min(by2, y2) - max(by1, y1)) > 0:
                heights.append(max(1, by2 - by1))
    line_height = sorted(heights)[len(heights) // 2] if heights else 24
    margin = max(18, min(72, int(round(line_height * 1.5))))
    return [max(lx1, x1 - margin), max(ly1, y1 - margin),
            min(lx2, x2 + margin), min(ly2, y2 + margin)]


def _printed_exclusion_regions(item, question, page, blocks, context_box, stem_box):
    """Collect printed OCR boxes usable as hard candidate exclusions.

    This uses the teacher/ExamTree text references and structural markers. It
    deliberately does not classify pixels as handwriting, so the no-ink-mask
    production path remains deterministic.
    """
    references = [item.question_text, question.question_title]
    references.extend(item.quality.get('printed_references') or [])
    refs = [normalized(value) for value in references if normalized(value)]
    result = []
    if _valid_xyxy(stem_box, page):
        result.append(list(stem_box))
    for block in blocks:
        if not _valid_xyxy(block.bbox, page):
            continue
        value = normalized(block.text)
        structural = bool(re.fullmatch(
            r'\s*(?:第\s*\d+\s*[题问]?|[（(]\s*\d{1,2}\s*[）)]|\d+\s*[.、．:：)）])\s*',
            str(block.text or '')))
        similar = bool(value and any(value in ref or SequenceMatcher(None, value, ref).ratio() >= .78
                                    for ref in refs if len(ref) >= 3))
        if similar or structural:
            result.append([int(round(v)) for v in block.bbox])
    # Deduplicate while preserving reading order.
    unique = []
    seen = set()
    for box in sorted(result, key=lambda b: (b[1], b[0])):
        key = tuple(box)
        if key not in seen:
            seen.add(key); unique.append(box)
    return unique


def normalized(text):
    text = unicodedata.normalize('NFKC', text or '')
    text = re.sub(r'\\(?:frac|sqrt|left|right|mathrm|text|square)', '', text)
    return re.sub(r'[\W_]+', '', text).lower()


def _structural_label(text):
    value = str(text or '')
    match = re.match(r'^\s*[（(\[]\s*(\d{1,2})\s*[）)\]]', value)
    if match:
        return ('sub', int(match.group(1)))
    match = re.match(r'^\s*(?:第\s*)?(\d{1,3})\s*[.、．:：)）]?', value)
    if match:
        return ('question', int(match.group(1)))
    return None


def _item_sub_number(item):
    """Return the printed subquestion number carried by an ExamItem.

    The tree normally stores ``(1)`` in ``item_name``.  Some providers put the
    marker at the beginning of ``question_text`` instead, so both fields are
    checked before falling back to the item id suffix.
    """
    for value in (getattr(item, 'item_name', ''), getattr(item, 'question_text', '')):
        marker = _structural_label(value)
        if marker and marker[0] == 'sub':
            return marker[1]
    match = re.search(r'[_:\-](\d{1,2})$', str(getattr(item, 'item_id', '') or ''))
    return int(match.group(1)) if match else None


def _subitem_slice(blocks, item, sibling_items):
    """Find the OCR block interval for one subitem when printed markers exist.

    This is deliberately block based.  It never infers handwriting or ink
    boundaries; when a scanner merges several markers into one OCR block the
    whole block is retained and the audit records that the domain is shared.
    """
    target = _item_sub_number(item)
    if target is None:
        return blocks, {'subitem': None, 'split': False, 'reason': 'NO_SUBITEM_MARKER'}
    marker_positions = []
    for position, block in enumerate(blocks):
        label = _structural_label(block.text)
        if label and label[0] == 'sub':
            marker_positions.append((label[1], position))
    # A marker can be embedded in a long OCR line.  Use the known sibling
    # order to keep that block with the first item rather than invent pixels.
    target_positions = [position for number, position in marker_positions if number == target]
    if not target_positions:
        return blocks, {'subitem': target, 'split': False, 'reason': 'SUBITEM_MARKER_NOT_GROUNDED'}
    start = target_positions[0]
    later = [position for number, position in marker_positions if position > start]
    end = min(later) if later else len(blocks)
    selected = blocks[start:end] or blocks[start:start + 1]
    return selected, {'subitem': target, 'split': bool(later or start),
                      'reason': 'OCR_SUBITEM_MARKERS'}


def _context_anchor(item, context, page):
    """Return the strongest current-page anchor for sibling partitioning."""
    if (item.stem_region and item.stem_region.page_index == page.index
            and _valid_xyxy(item.stem_region.bbox, page)):
        return [int(round(value)) for value in item.stem_region.bbox], 'stem_region'
    for key in ('anchor_bbox', 'question_context', 'bbox'):
        box = context.get(key)
        if _valid_xyxy(box, page):
            return [int(round(value)) for value in box], key
    return None, None


def _proposal_anchor(context, page, hint):
    regions = [region for region in (hint or {}).get('regions', [])
               if region.get('page_index') == page.index
               and isinstance(region.get('search_bbox'), list)
               and len(region['search_bbox']) == 4]
    if not regions:
        return None
    boxes = []
    coordinate_space = (hint or {}).get('coordinate_space', 'full_page_normalized')
    base = context.get('domain_before_sibling_partition',
                       context.get('answer_search_domain'))
    for region in regions:
        values = region['search_bbox']
        try:
            values = [float(value) for value in values]
        except (TypeError, ValueError):
            continue
        if not 0 <= values[0] < values[2] <= 1000 or not 0 <= values[1] < values[3] <= 1000:
            continue
        if coordinate_space == 'context_crop_normalized' and _valid_xyxy(base, page):
            x1, y1, x2, y2 = base
            box = [x1 + values[0] * (x2 - x1) / 1000,
                   y1 + values[1] * (y2 - y1) / 1000,
                   x1 + values[2] * (x2 - x1) / 1000,
                   y1 + values[3] * (y2 - y1) / 1000]
        else:
            box = [values[0] * page.width / 1000,
                   values[1] * page.height / 1000,
                   values[2] * page.width / 1000,
                   values[3] * page.height / 1000]
        if _valid_xyxy(box, page):
            boxes.append(box)
    if not boxes:
        return None
    return [int(round(min(box[0] for box in boxes))),
            int(round(min(box[1] for box in boxes))),
            int(round(max(box[2] for box in boxes))),
            int(round(max(box[3] for box in boxes)))]


def _layout_axis(records, hints, page):
    votes = []
    for item, _, _ in records:
        hint = (hints or {}).get(item.item_id) or {}
        axis = hint.get('axis')
        confidence = hint.get('confidence', 0)
        if axis in {'horizontal', 'vertical'} and isinstance(confidence, (int, float)):
            votes.append((float(confidence), axis))
    if votes:
        totals = {axis: sum(weight for weight, value in votes if value == axis)
                  for axis in ('horizontal', 'vertical')}
        if totals['horizontal'] != totals['vertical']:
            return max(totals, key=totals.get), 'vlm_layout_vote'

    centers = [((anchor[0] + anchor[2]) / 2.0,
                (anchor[1] + anchor[3]) / 2.0) for _, _, anchor in records]
    x_span = (max(point[0] for point in centers) - min(point[0] for point in centers)) / max(1, page.width)
    y_span = (max(point[1] for point in centers) - min(point[1] for point in centers)) / max(1, page.height)
    return ('horizontal', 'anchor_geometry') if x_span > y_span * 1.25 else ('vertical', 'reading_order')


def partition_sibling_answer_domains(question, pages, layout_hints=None):
    """Partition same-question leaf items into page-local, disjoint answer bands.

    OCR anchors provide the hard pixel boundaries. A VLM layout declaration can
    choose the reading axis and its rough regions are retained as audit evidence,
    but they never become final coordinates.
    """
    page_map = pages if isinstance(pages, dict) else {page.index: page for page in pages}
    by_page = {}
    for item in question.items:
        for context in item.quality.get('localization', {}).get('contexts', []):
            page = page_map.get(context.get('page_index'))
            if page is None or not _valid_xyxy(context.get('answer_search_domain'), page):
                continue
            anchor, anchor_source = _context_anchor(item, context, page)
            if anchor is None:
                continue
            proposal_anchor = _proposal_anchor(
                context, page, (layout_hints or {}).get(item.item_id) or {})
            by_page.setdefault(page.index, []).append(
                (item, context, anchor, anchor_source, proposal_anchor))

    updates = 0
    for page_index, page_records in by_page.items():
        distinct = {item.item_id for item, _, _, _, _ in page_records}
        if len(distinct) < 2:
            continue
        page = page_map[page_index]
        anchor_keys = [tuple(anchor) for _, _, anchor, _, _ in page_records]
        duplicate_anchors = len(set(anchor_keys)) < len(anchor_keys)
        resolved = []
        for item, context, anchor, anchor_source, proposal_anchor in page_records:
            if proposal_anchor is not None and (duplicate_anchors or anchor_source != 'stem_region'):
                resolved.append((item, context, proposal_anchor, 'vlm_proposal_order_hint'))
            else:
                resolved.append((item, context, anchor, anchor_source))
        records = [(item, context, anchor) for item, context, anchor, _ in resolved]
        axis, axis_source = _layout_axis(records, layout_hints or {}, page)
        coordinate = 0 if axis == 'horizontal' else 1
        ordered = sorted(resolved, key=lambda value: (
            (value[2][coordinate] + value[2][coordinate + 2]) / 2.0,
            value[2][1], value[2][0], value[0].item_id))
        group_id = '{}:page:{}'.format(question.question_id, page_index)

        for position, (item, context, anchor, anchor_source) in enumerate(ordered):
            original = [int(round(value)) for value in context.get(
                'domain_before_sibling_partition', context['answer_search_domain'])]
            band = list(original)
            previous_anchor = ordered[position - 1][2] if position else None
            next_anchor = ordered[position + 1][2] if position + 1 < len(ordered) else None
            boundaries = {}
            if axis == 'vertical':
                band[1] = max(band[1], anchor[3])
                if previous_anchor:
                    boundaries['previous_anchor_bottom'] = previous_anchor[3]
                if next_anchor:
                    band[3] = min(band[3], next_anchor[1])
                    boundaries['next_anchor_top'] = next_anchor[1]
            else:
                center = (anchor[0] + anchor[2]) / 2.0
                if previous_anchor:
                    previous_center = (previous_anchor[0] + previous_anchor[2]) / 2.0
                    boundary = int(round((previous_center + center) / 2.0))
                    band[0] = max(band[0], boundary)
                    boundaries['previous_midpoint_x'] = boundary
                if next_anchor:
                    next_center = (next_anchor[0] + next_anchor[2]) / 2.0
                    boundary = int(round((center + next_center) / 2.0))
                    band[2] = min(band[2], boundary)
                    boundaries['next_midpoint_x'] = boundary

            if _valid_xyxy(band, page):
                context['answer_search_domain'] = band
                context['sibling_group_id'] = group_id
                context['layout_axis'] = axis
                context['layout_axis_source'] = axis_source
                context['domain_source'] = 'ocr_anchor_sibling_partition'
                context['neighbor_boundaries'] = boundaries
                context['domain_before_sibling_partition'] = original
                context['anchor_source'] = anchor_source
                hint = (layout_hints or {}).get(item.item_id)
                if hint:
                    context['vlm_layout_hint'] = {
                        'axis': hint.get('axis'),
                        'confidence': hint.get('confidence'),
                        'regions': list(hint.get('regions') or []),
                    }
                updates += 1
    return updates


def match_lines(query, blocks):
    """Match adjacent lines; compare distinct start positions, not nested windows."""
    raw_query = str(query or '')
    query_label = _structural_label(raw_query)
    query = normalized(raw_query)
    query_body = normalized(re.sub(r'^\s*[（(\[]\s*\d{1,2}\s*[）)\]]\s*', '', raw_query))
    candidates = []
    for start in range(len(blocks)):
        best = None
        for end in range(start + 1, min(len(blocks), start + 4) + 1):
            if end > start + 1:
                a, b = blocks[end - 2].bbox, blocks[end - 1].bbox
                if b[1] - a[3] > max(a[3]-a[1], b[3]-b[1])*2.5:
                    break
            text = normalized(' '.join(b.text for b in blocks[start:end]))
            if len(text) < 3 and query_label is None:
                continue
            matcher = SequenceMatcher(None, query, text)
            score = matcher.ratio()
            coverage = sum(match.size for match in matcher.get_matching_blocks()) / max(1, len(query))
            score = max(score, min(1.0, coverage * .96))
            if query_label is not None:
                candidate_label = _structural_label(' '.join(b.text for b in blocks[start:end]))
                if candidate_label == query_label:
                    # A bare ``(1)`` is useful for a genuinely short label, but
                    # must not let a student's ``(1)解`` outrank the printed
                    # subquestion when the query has substantial body text.
                    if len(query_body) <= 4:
                        score = max(score, .90)
                    elif coverage >= .45:
                        score = min(1.0, score + .08)
            if query in text or text in query:
                score = max(score, min(1., min(len(query), len(text))/max(8, len(query))))
            candidate = (score, start, end)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best:
            candidates.append(best)
    candidates.sort(reverse=True)
    if not candidates or candidates[0][0] < (.82 if query_label else .5):
        return None
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < .05 and query_label is None:
        return None
    return candidates[0]


def ground_contexts(package, pages):
    """Rebuild coordinates on THIS document; uncertain identity uses labelled page context.

    Full-page contexts are explicitly unresolved. A visual proposal and candidate
    ownership review are mandatory before any answer coordinates can be accepted.
    """
    page_map = {p.index: p for p in pages}
    summary = {'grounded_items': 0, 'context_only_items': 0, 'unresolved_items': 0}
    for section in package.sections:
        for question in section.questions:
            for item in question.items:
                item.quality['question_identity'] = {'number': question.question_num, 'text': question.question_title}
                item.quality['printed_references'] = [question.question_title, item.question_text]
                refs = item.quality.get('structure_references') or []
                if not refs:
                    refs = [{'page_index': r.page_index, 'anchor': item.question_text}
                            for r in item.answer_regions]
                if not refs and item.stem_region:
                    refs = [{'page_index': item.stem_region.page_index, 'anchor': item.question_text}]
                contexts, regions, stems, details = [], [], [], []
                for index in dict.fromkeys(r['page_index'] for r in refs):
                    page = page_map.get(index)
                    if page is None:
                        details.append({'page_index': index, 'status': 'PAGE_MISSING'}); continue
                    width, height = page.width or 1654, page.height or 2338
                    groups = column_groups(page.ocr, width)
                    starts = [(g, k, b) for g in groups for k, b in enumerate(g)
                              if any(n == question.question_num and b is a for n, a in anchors(page))]
                    # Repeated numbering requires text evidence to disambiguate.
                    if len(starts) > 1:
                        ranked = sorted([(SequenceMatcher(None, normalized(question.question_title)[:120],
                                     normalized(b.text)[:120]).ratio(), pos) for pos, (_, _, b) in enumerate(starts)], reverse=True)
                        starts = [starts[ranked[0][1]]] if ranked[0][0] >= .5 and ranked[0][0]-ranked[1][0] > .1 else []
                    query = next(r['anchor'] for r in refs if r['page_index'] == index)
                    if not starts:
                        # Continuations can be grounded by their own printed anchor.
                        matches = [(match_lines(query, g), g) for g in groups]
                        matches = [(m, g) for m, g in matches if m and m[0] >= .75]
                        if len(matches) == 1:
                            m, g = matches[0]; starts = [(g, m[1], g[m[1]])]
                    if not starts:
                        whole_page = [0, 0, width, height]
                        contexts.append({'page_index': index, 'bbox': whole_page,
                                         'status': 'PAGE_CONTEXT_ONLY', 'coordinate_role': 'search_context',
                                         'question_context': list(whole_page),
                                         'answer_search_domain': list(whole_page),
                                         'printed_exclusion_regions': []})
                        details.append({'page_index': index, 'status': 'IDENTITY_REQUIRES_VISUAL_REVIEW',
                                        'rejection_reason': 'ANCHOR_NOT_FOUND_OR_AMBIGUOUS',
                                        'candidate_count': len(groups)})
                        continue
                    group, start, anchor = starts[0]
                    anchor_ids = {id(b) for _, b in anchors(page)}
                    end = next((k for k in range(start+1, len(group)) if id(group[k]) in anchor_ids), len(group))
                    blocks = group[start:end]
                    line_height = median(max(1, b.bbox[3]-b.bbox[1]) for b in blocks)
                    previous = next((group[k] for k in range(start-1, -1, -1) if id(group[k]) in anchor_ids), None)
                    top = max(0, anchor.bbox[1]-line_height*3)
                    if previous:
                        top = max(top, previous.bbox[3]+2)
                    bottom = min(height, group[end].bbox[1]-line_height*.6) if end < len(group) else height
                    left, right = max(0, min(b.bbox[0] for b in group)-line_height), min(width, max(b.bbox[2] for b in group)+line_height)
                    # Single-column context includes answers beyond the printed line ends.
                    if len(groups) == 1:
                        left, right = 0, width
                    context = [int(left), int(top), int(right), int(max(anchor.bbox[3], bottom))]
                    match = match_lines(query, blocks)
                    if match is None:
                        marker = _structural_label(query)
                        if marker and marker[0] == 'sub':
                            for position, candidate in enumerate(blocks):
                                if _structural_label(candidate.text) == marker:
                                    match = (.84, position, position + 1)
                                    break
                    if len(question.items) == 1:
                        match = match or (.8, 0, 1)
                    limits = [int(left), int(previous.bbox[3]+2) if previous else 0,
                              int(right), int(context[3])]
                    stem_box = None
                    if match:
                        score, first, last = match
                        chosen = blocks[first:last]
                        stem_box = [min(b.bbox[0] for b in chosen), min(b.bbox[1] for b in chosen),
                                    max(b.bbox[2] for b in chosen), max(b.bbox[3] for b in chosen)]
                        stems.append(PageRegion(index, page.path, stem_box, score, ' '.join(b.text for b in chosen)))
                    context_record = {'page_index': index, 'bbox': context, 'status': 'QUESTION_ANCHORED',
                                      'coordinate_role': 'question_context',
                                      'question_context': list(context), 'limits': limits,
                                      'anchor_bbox': [int(round(v)) for v in anchor.bbox]}
                    context_record['answer_search_domain'] = answer_domain_for_context(
                        item, context_record, page, stem_box) or list(context)
                    context_record['printed_exclusion_regions'] = _printed_exclusion_regions(
                        item, question, page, blocks, context, stem_box)
                    contexts.append(context_record)
                    regions.append(PageRegion(index, page.path, context, None, 'question_context_search_only'))
                    details.append({'page_index': index, 'status': 'ITEM_ANCHORED' if match else 'QUESTION_CONTEXT_ONLY',
                                    'matched_lines': [match[1], match[2]] if match else None,
                                    'candidate_score': match[0] if match else None,
                                    'rejection_reason': None if match else 'QUESTION_TEXT_NOT_UNIQUE'})
                item.quality['localization'] = {'status': 'GROUNDED' if stems else 'CONTEXT_ONLY' if contexts else 'UNRESOLVED',
                    'contexts': contexts, 'evidence': details, 'coordinate_source': 'current_document_ocr'}
                item.stem_region = stems[0] if stems else None
                item.answer_regions = regions
                item.option_regions = []
                item.blank_regions = []
                item.writing_regions = []
                if package.document_type == 'student':
                    item.student_regions = list(regions)
                summary['grounded_items' if stems else 'context_only_items' if contexts else 'unresolved_items'] += 1
            summary.setdefault('sibling_domains_partitioned', 0)
            summary['sibling_domains_partitioned'] += partition_sibling_answer_domains(
                question, page_map)
    return summary


def provisional_contexts(package, pages):
    """Create bounded, page-aware search contexts before OCR is available.

    A VLM tree supplies page ownership and printed anchors, but it is not
    allowed to supply pixel coordinates. Until OCR can ground those anchors,
    each referenced page is exposed as a labelled full-page search context.
    Slot proposals may use it to choose a local region; the context is
    replaced by :func:`ground_contexts` as soon as OCR validation completes.
    """
    page_map = {p.index: p for p in pages}
    summary = {'provisional_items': 0, 'unresolved_items': 0}
    for section in package.sections:
        for question in section.questions:
            for item in question.items:
                refs = item.quality.get('structure_references') or []
                page_ids = [r.get('page_index') for r in refs if r.get('page_index') in page_map]
                if not page_ids:
                    page_ids = [r.page_index for r in item.answer_regions if r.page_index in page_map]
                contexts = []
                regions = []
                for page_index in dict.fromkeys(page_ids):
                    page = page_map[page_index]
                    width, height = page.width or 1654, page.height or 2338
                    box = [0, 0, width, height]
                    contexts.append({'page_index': page_index, 'bbox': box,
                                     'status': 'PROVISIONAL_PAGE_CONTEXT',
                                     'coordinate_role': 'search_context',
                                     'question_context': list(box),
                                     'answer_search_domain': list(box),
                                     'printed_exclusion_regions': [],
                                     'anchor': next((r.get('anchor', '') for r in refs
                                                     if r.get('page_index') == page_index), '')})
                    regions.append(PageRegion(page_index, page.path, box, None,
                                              'provisional_vlm_page_context'))
                item.quality['localization'] = {
                    'status': 'PROVISIONAL' if contexts else 'UNRESOLVED',
                    'contexts': contexts,
                    'evidence': [{'page_index': c['page_index'], 'status': c['status']}
                                 for c in contexts],
                    'coordinate_source': 'vlm_page_ownership_only',
                }
                item.answer_regions = regions
                item.option_regions = []
                item.blank_regions = []
                item.writing_regions = []
                item.stem_region = None
                if package.document_type == 'student':
                    item.student_regions = list(regions)
                if contexts:
                    summary['provisional_items'] += 1
                else:
                    summary['unresolved_items'] += 1
    return summary
