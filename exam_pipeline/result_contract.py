"""Finalize answer outputs from verified slots, without inventing missing values."""
import math
from .contracts import PageRegion
from .roi import yxyx_to_xyxy


def valid_box(box, page, order='xyxy'):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        b = [float(v) for v in box]
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(v) for v in b):
        return False
    if order == 'yxyx':
        b = [b[1], b[0], b[3], b[2]]
    return 0 <= b[0] < b[2] <= (page.width or 1654) and 0 <= b[1] < b[3] <= (page.height or 2338)


def page_map(pages):
    result = {}
    for page in pages:
        if page.index < 1 or page.index in result:
            raise ValueError('Page IDs must be unique positive integers')
        result[page.index] = page
    return result


def update_pipeline_status(package):
    """Derive stage states from leaf evidence in one place.

    ``VALID`` is deliberately reserved for schema validity.  A package can be
    schema-valid while its structure, geometry, or answer content still needs
    review; only the conjunction of those gates is production-ready.
    """
    items = [item for section in package.sections for question in section.questions
             for item in question.items]
    slots = [slot for item in items for slot in item.slots]
    package.schema_status = "VALID"
    structure = package.structure_audit or {}
    package.structure_status = (
        "COMPLETE" if structure.get("status") in {"COMPLETE", "VALID"}
        else "UNRESOLVED" if structure else "PENDING"
    )
    if not slots:
        package.geometry_status = "UNRESOLVED" if items else "PENDING"
        package.content_status = "UNRESOLVED" if items else "PENDING"
    else:
        geometry_values = {s.geometry_status for s in slots}
        content_values = {s.content_status for s in slots}
        package.geometry_status = (
            "COMPLETE" if geometry_values and geometry_values <= {"ALIGNED", "ALIGNED_WITH_WARNING"}
            else "PARTIAL" if geometry_values & {"ALIGNED", "ALIGNED_WITH_WARNING"}
            else "UNRESOLVED"
        )
        package.content_status = (
            "COMPLETE" if content_values and content_values <= {"RECOGNIZED", "BLANK"}
            else "PARTIAL" if content_values & {"RECOGNIZED", "BLANK"}
            else "UNRESOLVED"
        )
    package.production_status = (
        "PRODUCTION_READY"
        if package.structure_status == "COMPLETE"
        and package.geometry_status == "COMPLETE"
        and package.content_status == "COMPLETE"
        and not package.extraction_errors
        and (not items or all((item.slot_semantics_audit or {}).get("status") == "ACCEPTED"
                              for item in items))
        else "NEED_REVIEW"
    )
    package.stage_status = {
        "schema": package.schema_status,
        "structure": package.structure_status,
        "geometry": package.geometry_status,
        "content": package.content_status,
        "production": package.production_status,
    }
    return package.stage_status


def finalize_answers(package, pages):
    """One authoritative output stream; retain null positions and partial text."""
    indexed = page_map(pages)
    errors = []
    student = package.document_type == 'student'
    for section in package.sections:
        for question in section.questions:
            for item in question.items:
                parts, regions = [], []
                for region in item.answer_regions + item.option_regions + item.blank_regions + item.writing_regions:
                    if region.page_index in indexed:
                        region.page_file = indexed[region.page_index].path
                if item.stem_region and item.stem_region.page_index in indexed:
                    item.stem_region.page_file = indexed[item.stem_region.page_index].path
                for slot in sorted(item.slots, key=lambda s: s.slot_idx):
                    page = indexed.get(slot.page_index)
                    b = slot.handwriting_bbox or slot.evidence_bbox
                    slot.semantic_id = slot.semantic_id or '{}:slot:{}'.format(item.item_id, slot.slot_idx)
                    if page is None:
                        slot.geometry_status = 'MISSING'
                        slot.content_status = 'NOT_EVALUATED'
                        slot.review_status = 'NEED_REVIEW'
                        code = 'PAGE_UNAVAILABLE'
                    elif (not slot.expected_bbox
                          and slot.audit.get('topology_source') == 'missing_placeholder'
                          and not slot.audit.get('blank_confirmed')):
                        slot.geometry_status = 'MISSING'
                        slot.content_status = 'NOT_EVALUATED'
                        slot.review_status = 'NEED_REVIEW'
                        code = 'SLOT_CANDIDATE_MISSING'
                    elif not slot.expected_bbox and slot.audit.get('blank_confirmed'):
                        slot.geometry_status = 'MISSING'
                        slot.content_status = 'BLANK'
                        slot.review_status = 'AUTO_PASS'
                        code = None
                    elif not valid_box(slot.expected_bbox, page, 'yxyx'):
                        slot.geometry_status = 'FAILED'
                        slot.content_status = 'NOT_EVALUATED'
                        slot.review_status = 'NEED_REVIEW'
                        code = 'INVALID_EXPECTED_BOX'
                    elif b and not valid_box(b, page, 'yxyx'):
                        slot.geometry_status = 'FAILED'
                        slot.review_status = 'NEED_REVIEW'
                        code = 'INVALID_ANSWER_BOX'
                    else:
                        code = None
                    if code:
                        error = {'code': code, 'item_id': item.item_id, 'slot_id': slot.semantic_id,
                                 'page_index': slot.page_index, 'retryable': False}
                        errors.append(error)
                        if error not in slot.errors:
                            slot.errors.append(error)
                    for error in slot.errors:
                        detailed = dict(error, item_id=item.item_id, slot_id=slot.semantic_id,
                                        page_index=slot.page_index)
                        if detailed not in errors:
                            errors.append(detailed)
                    fragments = []
                    for fragment in slot.answer_fragments:
                        fp = indexed.get(fragment.get('page_index'))
                        if fp and valid_box(fragment.get('bbox'), fp):
                            fragments.append(dict(fragment, page_file=fp.path))
                    if not fragments and page and b and valid_box(b, page, 'yxyx'):
                        fragments = [{'page_index': page.index, 'page_file': page.path,
                                      'bbox': yxyx_to_xyxy(b), 'text': slot.recognized_text}]
                    slot.answer_fragments = fragments
                    text = slot.student_answer if student else slot.expected_text
                    usable = slot.content_status == 'RECOGNIZED' and slot.geometry_status not in {'FAILED', 'MISSING'}
                    if not student:
                        usable = (slot.status == 'TEACHER_ANSWER_EXTRACTED'
                                  and slot.geometry_status in {'ALIGNED', 'ALIGNED_WITH_WARNING'}
                                  and slot.content_status == 'RECOGNIZED')
                    if not usable or code:
                        text = None
                        for issue in (slot.audit.get("recognition") or {}).get("issues", []):
                            errors.append({"code": issue, "item_id":item.item_id,
                                           "slot_id":slot.semantic_id,"page_index":slot.page_index,
                                           "retryable":False, "retry_exhausted":True})
                    state = ('RECOGNIZED' if text is not None else 'BLANK'
                             if slot.content_status == 'BLANK' else 'UNRESOLVED')
                    recognition = slot.audit.get('recognition') or {}
                    parts.append({'slot_id': slot.semantic_id, 'slot_idx': slot.slot_idx,
                                  'text': text, 'status': state, 'fragments': fragments,
                                  'warnings': list(recognition.get('warnings', [])),
                                  'issues': list(recognition.get('issues', [])),
                                  'agreement_type': recognition.get('agreement_type', 'NONE')})
                    for fragment in fragments:
                        regions.append(PageRegion(fragment['page_index'], fragment['page_file'],
                                                  fragment['bbox'], None, fragment.get('text') or ''))
                # Cross-page fragments of a free response retain one semantic
                # answer identity, while each physical occurrence is verified.
                grouped = {}
                for part in parts:
                    key = part["slot_id"]
                    if key not in grouped:
                        grouped[key] = dict(part, fragments=list(part["fragments"]))
                    else:
                        prior = grouped[key]
                        prior["fragments"].extend(part["fragments"])
                        if prior["text"] is not None and part["text"] is not None:
                            prior["text"] += "\n" + part["text"]
                        else:
                            prior["text"] = None
                            prior["status"] = "UNRESOLVED"
                parts = list(grouped.values())
                expected = max(item.expected_slot_count or 0, len(parts)) or len(parts)
                complete = (bool(parts) and len(parts) == expected
                            and all(p['status'] in {'RECOGNIZED', 'BLANK'} for p in parts)
                            and all(
                                s.geometry_status not in {'FAILED', 'UNCERTAIN'}
                                and (s.geometry_status != 'MISSING'
                                     or s.content_status == 'BLANK'
                                     and s.audit.get('blank_confirmed'))
                                for s in item.slots
                            ))
                if item.slot_semantics_audit and item.slot_semantics_audit.get('status') != 'ACCEPTED':
                    complete = False
                    errors.append({'code': 'SLOT_SEMANTICS_UNRESOLVED', 'item_id': item.item_id,
                                   'retryable': False, 'reason': item.slot_semantics_audit.get('reason'),
                                   'retry_exhausted': bool(item.slot_semantics_audit.get('attempts'))})
                item.answer_status = ('COMPLETE' if complete else 'PARTIAL' if any(p['text'] is not None for p in parts)
                                      else 'UNRESOLVED')
                item.answer_parts = parts
                recognized_slots = sum(p['status'] == 'RECOGNIZED' for p in parts)
                blank_slots = sum(p['status'] == 'BLANK' for p in parts)
                unresolved_slots = max(0, expected - recognized_slots - blank_slots)
                item.slot_evaluation = {
                    'expected_slots': expected,
                    'observed_slots': len(parts),
                    'recognized_slots': recognized_slots,
                    'blank_slots': blank_slots,
                    'unresolved_slots': unresolved_slots,
                    'geometry_verified_slots': len({
                        s.semantic_id or '{}:slot:{}'.format(item.item_id, s.slot_idx)
                        for s in item.slots
                        if s.geometry_status in {'ALIGNED', 'ALIGNED_WITH_WARNING'}}),
                    'geometry_warning_slots': len({
                        s.semantic_id or '{}:slot:{}'.format(item.item_id, s.slot_idx)
                        for s in item.slots if s.geometry_status == 'ALIGNED_WITH_WARNING'}),
                    'complete': complete,
                    'completion_rate': ((recognized_slots + blank_slots) / expected
                                        if expected else None),
                }
                values = [p['text'] for p in parts]
                # Multi-slot answers are ordered arrays with nulls, never a compressed string.
                answer = values[0] if len(values) == 1 else values if values else None
                if student:
                    item.student_answer = answer
                    item.student_regions = regions
                else:
                    item.standard_answer = answer
                if item.answer_status != 'COMPLETE':
                    errors.append({'code': 'ANSWER_INCOMPLETE', 'item_id': item.item_id,
                                   'retryable': False, 'status': item.answer_status})
    if package.structure_audit and package.structure_audit.get("status") != "COMPLETE":
        errors.append({"code":"STRUCTURE_INCOMPLETE", "retryable":False})
    package.extraction_errors = errors
    package.extraction_status = 'PARTIAL' if errors else 'COMPLETE'
    update_pipeline_status(package)
    return errors
