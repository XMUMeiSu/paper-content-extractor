"""One bounded localization retry driven by recognition evidence, never by truth."""
import copy
from pathlib import Path

CONTAMINATION = {'PRINTED_STEM_CONTAMINATION'}
REPAIRABLE = CONTAMINATION | {
    'ANCHOR_NOT_FOUND', 'FORMULA_DISAGREEMENT',
    'FORMULA_STRUCTURE_DISAGREEMENT', 'REGISTRATION_LOW',
    'PAGE_ID_MISMATCH', 'JSON_SCHEMA_ERROR',
}


def _aligned_count(item):
    return sum(slot.geometry_status in {'ALIGNED', 'ALIGNED_WITH_WARNING'}
               and len(slot.expected_bbox or []) == 4 for slot in item.slots)


def _recognized_count(item):
    return sum(slot.content_status == 'RECOGNIZED' and bool(
        str(slot.student_answer or slot.recognized_text or slot.expected_text or '').strip())
        for slot in item.slots)


def _issue_count(item):
    return sum(len((slot.audit.get('recognition') or {}).get('issues', []))
               + len(slot.errors or []) for slot in item.slots)


def _is_improvement(previous, revised):
    """Only adopt a retry that keeps evidence and improves a measured state."""
    old = (_aligned_count(previous), _recognized_count(previous), -_issue_count(previous))
    new = (_aligned_count(revised), _recognized_count(revised), -_issue_count(revised))
    return new[0] >= old[0] and new[1] >= old[1] and new > old


def failure_types(package):
    """Collect normalized failure codes for routing and audit output."""
    found = []
    for section in package.sections:
        for question in section.questions:
            for item in question.items:
                for slot in item.slots:
                    found.extend((entry.get('code') if isinstance(entry, dict) else entry)
                                 for entry in (slot.errors or []))
                    found.extend((slot.audit.get('recognition') or {}).get('issues', []))
                    found.append((slot.audit.get('local_validation') or {}).get('reason'))
                for detail in item.quality.get('localization', {}).get('evidence', []):
                    if detail.get('rejection_reason'):
                        found.append('ANCHOR_NOT_FOUND')
    for meta in package.registration or []:
        if str(meta.get('status', '')).startswith('FAILED'):
            found.append('REGISTRATION_LOW')
    return sorted({str(value) for value in found if value})


def repair_contaminated_answers(package, pages, semantic_service, recognize, output_dir, reference_pages=()):
    from .slot_evidence import intersection, area
    from .roi import yxyx_to_xyxy
    selected = []
    selected_reasons = {}
    for section in package.sections:
        for question in section.questions:
            for item in question.items:
                issues = {x for slot in item.slots for x in slot.audit.get('recognition',{}).get('issues',[])}
                issues |= {(slot.audit.get('local_validation') or {}).get('reason') for slot in item.slots}
                issues.discard(None)
                reasons = sorted(issues & REPAIRABLE)
                semantic_attempts = len((item.slot_semantics_audit or {}).get('attempts') or [])
                # OCR coordinate misses cannot be repaired by asking the VLM
                # for another pixel box. Only schema/page/anchor failures and
                # content-specific formula or contamination failures retry.
                aligned = _aligned_count(item)
                coordinate_failure = bool(set(reasons) & {
                    'ANCHOR_NOT_FOUND', 'REGISTRATION_LOW', 'PAGE_ID_MISMATCH',
                    'JSON_SCHEMA_ERROR',
                }) and aligned == 0
                retryable_after_budget = bool(set(reasons) & {
                    'PRINTED_STEM_CONTAMINATION',
                    'FORMULA_DISAGREEMENT', 'FORMULA_STRUCTURE_DISAGREEMENT',
                    'RECOGNITION_DISAGREEMENT', 'JSON_SCHEMA_ERROR', 'PAGE_ID_MISMATCH',
                }) or (coordinate_failure and semantic_attempts < 2)
                if reasons and retryable_after_budget and not item.quality.get('automatic_relocalization'):
                    selected.append(item)
                    selected_reasons[item.item_id] = reasons
    if not selected or semantic_service.request is None:
        return {'retried_items': 0, 'eligible_items': len(selected),
                'failure_types': failure_types(package), 'status': 'NO_RETRY_BACKEND' if selected else 'NO_FAILURES'}
    ids = {i.item_id for i in selected}
    retry = copy.deepcopy(package)
    for section in retry.sections:
        for question in section.questions:
            question.items = [i for i in question.items if i.item_id in ids]
            for item in question.items:
                item.quality['localization_retry_feedback'] = {
                    'reason': selected_reasons.get(item.item_id, ['UNCLASSIFIED_FAILURE'])[0],
                    'failure_types': selected_reasons.get(item.item_id, []),
                    'rejected_regions':[{'page_index':s.page_index,'bbox':s.expected_bbox} for s in item.slots]}
                item.slots=[]
        section.questions=[q for q in section.questions if q.items]
    retry.sections=[s for s in retry.sections if s.questions]
    output_dir = Path(output_dir)
    # Use the normal page-batched semantic protocol for repairs. Custom test
    # services without ``propose_package`` retain the direct compatibility path.
    if hasattr(semantic_service, 'propose_package'):
        semantic_service.propose_package(
            retry, pages, output_dir / 'proposals', reference_pages=reference_pages)
        semantic_service.enrich_package(
            retry, pages, output_dir / 'grounding', reference_pages,
            reuse_proposals=True)
    else:
        semantic_service.enrich_package(retry,pages,output_dir,reference_pages)
    occupied=[(s.page_index,yxyx_to_xyxy(s.expected_bbox)) for sec in package.sections for q in sec.questions
              for i in q.items if i.item_id not in ids for s in i.slots if len(s.expected_bbox or [])==4]
    for sec in retry.sections:
        for q in sec.questions:
            for item in q.items:
                for slot in item.slots:
                    if len(slot.expected_bbox or []) != 4: continue
                    box=yxyx_to_xyxy(slot.expected_bbox)
                    if any(page==slot.page_index and intersection(box,b)/max(1,min(area(box),area(b)))>.4 for page,b in occupied):
                        slot.expected_bbox=[];slot.geometry_status='MISSING';slot.audit['topology_source']='missing_placeholder'
                        item.slot_semantics_audit['status']='PARTIAL';item.slot_semantics_audit['reason']='SLOT_EVIDENCE_CONFLICT'
                    else: occupied.append((slot.page_index,box))
    recognize(retry)
    revised={i.item_id:i for s in retry.sections for q in s.questions for i in q.items}
    adopted = 0
    retained = 0
    for old in selected:
        new=revised[old.item_id]
        repair_audit={'attempts':1,'trigger':'FAILURE_TYPE_ROUTING',
            'failure_types': selected_reasons.get(old.item_id, []),
            'previous_semantics':copy.deepcopy(old.slot_semantics_audit),
            'previous_recognition':[s.audit.get('recognition') for s in old.slots],
            'previous_score': {'aligned': _aligned_count(old),
                               'recognized': _recognized_count(old),
                               'issues': _issue_count(old)},
            'retry_score': {'aligned': _aligned_count(new),
                            'recognized': _recognized_count(new),
                            'issues': _issue_count(new)}}
        if _is_improvement(old, new):
            old.slots=new.slots;old.slot_semantics_audit=new.slot_semantics_audit
            old.semantic_slot_plan=new.semantic_slot_plan;old.expected_slot_count=new.expected_slot_count
            repair_audit['status']='ADOPTED_IMPROVEMENT'
            adopted += 1
        else:
            repair_audit['status']='RETAINED_PREVIOUS_RESULT'
            retained += 1
        old.quality['automatic_relocalization']=repair_audit
    for item in selected:
        item.quality.setdefault('automatic_relocalization', {})['failure_types'] = selected_reasons.get(item.item_id, [])
        if not any(slot.geometry_status in {'ALIGNED', 'ALIGNED_WITH_WARNING'} for slot in item.slots):
            item.quality['automatic_relocalization']['status'] = 'AUTO_REJECT'
            for slot in item.slots:
                slot.review_status = 'AUTO_REJECT'
                slot.errors.append({'code': 'AUTO_REJECT', 'retry_exhausted': True, 'retryable': False})
    return {'retried_items':len(selected), 'adopted_items': adopted,
            'retained_items': retained, 'failure_types': failure_types(package),
            'status': 'RETRIED_WITH_AUTO_REJECTS' if any(
                item.quality.get('automatic_relocalization', {}).get('status') == 'AUTO_REJECT'
                for item in selected) else 'RETRIED'}
