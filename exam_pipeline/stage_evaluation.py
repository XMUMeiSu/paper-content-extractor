"""Stage diagnostics and optional independently annotated localization evaluation."""
from collections import Counter
from .evaluation import equal_answer, iou


def stage_diagnostics(package):
    stages = Counter(); failures = Counter(); warnings = Counter()
    for section in package.sections:
        for question in section.questions:
            for item in question.items:
                stages['items'] += 1
                loc = item.quality.get('localization', {})
                stages['items_with_context'] += bool(loc.get('contexts'))
                stages['items_with_ocr_anchor'] += loc.get('status') == 'GROUNDED'
                stages['items_with_verified_region'] += any(s.expected_bbox for s in item.slots)
                stages['items_complete'] += item.answer_status == 'COMPLETE'
                stages['items_partial'] += item.answer_status == 'PARTIAL'
                stages['items_unresolved'] += item.answer_status == 'UNRESOLVED'
                slot_eval = item.slot_evaluation or {}
                stages['logical_slots_expected'] += slot_eval.get(
                    'expected_slots', item.expected_slot_count or len(item.answer_parts))
                stages['logical_slots_recognized'] += slot_eval.get('recognized_slots', 0)
                stages['logical_slots_blank'] += slot_eval.get('blank_slots', 0)
                stages['logical_slots_unresolved'] += slot_eval.get('unresolved_slots', 0)
                stages['items_relocalized_after_recognition'] += bool(item.quality.get('automatic_relocalization'))
                for slot in item.slots:
                    stages['physical_regions'] += 1
                    stages['regions_locally_verified'] += slot.geometry_status in {
                        'ALIGNED', 'ALIGNED_WITH_WARNING'}
                    stages['regions_aligned_with_warning'] += slot.geometry_status == 'ALIGNED_WITH_WARNING'
                    review = slot.audit.get('local_validation',{}).get('visual_review',{}).get('response',{})
                    stages['regions_visually_accepted'] += review.get('decision') == 'ACCEPT' and bool(slot.expected_bbox)
                    stages['regions_transcribed'] += slot.content_status == 'RECOGNIZED'
                    for a in slot.audit.get('recognition',{}).get('attempts',[]):
                        failures.update(a.get('issues',[]))
                        warnings.update(a.get('warnings',[]))
                for a in item.slot_semantics_audit.get('attempts',[]):
                    failures.update(f['reason'] for f in a.get('failures',[]))
                    if a.get('reason'): failures.update([a['reason']])
    expected = stages['logical_slots_expected']
    completed = stages['logical_slots_recognized'] + stages['logical_slots_blank']
    return {'scope':'internal_completion_only_not_accuracy', 'counts':dict(stages),
            'slot_completion_rate': completed / expected if expected else None,
            'item_complete_rate': stages['items_complete'] / stages['items'] if stages['items'] else None,
            'attempt_failure_counts':dict(failures),
            'recognition_warning_counts':dict(warnings)}


def evaluate_stages(prediction, truth):
    """Truth: questions/items/slots as evaluation.py; match ID+page, never text.

    Context recall = >=95% of annotated answer pixels in a search context.
    Localization pass = IoU>=.5. Conditional transcription evaluates ONLY those
    localized pairs. Missing annotations remain absent denominators, not passes.
    """
    items={i['item_id']:i for s in prediction.get('sections',[]) for q in s.get('questions',[]) for i in q.get('items',[])}
    counts=Counter(annotated_regions=0, context_contains_answer=0, localized=0, localized_text_exact=0,
                   answers_expected=0, answers_exact=0)
    key='standard_answer' if prediction.get('document_type')=='teacher' else 'student_answer'
    for question in truth.get('questions',[]):
        for target in question.get('items',[]):
            item=items.get(target['id'],{})
            if 'answer' in target:
                counts['answers_expected']+=1;counts['answers_exact']+=equal_answer(item.get(key),target['answer'])
            used=set()
            for region in target.get('slots',[]):
                counts['annotated_regions']+=1;b=region['bbox'];page=region['page_index']
                for context in item.get('quality',{}).get('localization',{}).get('contexts',[]):
                    if context['page_index']!=page:continue
                    c=context['bbox'];overlap=max(0,min(c[2],b[2])-max(c[0],b[0]))*max(0,min(c[3],b[3])-max(c[1],b[1]))
                    if overlap/max(1,(b[2]-b[0])*(b[3]-b[1]))>=.95:
                        counts['context_contains_answer']+=1;break
                candidates=[]
                for n,slot in enumerate(item.get('slots',[])):
                    sid=slot.get('semantic_id') or '{}:slot:{}'.format(target['id'],slot['slot_idx'])
                    box=slot.get('handwriting_bbox') or slot.get('expected_bbox')
                    if n not in used and sid==region['id'] and slot['page_index']==page and len(box or [])==4:
                        candidates.append((iou([box[1],box[0],box[3],box[2]],b),n,slot))
                if candidates:
                    overlap,n,slot=max(candidates,key=lambda c:c[0]);used.add(n)
                    if overlap>=.5:
                        counts['localized']+=1
                        counts['localized_text_exact']+=equal_answer(slot.get('expected_text') if key=='standard_answer' else slot.get('student_answer'),region.get('text'))
    ratio=lambda n,d: counts[n]/counts[d] if counts[d] else None
    return {'counts':dict(counts),'context_recall':ratio('context_contains_answer','annotated_regions'),
            'localization_recall_iou50':ratio('localized','annotated_regions'),
            'transcription_exact_given_localized':ratio('localized_text_exact','localized'),
            'complete_answer_exact':ratio('answers_exact','answers_expected')}
