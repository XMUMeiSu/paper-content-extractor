"""Visual question-context refinement using OCR bounds as hard evidence."""
import copy
import json
import math
from pathlib import Path

from .contracts import PageRegion
from .io_utils import atomic_write_json


REGION = {
    'type': 'object', 'additionalProperties': False,
    'required': ['page_index', 'search_bbox'],
    'properties': {
        'page_index': {'type': 'integer', 'minimum': 1},
        'search_bbox': {'type': 'array', 'minItems': 4, 'maxItems': 4,
                        'items': {'type': 'number', 'minimum': 0, 'maximum': 1000}},
    },
}
ITEM = {
    'type': 'object', 'additionalProperties': False,
    'required': ['item_id', 'regions', 'confidence'],
    'properties': {
        'item_id': {'type': 'string'},
        'regions': {'type': 'array', 'minItems': 1, 'maxItems': 20, 'items': REGION},
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
    },
}
QUESTION_LOCALIZATION_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['items'],
    'properties': {'items': {'type': 'array', 'minItems': 1, 'maxItems': 500,
                             'items': ITEM}},
}


def _items(package):
    return [item for section in package.sections for question in section.questions
            for item in question.items]


def _intersection_box(left, right):
    box = [max(left[0], right[0]), max(left[1], right[1]),
           min(left[2], right[2]), min(left[3], right[3])]
    return box if box[0] < box[2] and box[1] < box[3] else None


class VisualQuestionLocalizer:
    def __init__(self, request=None, provider='none', max_attempts=2):
        self.request = request
        self.provider = provider
        self.max_attempts = max(1, min(2, max_attempts))

    @staticmethod
    def _validate(raw, package, pages):
        if not isinstance(raw, dict) or set(raw) != {'items'} or not isinstance(raw['items'], list):
            raise ValueError('INVALID_QUESTION_LOCALIZATION_RESPONSE')
        expected = {item.item_id: item for item in _items(package)}
        page_map = {page.index: page for page in pages}
        result = {}
        for entry in raw['items']:
            if not isinstance(entry, dict) or set(entry) != {'item_id', 'regions', 'confidence'}:
                raise ValueError('INVALID_QUESTION_LOCALIZATION_FIELDS')
            item_id = entry['item_id']
            if item_id not in expected or item_id in result:
                raise ValueError('UNKNOWN_OR_DUPLICATE_LOCALIZATION_ITEM')
            confidence = entry['confidence']
            if type(confidence) not in (int, float) or not math.isfinite(confidence) or confidence < .6:
                raise ValueError('LOW_QUESTION_LOCALIZATION_CONFIDENCE')
            allowed = {ref.get('page_index') for ref in
                       expected[item_id].quality.get('structure_references', [])}
            regions = []
            for region in entry['regions']:
                if not isinstance(region, dict) or set(region) != {'page_index', 'search_bbox'}:
                    raise ValueError('INVALID_QUESTION_LOCALIZATION_REGION')
                page_index = region['page_index']
                box = region['search_bbox']
                if (type(page_index) is not int or page_index not in page_map
                        or allowed and page_index not in allowed):
                    raise ValueError('QUESTION_LOCALIZATION_PAGE_CONFLICT')
                if (not isinstance(box, list) or len(box) != 4
                        or not all(type(value) in (int, float) and math.isfinite(value)
                                   for value in box)
                        or not 0 <= box[0] < box[2] <= 1000
                        or not 0 <= box[1] < box[3] <= 1000):
                    raise ValueError('INVALID_QUESTION_LOCALIZATION_BOX')
                regions.append({'page_index': page_index, 'search_bbox': list(box)})
            if not regions:
                raise ValueError('MISSING_QUESTION_LOCALIZATION_REGION')
            result[item_id] = {'regions': regions, 'confidence': float(confidence)}
        missing = sorted(set(expected) - set(result))
        if missing:
            raise ValueError('MISSING_QUESTION_LOCALIZATION_ITEMS:' + ','.join(missing[:10]))
        return result

    @staticmethod
    def _tree_payload(package):
        return {'subject': package.subject, 'sections': [
            {'section_id': section.section_id, 'title': section.section_title,
             'questions': [
                 {'question_id': question.question_id, 'number': question.question_num,
                  'text': question.question_title,
                  'items': [
                      {'item_id': item.item_id, 'label': item.item_name,
                       'text': item.question_text,
                       'references': copy.deepcopy(item.quality.get('structure_references', []))}
                      for item in question.items]}
                 for question in section.questions]}
            for section in package.sections]}

    def localize(self, package, pages, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        audit = {'provider': self.provider, 'mode': 'ocr_bounded_visual_question_localization',
                 'coordinate_role': 'question_search_context_only', 'status': 'UNRESOLVED',
                 'attempts': []}
        failures = []
        selected = None
        images = [Path(page.path) for page in pages]
        for attempt in range(self.max_attempts if self.request else 0):
            prompt = (
                'Locate every listed exam item on the ORIGINAL pages before OCR. Return one or more coarse '
                'question search regions per item as normalized xyxy 0..1000 relative to the full page. '
                'Each region must contain the complete printed prompt, options, blanks and likely writing area, '
                'while excluding neighboring questions when possible. These are search contexts only, never final '
                'answer coordinates. Do not solve or transcribe answers. Preserve item IDs and declared page '
                'ownership. On retry correct the listed protocol failures. Return only schema JSON.\n'
            )
            context = {'tree': self._tree_payload(package),
                       'pages': [{'page_index': p.index, 'width': p.width, 'height': p.height}
                                 for p in pages],
                       'failures': failures, 'schema': QUESTION_LOCALIZATION_SCHEMA}
            record = {'attempt': attempt + 1}
            try:
                raw = self.request(prompt + json.dumps(context, ensure_ascii=False),
                                   images, QUESTION_LOCALIZATION_SCHEMA)
                proposal = self._validate(raw, package, pages)
                record.update(status='PROPOSED', proposal=raw)
                selected = proposal
                audit['attempts'].append(record)
                break
            except Exception as exc:
                reason = str(exc) if isinstance(exc, ValueError) else 'QUESTION_LOCALIZATION_BACKEND_ERROR'
                failures = [{'reason': reason}]
                record.update(status='FAILED', reason=reason, error_type=type(exc).__name__)
                audit['attempts'].append(record)
        if selected is None:
            audit['reason'] = failures[-1]['reason'] if failures else 'QUESTION_LOCALIZATION_BACKEND_UNAVAILABLE'
            atomic_write_json(output_dir / 'question_localization.json', audit)
            return None, audit

        page_map = {page.index: page for page in pages}
        for item in _items(package):
            prior_localization = copy.deepcopy(item.quality.get('localization') or {})
            prior_contexts = {
                context.get('page_index'): context
                for context in prior_localization.get('contexts', [])
                if isinstance(context, dict) and context.get('page_index') is not None
            }
            prior_stem = item.stem_region
            contexts = []
            regions = []
            entry = selected[item.item_id]
            for region in entry['regions']:
                page = page_map[region['page_index']]
                width, height = page.width or 1654, page.height or 2338
                normalized = region['search_bbox']
                box = [int(round(normalized[0] * width / 1000)),
                       int(round(normalized[1] * height / 1000)),
                       int(round(normalized[2] * width / 1000)),
                       int(round(normalized[3] * height / 1000))]
                prior = prior_contexts.get(page.index)
                source = 'visual_question_context'
                status = 'VISUAL_QUESTION_CONTEXT'
                if prior and prior.get('status') == 'QUESTION_ANCHORED':
                    bounded = _intersection_box(box, prior.get('bbox') or box)
                    if bounded is None:
                        # OCR established the item identity and reading-order
                        # boundary. A contradictory visual hint cannot replace it.
                        bounded = list(prior['bbox'])
                        status = 'OCR_CONTEXT_RETAINED'
                        source = 'ocr_context_after_visual_conflict'
                    else:
                        status = 'QUESTION_CONTEXT_CROSS_VALIDATED'
                        source = 'ocr_visual_context_intersection'
                    box = [int(round(value)) for value in bounded]
                context = {'page_index': page.index, 'bbox': box,
                           'status': status, 'coordinate_role': 'search_context',
                           'confidence': entry['confidence'],
                           'visual_bbox': [int(round(value)) for value in [
                               normalized[0] * width / 1000,
                               normalized[1] * height / 1000,
                               normalized[2] * width / 1000,
                               normalized[3] * height / 1000,
                           ]]}
                if prior:
                    context['ocr_bbox'] = list(prior.get('bbox') or [])
                    if prior.get('limits'):
                        context['limits'] = list(prior['limits'])
                    # Preserve the OCR-derived answer corridor and printed
                    # exclusions.  Visual question localization may narrow
                    # the prompt context, but it must never shrink a large
                    # answer region back to the prompt rectangle.
                    context['question_context'] = list(box)
                    context['answer_search_domain'] = list(prior.get(
                        'answer_search_domain', prior.get('limits', box)))
                    context['printed_exclusion_regions'] = copy.deepcopy(
                        prior.get('printed_exclusion_regions', []))
                    for key in ('sibling_group_id', 'layout_axis', 'layout_axis_source',
                                'domain_source', 'neighbor_boundaries',
                                'domain_before_sibling_partition', 'anchor_source',
                                'vlm_layout_hint'):
                        if key in prior:
                            context[key] = copy.deepcopy(prior[key])
                else:
                    context['question_context'] = list(box)
                    context['answer_search_domain'] = list(box)
                    context['printed_exclusion_regions'] = []
                contexts.append(context)
                regions.append(PageRegion(page.index, page.path, box, entry['confidence'], source))
            item.quality['localization'] = {
                'status': ('CROSS_VALIDATED' if any(
                    context['status'] == 'QUESTION_CONTEXT_CROSS_VALIDATED'
                    for context in contexts) else 'PROVISIONAL_VISUAL'),
                'contexts': contexts,
                'evidence': [{'page_index': c['page_index'], 'status': c['status']}
                             for c in contexts],
                'coordinate_source': 'ocr_visual_cross_validation',
                'prior_ocr_status': prior_localization.get('status'),
            }
            item.answer_regions = regions
            item.option_regions = []
            item.blank_regions = []
            item.writing_regions = []
            # OCR owns the printed anchor coordinates. The visual context may
            # narrow the search domain, but it cannot erase that evidence.
            item.stem_region = prior_stem
            if package.document_type == 'student':
                item.student_regions = list(regions)
        audit.update(status='PROPOSED', localized_items=len(selected))
        atomic_write_json(output_dir / 'question_localization.json', audit)
        return {'provisional_items': len(selected), 'unresolved_items': 0,
                'source': 'visual_question_search_regions'}, audit


__all__ = ['VisualQuestionLocalizer', 'QUESTION_LOCALIZATION_SCHEMA']
