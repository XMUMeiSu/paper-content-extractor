"""Whole-document visual topology with VLM schema and page validation."""
import copy
import hashlib
import json
import logging
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from .contracts import ExamItem, ExamQuestion, ExamSection, PageRegion
from .io_utils import atomic_write_json
from .prompt_loader import load_prompt

LOGGER = logging.getLogger(__name__)


def obj(properties):
    return {'type': 'object', 'additionalProperties': False,
            'required': list(properties), 'properties': properties}


def array(items, minimum=1):
    return {'type': 'array', 'items': items, 'minItems': minimum}


STRING = {'type': 'string'}
PAGE_SOURCE = {'type': 'string', 'enum': ['original', 'supplementary_template']}
REFERENCE = obj({'page_index': {'type': 'integer', 'minimum': 1}, 'anchor': STRING})
ITEM = obj({'item_id': STRING, 'label': STRING, 'text': STRING,
            'type': {'type': 'string', 'enum': ['choice', 'fill', 'solve', 'large_writing', 'other']},
            'references': array(REFERENCE)})
QUESTION = obj({'question_id': STRING, 'number': {'type': 'integer', 'minimum': 1},
                'text': STRING, 'references': array(REFERENCE), 'items': array(ITEM)})
DOCUMENT_STRUCTURE_SCHEMA = obj({'title': STRING, 'sections': array(obj({
    'section_id': STRING, 'title': STRING, 'questions': array(QUESTION)})),
    'non_question_pages': array(obj({'source': PAGE_SOURCE,
                                    'page_index': {'type': 'integer', 'minimum': 1},
                                    'reason': STRING}), 0)})

# The visual model first returns only topology. IDs are deliberately absent:
# they are generated locally after the complete document has been observed.
# Full text is added by bounded page requests, so a truncated transcription
# cannot discard an otherwise valid whole-document hierarchy.
TOPOLOGY_ITEM = obj({'label': STRING,
                     'type': {'type': 'string', 'enum': [
                         'choice', 'fill', 'solve', 'large_writing', 'other']},
                     'references': array(REFERENCE)})
TOPOLOGY_QUESTION = obj({'number': {'type': 'integer', 'minimum': 1},
                         'references': array(REFERENCE),
                         'items': array(TOPOLOGY_ITEM)})
TOPOLOGY_SECTION = obj({'title': STRING, 'questions': array(TOPOLOGY_QUESTION)})
DOCUMENT_TOPOLOGY_SCHEMA = obj({
    'title': STRING,
    'sections': array(TOPOLOGY_SECTION),
    'non_question_pages': array(obj({'source': PAGE_SOURCE,
                                     'page_index': {'type': 'integer', 'minimum': 1},
                                     'reason': STRING}), 0),
})
PAGE_TOPOLOGY_SCHEMA = obj({
    'title': STRING,
    'sections': array(TOPOLOGY_SECTION, 0),
    'non_question_pages': array(obj({'source': PAGE_SOURCE,
                                     'page_index': {'type': 'integer', 'minimum': 1},
                                     'reason': STRING}), 0),
})
PAGE_TEXT_ITEM = obj({'item_id': STRING, 'text': STRING, 'anchor': STRING})
PAGE_TEXT_QUESTION = obj({'question_id': STRING, 'text': STRING, 'anchor': STRING,
                          'items': array(PAGE_TEXT_ITEM, 0)})
PAGE_TEXT_SCHEMA = obj({'page_index': {'type': 'integer', 'minimum': 1},
                        'questions': array(PAGE_TEXT_QUESTION, 0)})


def check_schema(value, schema, path='document'):
    kind = schema['type']
    valid = {'object': lambda: isinstance(value, dict), 'array': lambda: isinstance(value, list),
             'string': lambda: isinstance(value, str), 'integer': lambda: type(value) is int}[kind]()
    if not valid:
        raise ValueError('INVALID_TYPE:' + path)
    if kind == 'object':
        if set(value) != set(schema['properties']):
            raise ValueError('INVALID_FIELDS:' + path)
        for key, child in schema['properties'].items():
            check_schema(value[key], child, path + '.' + key)
    elif kind == 'array':
        if len(value) < schema.get('minItems', 0):
            raise ValueError('EMPTY_ARRAY:' + path)
        for i, child in enumerate(value):
            check_schema(child, schema['items'], path + '[{}]'.format(i))
    elif kind == 'integer' and value < schema.get('minimum', value):
        raise ValueError('INVALID_NUMBER:' + path)
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError('INVALID_ENUM:' + path)


def norm(text):
    # Keep math symbols and bracketed numbering in a comparable form while
    # removing OCR spacing/case noise.
    value = unicodedata.normalize('NFKC', text or '')
    value = value.replace('−', '-').replace('–', '-').replace('×', '*')
    return re.sub(r'[\W_]+', '', value).lower()


def _printed_number(text):
    match = re.match(r'^\s*(?:第\s*)?(\d{1,3})\s*[.、．:：)）]?', str(text or ''))
    return int(match.group(1)) if match else None


def _subquestion_number(text):
    match = re.search(r'[（(\[]\s*(\d{1,2})\s*[）)\]]', str(text or ''))
    return int(match.group(1)) if match else None


def similarity(text, block):
    a, b = norm(text), norm(block.text)
    if not a or not b:
        return 0.0
    # Question and subquestion labels are structural tokens, not ordinary
    # prose. Exact label agreement is strong evidence even for "5."/"(1)".
    if _printed_number(text) is not None and _printed_number(text) == _printed_number(block.text):
        if len(a) <= 3 or len(b) <= 3:
            return 0.92
    if _subquestion_number(text) is not None and _subquestion_number(text) == _subquestion_number(block.text):
        if len(a) <= 3 or len(b) <= 3:
            return 0.90
    if min(len(a), len(b)) < 3:
        return 0.0
    if a in b or b in a:
        return min(1.0, min(len(a), len(b)) / 8.0)
    return SequenceMatcher(None, a, b).ratio()


class DocumentStructureService:
    def __init__(self, request=None, provider='none', max_attempts=2):
        self.request = request
        self.provider = provider
        self.max_attempts = max(1, min(3, max_attempts))

    @staticmethod
    def _is_full_structure(raw):
        if not isinstance(raw, dict):
            return False
        sections = raw.get('sections') or []
        return bool(sections and isinstance(sections[0], dict)
                    and 'section_id' in sections[0])

    @staticmethod
    def _response_audit(raw):
        return copy.deepcopy(getattr(raw, 'response_audit', {}) or {})

    @staticmethod
    def _failure(exc, output_dir=None, artifact_name=None):
        explicit = str(getattr(exc, 'code', '') or '')
        detail = str(getattr(exc, 'detail', '') or '')
        if explicit:
            code = explicit
        elif isinstance(exc, ValueError) and str(exc).startswith((
                'INVALID_', 'EMPTY_', 'ITEM_OUTSIDE_', 'UNKNOWN_',
                'PAGE_TEXT_', 'TOPOLOGY_')):
            code = str(exc)
            detail = str(exc)
        else:
            code = ('INVALID_STRUCTURE_RESPONSE' if isinstance(exc, ValueError)
                    else 'STRUCTURE_BACKEND_ERROR')
        failure = {'code': code, 'error_type': type(exc).__name__}
        if detail:
            failure['detail'] = detail
        response = copy.deepcopy(getattr(exc, 'response_audit', {}) or {})
        if response:
            failure['response'] = response
        raw_text = str(getattr(exc, 'raw_text', '') or '')
        if raw_text and output_dir and artifact_name:
            target = Path(output_dir) / artifact_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(raw_text, encoding='utf-8')
            failure['raw_response_path'] = str(target)
            failure['raw_response_sha256'] = hashlib.sha256(
                raw_text.encode('utf-8')).hexdigest()
            failure['raw_response_chars'] = len(raw_text)
        return failure

    @staticmethod
    def _compact_from_full(raw):
        return {
            'title': raw.get('title', ''),
            'sections': [
                {'title': section.get('title', ''), 'questions': [
                    {'number': question['number'],
                     'references': copy.deepcopy(question.get('references') or []),
                     'items': [
                         {'label': item.get('label', ''), 'type': item.get('type', 'other'),
                          'references': copy.deepcopy(item.get('references') or [])}
                         for item in question.get('items') or []]}
                    for question in section.get('questions') or []]}
                for section in raw.get('sections') or []],
            'non_question_pages': copy.deepcopy(raw.get('non_question_pages') or []),
        }

    @staticmethod
    def _normalize_topology(topology):
        """Apply deterministic exam conventions before assigning identities."""
        topology = copy.deepcopy(topology)
        for section in topology.get('sections') or []:
            for question in section.get('questions') or []:
                items = question.get('items') or []
                if (len(items) > 1 and all(item.get('type') == 'choice' for item in items)
                        and any(re.fullmatch(r'\s*[A-H][.、．)]?\s*', item.get('label', ''), re.I)
                                for item in items)):
                    refs = []
                    seen = set()
                    for item in items:
                        for ref in item.get('references') or []:
                            key = (ref.get('page_index'), ref.get('anchor'))
                            if key not in seen:
                                refs.append(copy.deepcopy(ref)); seen.add(key)
                    question['items'] = [{
                        'label': str(question.get('number', '')),
                        'type': 'choice',
                        'references': refs or copy.deepcopy(question.get('references') or []),
                    }]
        return topology

    @staticmethod
    def _merge_child_pages_into_questions(structure):
        """Make parent page ownership include every physical child page.

        A printed question number commonly appears at the bottom of one page
        while later numbered subquestions continue at the top of the next.
        The child's short anchor is the best available parent anchor on that
        continuation page.
        """
        structure = copy.deepcopy(structure)
        for section in structure.get('sections') or []:
            for question in section.get('questions') or []:
                references = question.get('references') or []
                known_pages = {ref.get('page_index') for ref in references}
                for item in question.get('items') or []:
                    for ref in item.get('references') or []:
                        page_index = ref.get('page_index')
                        if page_index not in known_pages:
                            references.append(copy.deepcopy(ref))
                            known_pages.add(page_index)
                question['references'] = references
        return structure

    @staticmethod
    def _local_ids(topology):
        counts = {}
        for section in topology['sections']:
            for question in section['questions']:
                number = question['number']
                counts[number] = counts.get(number, 0) + 1
        identifiers = {}
        used_questions = set()
        used_items = set()
        for section_index, section in enumerate(topology['sections'], 1):
            for question_index, question in enumerate(section['questions'], 1):
                number = question['number']
                base = 'q{}'.format(number) if counts[number] == 1 else 's{}_q{}'.format(
                    section_index, number)
                question_id = base
                suffix = 2
                while question_id in used_questions:
                    question_id = '{}_{}'.format(base, suffix); suffix += 1
                used_questions.add(question_id)
                item_ids = []
                for item_index, item in enumerate(question['items'], 1):
                    if len(question['items']) == 1:
                        candidate = question_id
                    else:
                        label_number = re.search(r'\d+', item.get('label', ''))
                        candidate = '{}_{}'.format(
                            question_id, label_number.group(0) if label_number else item_index)
                    if candidate in used_items:
                        candidate = '{}_{}'.format(question_id, item_index)
                    item_ids.append(candidate); used_items.add(candidate)
                identifiers[(section_index, question_index)] = (question_id, item_ids)
        return identifiers

    @classmethod
    def _topology_to_full(cls, topology, pages):
        check_schema(topology, DOCUMENT_TOPOLOGY_SCHEMA)
        topology = cls._merge_child_pages_into_questions(topology)
        page_ids = {page.index for page in pages}
        identifiers = cls._local_ids(topology)
        sections = []
        for section_index, source_section in enumerate(topology['sections'], 1):
            section = {'section_id': 'section_{}'.format(section_index),
                       'title': source_section['title'], 'questions': []}
            for question_index, source in enumerate(source_section['questions'], 1):
                question_id, item_ids = identifiers[(section_index, question_index)]
                qrefs = copy.deepcopy(source['references'])
                if any(ref['page_index'] not in page_ids for ref in qrefs):
                    raise ValueError('INVALID_PAGE_REFERENCES:' + question_id)
                qanchor = next((ref['anchor'].strip() for ref in qrefs
                                if ref['anchor'].strip()), '')
                if not qanchor:
                    raise ValueError('EMPTY_QUESTION_ANCHOR:' + question_id)
                question = {'question_id': question_id, 'number': source['number'],
                            'text': qanchor, 'references': qrefs, 'items': []}
                for item_index, source_item in enumerate(source['items']):
                    item_id = item_ids[item_index]
                    refs = copy.deepcopy(source_item['references'])
                    if any(ref['page_index'] not in page_ids for ref in refs):
                        raise ValueError('INVALID_PAGE_REFERENCES:' + item_id)
                    anchor = next((ref['anchor'].strip() for ref in refs
                                   if ref['anchor'].strip()), qanchor)
                    question['items'].append({
                        'item_id': item_id, 'label': source_item['label'],
                        'text': anchor, 'type': source_item['type'],
                        'references': refs,
                    })
                section['questions'].append(question)
            sections.append(section)
        return {'title': topology['title'], 'sections': sections,
                'non_question_pages': copy.deepcopy(topology['non_question_pages'])}

    @staticmethod
    def _numbering_gaps(raw):
        gaps = []
        for section in raw.get('sections') or []:
            questions = section.get('questions') or []
            numbers = [question.get('number') for question in questions
                       if type(question.get('number')) is int]
            for previous, current in zip(numbers, numbers[1:]):
                if previous < current - 1:
                    gaps.append({'code': 'QUESTION_NUMBER_GAP',
                                 'section': section.get('section_id', section.get('title', '')),
                                 'after': previous, 'before': current,
                                 'missing_candidates': list(range(previous + 1, current))})
        return gaps

    @staticmethod
    def _append_unique(parts, value):
        value = str(value or '').strip()
        if value and value not in parts:
            parts.append(value)

    def _enrich_topology(self, topology, pages, output_dir=None, request=None):
        request = request or self.request
        full = self._topology_to_full(topology, pages)
        page_map = {page.index: page for page in pages}
        question_map = {}
        item_map = {}
        question_parts = {}
        item_parts = {}
        for section in full['sections']:
            for question in section['questions']:
                question_map[question['question_id']] = question
                question_parts[question['question_id']] = []
                for item in question['items']:
                    item_map[item['item_id']] = item
                    item_parts[item['item_id']] = []
        failures, attempts = [], []
        for page_index, page in page_map.items():
            expected = []
            for section in full['sections']:
                for question in section['questions']:
                    if page_index not in {ref['page_index'] for ref in question['references']}:
                        continue
                    expected.append({
                        'question_id': question['question_id'],
                        'number': question['number'],
                        'items': [{'item_id': item['item_id'], 'label': item['label'],
                                   'type': item['type']}
                                  for item in question['items']
                                  if page_index in {ref['page_index'] for ref in item['references']}],
                    })
            if not expected:
                continue
            page_failure = None
            accepted = None
            for attempt_index in range(1, self.max_attempts + 1):
                prompt = load_prompt('page_text_enrichment')
                context = {'page_index': page_index, 'expected_nodes': expected,
                           'previous_failure': page_failure, 'schema': PAGE_TEXT_SCHEMA}
                record = {'phase': 'page_text_enrichment', 'page_index': page_index,
                          'attempt': attempt_index}
                try:
                    raw = request(prompt + json.dumps(context, ensure_ascii=False),
                                  [Path(page.path)], PAGE_TEXT_SCHEMA)
                    check_schema(raw, PAGE_TEXT_SCHEMA)
                    if raw['page_index'] != page_index:
                        raise ValueError('PAGE_TEXT_WRONG_PAGE')
                    expected_questions = {entry['question_id']: entry for entry in expected}
                    returned_questions = {entry['question_id']: entry for entry in raw['questions']}
                    if set(returned_questions) != set(expected_questions):
                        raise ValueError('PAGE_TEXT_QUESTION_COVERAGE')
                    for question_id, expected_question in expected_questions.items():
                        returned = returned_questions[question_id]
                        expected_items = {entry['item_id'] for entry in expected_question['items']}
                        returned_items = {entry['item_id'] for entry in returned['items']}
                        if returned_items != expected_items:
                            raise ValueError('PAGE_TEXT_ITEM_COVERAGE:' + question_id)
                        if not returned['text'].strip() or not returned['anchor'].strip():
                            raise ValueError('PAGE_TEXT_EMPTY_QUESTION:' + question_id)
                        if any(not item['text'].strip() or not item['anchor'].strip()
                               for item in returned['items']):
                            raise ValueError('PAGE_TEXT_EMPTY_ITEM:' + question_id)
                    accepted = raw
                    record.update(status='ACCEPTED', response=self._response_audit(raw))
                    attempts.append(record)
                    break
                except Exception as exc:
                    failure = self._failure(
                        exc, output_dir,
                        'raw_page_text_p{:02d}_a{}.txt'.format(page_index, attempt_index))
                    page_failure = failure
                    record.update(status='FAILED', failure=failure)
                    attempts.append(record)
            if accepted is None:
                failures.append({'code': 'PAGE_TEXT_ENRICHMENT_FAILED',
                                 'page_index': page_index,
                                 'cause': page_failure or {'code': 'UNKNOWN'}})
                continue
            for returned in accepted['questions']:
                question = question_map[returned['question_id']]
                self._append_unique(question_parts[returned['question_id']], returned['text'])
                for ref in question['references']:
                    if ref['page_index'] == page_index:
                        ref['anchor'] = returned['anchor']
                for returned_item in returned['items']:
                    item = item_map[returned_item['item_id']]
                    self._append_unique(item_parts[returned_item['item_id']], returned_item['text'])
                    for ref in item['references']:
                        if ref['page_index'] == page_index:
                            ref['anchor'] = returned_item['anchor']
        for question_id, question in question_map.items():
            if question_parts[question_id]:
                question['text'] = '\n'.join(question_parts[question_id])
        for item_id, item in item_map.items():
            if item_parts[item_id]:
                item['text'] = '\n'.join(item_parts[item_id])
        return full, failures, attempts

    @staticmethod
    def _merge_page_topologies(records):
        merged = {'title': '', 'sections': [], 'non_question_pages': []}
        section_map = {}
        for raw in records:
            if not merged['title'] and raw.get('title'):
                merged['title'] = raw['title']
            merged['non_question_pages'].extend(copy.deepcopy(raw.get('non_question_pages') or []))
            for source_section in raw.get('sections') or []:
                key = norm(source_section.get('title', ''))
                if key not in section_map:
                    target = {'title': source_section.get('title', ''), 'questions': []}
                    merged['sections'].append(target); section_map[key] = target
                target = section_map[key]
                for source_question in source_section.get('questions') or []:
                    source_pages = {ref['page_index'] for ref in source_question['references']}
                    global_matches = [
                        question
                        for candidate_section in merged['sections']
                        for question in candidate_section['questions']
                        if question['number'] == source_question['number']
                        and source_pages.intersection(
                            ref['page_index'] for ref in question['references'])
                    ]
                    existing = (global_matches[0] if len(global_matches) == 1 else
                                next((question for question in target['questions']
                                      if question['number'] == source_question['number']), None))
                    if existing is None:
                        target['questions'].append(copy.deepcopy(source_question)); continue
                    known_pages = {ref['page_index'] for ref in existing['references']}
                    existing['references'].extend(
                        copy.deepcopy(ref) for ref in source_question['references']
                        if ref['page_index'] not in known_pages)
                    for item_index, source_item in enumerate(source_question['items']):
                        item = next((candidate for candidate in existing['items']
                                     if norm(candidate['label']) == norm(source_item['label'])), None)
                        if item is None and item_index < len(existing['items']):
                            item = existing['items'][item_index]
                        if item is None:
                            existing['items'].append(copy.deepcopy(source_item)); continue
                        item_pages = {ref['page_index'] for ref in item['references']}
                        item['references'].extend(
                            copy.deepcopy(ref) for ref in source_item['references']
                            if ref['page_index'] not in item_pages)
        unique_non_questions = {}
        for entry in merged['non_question_pages']:
            unique_non_questions[(entry['source'], entry['page_index'])] = entry
        merged['non_question_pages'] = list(unique_non_questions.values())
        return DocumentStructureService._normalize_topology(merged)

    def _pagewise_topology(self, package, original_pages, output_dir=None, prior_failures=()):
        records, attempts, failures = [], [], []
        for page in original_pages:
            accepted = None
            page_failure = None
            for attempt_index in range(1, self.max_attempts + 1):
                prompt = load_prompt('page_topology_recovery')
                context = {'subject': package.subject, 'physical_page_index': page.index,
                           'whole_document_failures': list(prior_failures),
                           'previous_failure': page_failure,
                           'schema': PAGE_TOPOLOGY_SCHEMA}
                record = {'phase': 'page_topology_recovery', 'page_index': page.index,
                          'attempt': attempt_index}
                try:
                    raw = self.request(prompt + json.dumps(context, ensure_ascii=False),
                                       [Path(page.path)], PAGE_TOPOLOGY_SCHEMA)
                    compact = self._compact_from_full(raw) if self._is_full_structure(raw) else copy.deepcopy(raw)
                    check_schema(compact, PAGE_TOPOLOGY_SCHEMA)
                    compact = self._normalize_topology(compact)
                    referenced = {
                        ref['page_index']
                        for section in compact['sections']
                        for question in section['questions']
                        for ref in question['references']
                    } | {
                        ref['page_index']
                        for section in compact['sections']
                        for question in section['questions']
                        for item in question['items']
                        for ref in item['references']
                    }
                    non_question = {entry['page_index'] for entry in compact['non_question_pages']}
                    if (referenced | non_question) - {page.index}:
                        raise ValueError('INVALID_PAGE_REFERENCES:page_{}'.format(page.index))
                    accepted = compact
                    record.update(status='ACCEPTED', proposal=compact,
                                  response=self._response_audit(raw))
                    attempts.append(record)
                    break
                except Exception as exc:
                    failure = self._failure(
                        exc, output_dir,
                        'raw_page_topology_p{:02d}_a{}.txt'.format(page.index, attempt_index))
                    page_failure = failure
                    record.update(status='FAILED', failure=failure)
                    attempts.append(record)
            if accepted is None:
                failures.append({'code': 'PAGE_TOPOLOGY_RECOVERY_FAILED',
                                 'page_index': page.index,
                                 'cause': page_failure or {'code': 'UNKNOWN'}})
            else:
                records.append(accepted)
        return (self._merge_page_topologies(records) if records else None), failures, attempts

    def validate_and_convert(self, raw, pages, templates=(), validate_ocr=False):
        if validate_ocr:
            raise ValueError("本项目只支持 VLM Schema 与页覆盖校验")
        raw = copy.deepcopy(raw)
        for record in raw.get('non_question_pages', []):
            record.setdefault('source', 'original')
        check_schema(raw, DOCUMENT_STRUCTURE_SCHEMA)
        raw = self._merge_child_pages_into_questions(raw)
        page_map = {p.index: p for p in pages}
        template_map = {p.index: p for p in templates}

        # This pass creates the logical tree and page ownership. Final pixel
        # coordinates are supplied by the later page-level VLM extraction.
        if not validate_ocr:
            sections = []
            used = {'section': set(), 'question': set(), 'item': set()}

            def proposal_identity(kind, value):
                if not isinstance(value, str) or not value.strip() or value in used[kind]:
                    raise ValueError('INVALID_OR_DUPLICATE_{}_ID'.format(kind.upper()))
                used[kind].add(value)

            for source_section in raw['sections']:
                proposal_identity('section', source_section['section_id'])
                section = ExamSection(source_section['section_id'], source_section['title'], [])
                for source in source_section['questions']:
                    proposal_identity('question', source['question_id'])
                    qpages = [ref['page_index'] for ref in source['references']]
                    if not qpages or any(index not in page_map for index in qpages):
                        raise ValueError('INVALID_PAGE_REFERENCES:' + source['question_id'])
                    question = ExamQuestion(source['question_id'], source['number'], source['text'], [])
                    for child in source['items']:
                        proposal_identity('item', child['item_id'])
                        ipages = [ref['page_index'] for ref in child['references']]
                        if not ipages or any(index not in page_map for index in ipages):
                            raise ValueError('INVALID_PAGE_REFERENCES:' + child['item_id'])
                        item = ExamItem(child['item_id'], child['label'], child['text'],
                                        item_type=child['type'], confidence=0.0,
                                        is_cross_page=len(ipages) > 1)
                        item.quality['structure_references'] = copy.deepcopy(child['references'])
                        for page_index in ipages:
                            page = page_map[page_index]
                            width, height = page.width or 1654, page.height or 2338
                            # Full-page regions are temporary search contexts.
                            region = PageRegion(page_index, page.path, [0, 0, width, height], None,
                                                'deferred_ocr_search_context')
                            item.answer_regions.append(region)
                            if item.stem_region is None:
                                item.stem_region = region
                        question.items.append(item)
                    section.questions.append(question)
                sections.append(section)
            return sections, [], {
                'expected_anchors': [], 'covered_anchors': [],
                'non_question_pages': raw['non_question_pages'],
                'non_question_warnings': [], 'validation_deferred': True,
            }
        errors, sections, used = [], [], {'section': set(), 'question': set(), 'item': set()}
        covered, claimed, positions = set(), set(), []
        evidence = {(p.index, n, b.bbox[0], b.bbox[1]) for p in pages for n, b in anchors(p)}

        def issue(code, **details):
            errors.append({'code': code, **details})

        def identity(kind, value):
            if not value.strip() or value in used[kind]:
                raise ValueError('INVALID_OR_DUPLICATE_{}_ID:{}'.format(kind.upper(), value))
            used[kind].add(value)

        def references(refs, owner):
            ids = [r['page_index'] for r in refs]
            if len(ids) != len(set(ids)) or any(n not in page_map for n in ids):
                raise ValueError('INVALID_PAGE_REFERENCES:' + owner)
            order = [list(page_map).index(n) for n in ids]
            if order != sorted(order):
                issue('PAGE_ORDER_CONFLICT', owner=owner, pages=ids)
            claimed.update(ids)
            return ids

        for source_section in raw['sections']:
            identity('section', source_section['section_id'])
            section = ExamSection(source_section['section_id'], source_section['title'], [])
            for source in source_section['questions']:
                qid = source['question_id']
                identity('question', qid)
                qpages = references(source['references'], qid)
                if not source['text'].strip():
                    issue('EMPTY_QUESTION_TEXT', owner=qid)
                question = ExamQuestion(qid, source['number'], source['text'], [])
                domains = {}
                for ref_index, ref in enumerate(source['references']):
                    page = page_map[ref['page_index']]
                    groups = column_groups(page.ocr, page.width or 1654)
                    numbered = [b for n, b in anchors(page) if n == source['number']]
                    candidates = numbered if ref_index == 0 and numbered else [b for g in groups for b in g]
                    ranked = sorted(((similarity(ref['anchor'], b)
                                      + (0.12 if _printed_number(ref['anchor']) == source['number']
                                         and _printed_number(b.text) == source['number'] else 0.0), i, b)
                                     for i, b in enumerate(candidates)),
                                    key=lambda v: v[0], reverse=True)
                    anchor = ranked[0][2] if ranked and (ranked[0][0] >= .45 or len(numbered) == 1 and ref_index == 0) else None
                    if ranked and len(ranked) > 1 and ranked[0][0] - ranked[1][0] < .05:
                        anchor = None
                    if anchor is None:
                        issue('QUESTION_ANCHOR_UNRESOLVED', owner=qid, page_index=page.index)
                        continue
                    group_index, group = next((i, g) for i, g in enumerate(groups) if anchor in g)
                    start = group.index(anchor)
                    numbered_ids = {id(b) for _, b in anchors(page)}
                    end = next((i for i in range(start + 1, len(group)) if id(group[i]) in numbered_ids), len(group))
                    blocks = group[start:end]
                    bottom = group[end].bbox[1] - 4 if end < len(group) else (page.height or 2338) - 20
                    domains[page.index] = (page, blocks, max(anchor.bbox[3], bottom))
                    if anchor in numbered:
                        key = (page.index, source['number'], anchor.bbox[0], anchor.bbox[1])
                        if key in covered:
                            issue('DUPLICATE_QUESTION_ANCHOR', owner=qid, page_index=page.index)
                        covered.add(key)
                    elif ref_index == 0:
                        issue('QUESTION_NUMBER_UNCONFIRMED', owner=qid, page_index=page.index)
                    if ref_index == 0:
                        positions.append((list(page_map).index(page.index), group_index, start, qid))
                if len(source['items']) > 1 and all(c['type'] == 'choice' for c in source['items']) and any(
                        re.fullmatch(r'\s*[A-H][.、．)]?\s*', c['label']) for c in source['items']):
                    issue('OPTIONS_ARE_NOT_SUBQUESTIONS', owner=qid,
                          correction='Keep the entire multiple-choice question and all options in ONE item')
                local_starts = {}
                item_pages = set()
                for child in source['items']:
                    iid = child['item_id']
                    identity('item', iid)
                    ipages = references(child['references'], iid)
                    item_pages.update(ipages)
                    item = ExamItem(iid, child['label'], child['text'], item_type=child['type'],
                                    confidence=0.0, is_cross_page=len(ipages) > 1)
                    item.quality['structure_references'] = copy.deepcopy(child['references'])
                    if not child['text'].strip():
                        issue('EMPTY_ITEM_TEXT', owner=iid)
                    for ref in child['references']:
                        domain = domains.get(ref['page_index'])
                        if domain is None:
                            issue('ITEM_ANCHOR_UNRESOLVED', owner=iid, page_index=ref['page_index'])
                            continue
                        page, blocks, bottom = domain
                        multiline = match_lines(ref['anchor'], blocks)
                        ranked = sorted(((similarity(ref['anchor'], b), i, b) for i, b in enumerate(blocks)),
                                        key=lambda v: v[0], reverse=True)
                        if multiline:
                            ranked = [(multiline[0], multiline[1], blocks[multiline[1]])]
                        if len(source['items']) == 1 and blocks and similarity(child['text'], blocks[0]) >= .45:
                            ranked = [(max(.8, ranked[0][0] if ranked else 0), 0, blocks[0])]
                        # A short subquestion label can be sufficient when its
                        # neighboring text and reading position are unique.
                        short_structural = (_subquestion_number(ref['anchor']) is not None
                                            and ranked and ranked[0][0] >= .86)
                        if not ranked or (ranked[0][0] < .45 and not short_structural) or (len(ranked) > 1 and ranked[0][0] - ranked[1][0] < .05 and not short_structural):
                            issue('ITEM_ANCHOR_UNRESOLVED', owner=iid, page_index=page.index)
                            continue
                        score, start, block = ranked[0]
                        if any(old_start == start for old_start, _ in local_starts.get(page.index, [])):
                            # Inline subquestions may share an OCR line. Do not invent a split.
                            issue('SHARED_ITEM_OCR_ANCHOR', owner=iid, page_index=page.index)
                        local_starts.setdefault(page.index, []).append((start, item))
                        if item.stem_region is None:
                            item.stem_region = PageRegion(page.index, page.path, list(block.bbox), score, block.text)
                        item.confidence = score
                    question.items.append(item)
                if set(qpages) != item_pages:
                    issue('QUESTION_PAGE_WITHOUT_ITEM', owner=qid, pages=sorted(set(qpages) - item_pages))
                for page_index, starts in local_starts.items():
                    page, blocks, bottom = domains[page_index]
                    if [start for start, _ in starts] != sorted(start for start, _ in starts):
                        issue('ITEM_READING_ORDER_CONFLICT', owner=qid, page_index=page_index)
                    starts.sort(key=lambda pair: pair[0])
                    for i, (start, item) in enumerate(starts):
                        end = starts[i + 1][0] if i + 1 < len(starts) else len(blocks)
                        span = blocks[start:max(start + 1, end)]
                        lower = blocks[end].bbox[1] - 4 if end < len(blocks) else bottom
                        box = [max(0, min(b.bbox[0] for b in span) - 12), blocks[start].bbox[1],
                               min(page.width or 1654, max(b.bbox[2] for b in span) + 12),
                               min(page.height or 2338, max(blocks[start].bbox[3], lower))]
                        item.answer_regions.append(PageRegion(page.index, page.path, box, None, ''))
                        item.quality['structure_geometry_role'] = 'ocr_search_domain_only'
                        if item.item_type == 'choice':
                            # Preserve option image evidence for graphical
                            # choices.  A/B/C/D labels alone are insufficient
                            # when the option contains a diagram.
                            option_blocks = [b for b in span
                                             if re.match(r'^\s*[A-H][.、．)）]', b.text or '')]
                            for option in option_blocks:
                                option_box = [max(0, int(option.bbox[0] - 8)),
                                              max(0, int(option.bbox[1] - 4)),
                                              min(page.width or 1654, int(option.bbox[2] + 8)),
                                              min(page.height or 2338, int(option.bbox[3] + 4))]
                                item.option_regions.append(PageRegion(
                                    page.index, page.path, option_box,
                                    option.confidence, option.text,
                                    coordinate_role='option_image_search'))
                        formula_text = str(child.get('text') or '')
                        if re.search(r'[-−=^_√∑∫]|\\(?:frac|sqrt|sum|int)', formula_text):
                            observed_text = ' '.join(b.text for b in blocks)
                            item.quality['formula_audit'] = {
                                'expected_minus': '-' in formula_text or '−' in formula_text,
                                'observed_minus': '-' in observed_text or '−' in observed_text,
                                'structure_tokens': sorted(set(re.findall(
                                    r'[-−=^_√∑∫]|\\(?:frac|sqrt|sum|int)', formula_text))),
                                'status': 'NEEDS_FORMULA_REREAD' if (
                                    ('-' in formula_text or '−' in formula_text)
                                    and not ('-' in observed_text or '−' in observed_text)
                                ) else 'OBSERVED',
                            }
                # Only leading subquestion labels are evidence; coordinates and formulas are not labels.
                observed = {m.group(1) for _, blocks, _ in domains.values() for b in blocks
                            for m in [re.match(r'^\s*[（(](\d{1,2})[）)]', b.text)] if m}
                declared = {m.group(1) for c in source['items']
                            for m in [re.search(r'[（(](\d{1,2})[）)]', c['label'])] if m}
                if len(observed) >= 2 and not observed.issubset(declared):
                    issue('SUBQUESTION_COVERAGE_CONFLICT', owner=qid, missing=sorted(observed - declared))
                option_pattern = r'(?:^|\s)([A-H])[.、．)]'
                observed_options = {m for _, blocks, _ in domains.values() for b in blocks
                                    for m in re.findall(option_pattern, b.text)}
                proposed_options = set(re.findall(option_pattern, source['text'] + '\n' +
                                                  '\n'.join(c['text'] for c in source['items'])))
                if len(observed_options) >= 2 and not observed_options.issubset(proposed_options):
                    issue('OPTION_COVERAGE_CONFLICT', owner=qid, missing=sorted(observed_options - proposed_options))
                section.questions.append(question)
            sections.append(section)
        if positions != sorted(positions, key=lambda p: p[:3]):
            issue('READING_ORDER_CONFLICT')
        ignored = set()
        non_question_warnings = []
        for record in raw['non_question_pages']:
            n = record['page_index']
            source = record['source']
            if not record['reason'].strip():
                raise ValueError('INVALID_NON_QUESTION_PAGE')
            if source == 'supplementary_template':
                # Templates are auxiliary views of original pages. Their
                # page indices intentionally mirror the original document and
                # must not participate in physical-page coverage validation.
                if n not in template_map:
                    raise ValueError('INVALID_NON_QUESTION_PAGE')
                non_question_warnings.append({
                    'code': 'SUPPLEMENTARY_TEMPLATE_EXCLUDED',
                    'source': source,
                    'page_index': n,
                    'reason': record['reason'],
                })
                continue
            if n not in page_map or n in ignored or n in claimed:
                raise ValueError('INVALID_NON_QUESTION_PAGE')
            ignored.add(n)
        for page_index, number, x, y in sorted(evidence - covered):
            issue('OCR_QUESTION_NOT_COVERED', page_index=page_index, number=number, ocr_anchor_xy=[x, y])
        for page_index in page_map.keys() - claimed - ignored:
            issue('UNEXPLAINED_PAGE', page_index=page_index)
        if not evidence:
            issue('NO_OCR_NUMBER_EVIDENCE')
        return sections, errors, {'expected_anchors': [list(x) for x in sorted(evidence)],
                                 'covered_anchors': [list(x) for x in sorted(covered)],
                                 'non_question_pages': raw['non_question_pages'],
                                 'non_question_warnings': non_question_warnings}

    def generate(self, package, original_pages, validation_pages, templates=(), output_dir=None,
                 validate_ocr=False, vlm_only=True):
        paths = [Path(page.path) for page in original_pages] + [Path(page.path) for page in templates]
        views = [{'page_index': page.index, 'source': source, 'path': page.path}
                 for source, pages in [('original', original_pages),
                                       ('supplementary_template', templates)]
                 for page in pages]
        audit = {
            'mode': 'vlm_lightweight_topology_then_page_text_validation',
            'provider': self.provider, 'status': 'UNRESOLVED',
            'topology_source': 'vlm', 'coordinate_authority': 'vlm_original_page_pixels',
            'coverage_scope': 'whole_document_vlm_schema_and_page_coverage',
            'id_authority': 'local_deterministic', 'views': views, 'attempts': [],
        }
        failures, best = [], None
        transport_codes = {
            'OUTPUT_INCOMPLETE', 'INVALID_JSON', 'JSON_OBJECT_NOT_FOUND',
            'JSON_ROOT_NOT_OBJECT', 'MULTIPLE_JSON_OBJECTS',
            'INVALID_STRUCTURE_RESPONSE', 'STRUCTURE_BACKEND_ERROR',
        }
        for attempt_index in range(1, self.max_attempts + 1 if self.request else 1):
            retry_topology = True
            prompt = load_prompt('document_topology')
            context = {'subject': package.subject, 'image_order': views,
                       'validation_failures': failures, 'schema': DOCUMENT_TOPOLOGY_SCHEMA}
            if failures:
                if best:
                    context['previous_topology'] = best[5]
                if validate_ocr:
                    context['unsupported_validation_evidence'] = [
                        {'page_index': page.index, 'lines': [block.text for block in page.ocr]}
                        for page in validation_pages]
            record = {'attempt': attempt_index, 'kind': 'whole_document_topology'}
            LOGGER.info('document_structure topology_start provider=%s attempt=%s pages=%s',
                        self.provider, attempt_index, len(original_pages))
            try:
                raw = self.request(prompt + json.dumps(context, ensure_ascii=False),
                                   paths, DOCUMENT_TOPOLOGY_SCHEMA)
                response_audit = self._response_audit(raw)
                if self._is_full_structure(raw):
                    full = copy.deepcopy(raw)
                    compact = self._normalize_topology(self._compact_from_full(full))
                    check_schema(full, DOCUMENT_STRUCTURE_SCHEMA)
                    enrichment_failures, enrichment_attempts = [], []
                else:
                    compact = self._normalize_topology(raw)
                    check_schema(compact, DOCUMENT_TOPOLOGY_SCHEMA)
                    full, enrichment_failures, enrichment_attempts = self._enrich_topology(
                        compact, original_pages, output_dir, self.request)
                sections, validation_failures, coverage = self.validate_and_convert(
                    full, validation_pages, validate_ocr=validate_ocr)
                numbering_failures = self._numbering_gaps(full)
                candidate_failures = (list(validation_failures)
                                      + numbering_failures
                                      + list(enrichment_failures))
                retry_topology = bool(validation_failures or numbering_failures)
                record.update(proposal=full, topology_proposal=compact,
                              response=response_audit,
                              text_enrichment={'attempts': enrichment_attempts,
                                               'failures': enrichment_failures},
                              failures=candidate_failures)
                candidate = (full, sections, copy.deepcopy(candidate_failures), coverage,
                             attempt_index, compact)
                if best is None or len(candidate_failures) < len(best[2]):
                    best = candidate
                failures = candidate_failures
            except Exception as exc:
                failure = self._failure(
                    exc, output_dir, 'raw_topology_attempt_{}.txt'.format(attempt_index))
                failures = [failure]
                record['failures'] = failures
            audit['attempts'].append(record)
            if output_dir:
                atomic_write_json(Path(output_dir) / 'document_structure.json', audit)
            LOGGER.info('document_structure topology provider=%s attempt=%s failures=%s',
                        self.provider, attempt_index, len(failures))
            if best and (not failures or not retry_topology):
                break

        # Malformed, incomplete, or unavailable whole-document output is
        # retried page by page. Semantic contract violations stay bounded and
        # are not hidden by a second, weaker interpretation path.
        if best is None and self.request and failures and any(
                failure.get('code', '').split(':', 1)[0] in transport_codes
                for failure in failures):
            compact, page_failures, page_attempts = self._pagewise_topology(
                package, original_pages, output_dir, failures)
            audit['page_recovery_attempts'] = page_attempts
            if compact:
                try:
                    full, enrichment_failures, enrichment_attempts = self._enrich_topology(
                        compact, original_pages, output_dir, self.request)
                    sections, validation_failures, coverage = self.validate_and_convert(
                        full, validation_pages, validate_ocr=validate_ocr)
                    candidate_failures = (list(page_failures) + list(validation_failures)
                                          + self._numbering_gaps(full)
                                          + list(enrichment_failures))
                    selected = len(audit['attempts']) + 1
                    record = {'attempt': selected, 'kind': 'pagewise_topology_recovery',
                              'proposal': full, 'topology_proposal': compact,
                              'text_enrichment': {'attempts': enrichment_attempts,
                                                  'failures': enrichment_failures},
                              'failures': candidate_failures}
                    audit['attempts'].append(record)
                    best = (full, sections, copy.deepcopy(candidate_failures), coverage,
                            selected, compact)
                    failures = candidate_failures
                except Exception as exc:
                    failure = self._failure(exc, output_dir, 'raw_pagewise_merge.txt')
                    failures = list(page_failures) + [failure]

        if best:
            raw, package.sections, failures, coverage, selected, _ = best
            package.exam_title = raw['title']
            if vlm_only and not validate_ocr:
                page_ids = {page.index for page in original_pages}
                referenced = {
                    reference['page_index']
                    for section in raw['sections']
                    for question in section['questions']
                    for reference in question['references']
                } | {
                    reference['page_index']
                    for section in raw['sections']
                    for question in section['questions']
                    for item in question['items']
                    for reference in item['references']
                }
                non_question = {
                    entry['page_index'] for entry in raw['non_question_pages']
                }
                for page_index in sorted(page_ids - referenced - non_question):
                    failures.append({
                        'code': 'UNEXPLAINED_PAGE', 'page_index': page_index,
                    })
                for page_index in sorted((referenced | non_question) - page_ids):
                    failures.append({
                        'code': 'INVALID_PAGE_REFERENCE', 'page_index': page_index,
                    })
                for page_index in sorted(referenced & non_question):
                    failures.append({
                        'code': 'QUESTION_PAGE_MARKED_NON_QUESTION',
                        'page_index': page_index,
                    })
                coverage.update({
                    'validation_deferred': False,
                    'validation_mode': 'vlm_schema_page_coverage',
                    'referenced_pages': sorted(referenced),
                    'covered_pages': sorted(referenced | non_question),
                    'ocr_used': False,
                })
            audit.update(coverage)
            audit.update(
                mode=('vlm_whole_document_topology_and_page_coverage'
                      if vlm_only and not validate_ocr else audit['mode']),
                topology_source=('vlm' if validate_ocr or vlm_only else 'vlm_proposal'),
                coordinate_authority=('vlm_original_page_pixels'
                                      if vlm_only else audit['coordinate_authority']),
                coverage_scope=('whole_document_vlm_schema_and_page_coverage'
                                if vlm_only else audit['coverage_scope']),
                status=(('COMPLETE' if not failures else 'UNRESOLVED')
                        if validate_ocr or vlm_only else 'PROPOSED'),
                failures=failures, selected_attempt=selected,
                page_validation_deferred=False,
                ocr_used=False if vlm_only else bool(validate_ocr),
            )
        else:
            package.sections = []
            audit['failures'] = failures or [{'code': 'STRUCTURE_BACKEND_UNAVAILABLE'}]
        package.structure_audit.update(audit)
        package.topology_locked = False
        if audit['status'] == 'UNRESOLVED':
            package.warnings.append(
                '全卷视觉题目树验证未通过，保留草稿及失败原因；不会自动锁定为可靠模板')
        elif audit['status'] == 'PROPOSED':
            package.warnings.append('全卷视觉题目树已生成，等待页级 VLM 提取')
        if output_dir:
            atomic_write_json(Path(output_dir) / 'document_structure.json', audit)
        return audit

    def revalidate(self, package, validation_pages, output_dir=None,
                   request=None, original_pages=()):
        raise RuntimeError("VLM-only 流程不支持 OCR 二次校验")
        """Removed OCR revalidation implementation retained below for schema migration."""
        audit = package.structure_audit or {}
        pending_proposals = {
            item.item_id: copy.deepcopy(item.quality.get('_pending_slot_proposal'))
            for section in package.sections for question in section.questions
            for item in question.items if item.quality.get('_pending_slot_proposal')
        }
        attempts = audit.get('attempts') or []
        selected = audit.get('selected_attempt')
        selected_record = next((entry for entry in attempts
                                if entry.get('attempt') == selected), None)
        raw = selected_record.get('proposal') if selected_record else None
        if not isinstance(raw, dict):
            audit.update({'status': 'UNRESOLVED', 'topology_source': 'vlm_unresolved',
                          'retired_validation': {'status': 'UNRESOLVED',
                                             'failures': [{'code': 'MISSING_VLM_PROPOSAL'}]}})
            package.structure_audit = audit
            return audit

        sections, validation_failures, coverage = self.validate_and_convert(raw, validation_pages)
        enrichment_failures = list(
            (selected_record.get('text_enrichment') or {}).get('failures') or [])
        failures = list(validation_failures) + self._numbering_gaps(raw) + enrichment_failures
        compact = (selected_record.get('topology_proposal')
                   or self._compact_from_full(raw))
        best = (raw, sections, copy.deepcopy(failures), coverage, selected or 1, compact)
        retry_records = []
        active_request = request or self.request

        if failures and active_request and original_pages:
            views = [{'page_index': page.index, 'source': 'original', 'path': page.path}
                     for page in original_pages]
            paths = [Path(page.path) for page in original_pages]
            prompt = load_prompt('topology_correction')
            record = {'attempt': len(audit.get('attempts', [])) + 1,
                      'kind': 'post_ocr_topology_correction'}
            context = {
                'subject': package.subject, 'image_order': views,
                'validation_failures': failures, 'previous_topology': compact,
                'unsupported_validation_evidence': [
                    {'page_index': page.index, 'lines': [block.text for block in page.ocr]}
                    for page in validation_pages],
                'schema': DOCUMENT_TOPOLOGY_SCHEMA,
            }
            try:
                corrected = active_request(prompt + json.dumps(context, ensure_ascii=False),
                                           paths, DOCUMENT_TOPOLOGY_SCHEMA)
                if self._is_full_structure(corrected):
                    corrected_full = copy.deepcopy(corrected)
                    corrected_compact = self._normalize_topology(
                        self._compact_from_full(corrected_full))
                    enrichment_failures, enrichment_attempts = [], []
                else:
                    corrected_compact = self._normalize_topology(corrected)
                    check_schema(corrected_compact, DOCUMENT_TOPOLOGY_SCHEMA)
                    corrected_full, enrichment_failures, enrichment_attempts = self._enrich_topology(
                        corrected_compact, original_pages, output_dir, active_request)
                retry_sections, retry_validation, retry_coverage = self.validate_and_convert(
                    corrected_full, validation_pages)
                retry_failures = (list(retry_validation)
                                  + self._numbering_gaps(corrected_full)
                                  + list(enrichment_failures))
                record.update(proposal=corrected_full,
                              topology_proposal=corrected_compact,
                              response=self._response_audit(corrected),
                              text_enrichment={'attempts': enrichment_attempts,
                                               'failures': enrichment_failures},
                              failures=retry_failures)
                candidate = (corrected_full, retry_sections, copy.deepcopy(retry_failures),
                             retry_coverage, record['attempt'], corrected_compact)
                if len(retry_failures) < len(best[2]):
                    best = candidate
                failures = retry_failures
            except Exception as exc:
                failure = self._failure(
                    exc, output_dir,
                    'raw_post_ocr_correction_{}.txt'.format(record['attempt']))
                record['failures'] = [failure]
            retry_records.append(record)
            audit.setdefault('attempts', []).append(record)

            # If the compact correction still misses an OCR-supported question
            # or has a numbering gap, re-read only implicated physical pages and
            # merge any newly observed nodes into the locally-owned topology.
            topology_recovery_codes = {
                'OCR_QUESTION_NOT_COVERED', 'QUESTION_NUMBER_GAP',
                'SUBQUESTION_COVERAGE_CONFLICT', 'QUESTION_PAGE_WITHOUT_ITEM',
                'UNEXPLAINED_PAGE',
            }
            topology_failures = [failure for failure in best[2]
                                 if failure.get('code') in topology_recovery_codes]
            if topology_failures:
                affected = set()
                for failure in topology_failures:
                    if type(failure.get('page_index')) is int:
                        affected.add(failure['page_index'])
                    affected.update(page for page in failure.get('pages', []) if type(page) is int)
                if not affected:
                    affected = {page.index for page in original_pages}
                recovery_pages = [page for page in original_pages if page.index in affected]
                patch_topology, page_failures, page_attempts = self._pagewise_topology(
                    package, recovery_pages, output_dir, topology_failures)
                audit.setdefault('page_recovery_attempts', []).extend(page_attempts)
                if patch_topology:
                    try:
                        merged = self._merge_page_topologies([best[5], patch_topology])
                        merged_full, merge_enrichment_failures, merge_enrichment_attempts = self._enrich_topology(
                            merged, original_pages, output_dir, active_request)
                        merge_sections, merge_validation, merge_coverage = self.validate_and_convert(
                            merged_full, validation_pages)
                        merge_failures = (list(page_failures) + list(merge_validation)
                                          + self._numbering_gaps(merged_full)
                                          + list(merge_enrichment_failures))
                        merge_record = {
                            'attempt': len(audit.get('attempts', [])) + 1,
                            'kind': 'affected_page_topology_recovery',
                            'proposal': merged_full, 'topology_proposal': merged,
                            'text_enrichment': {'attempts': merge_enrichment_attempts,
                                                'failures': merge_enrichment_failures},
                            'failures': merge_failures,
                        }
                        audit['attempts'].append(merge_record)
                        retry_records.append(merge_record)
                        candidate = (merged_full, merge_sections, copy.deepcopy(merge_failures),
                                     merge_coverage, merge_record['attempt'], merged)
                        if len(merge_failures) < len(best[2]):
                            best = candidate
                    except Exception as exc:
                        audit.setdefault('page_recovery_failures', []).append(
                            self._failure(exc, output_dir, 'raw_page_recovery_merge.txt'))

        raw, sections, failures, coverage, selected, _ = best
        if sections:
            package.sections = sections
            for section in package.sections:
                for question in section.questions:
                    for item in question.items:
                        pending = pending_proposals.get(item.item_id)
                        if pending:
                            item.quality['_pending_slot_proposal'] = pending
        audit['retired_validation'] = {
            'status': 'COMPLETE' if not failures else 'UNRESOLVED',
            'failures': failures, 'retry_count': len(retry_records), **coverage,
        }
        audit.update(coverage, failures=failures, selected_attempt=selected,
                     status='COMPLETE' if not failures else 'UNRESOLVED',
                     topology_source='vlm' if not failures else 'vlm_proposal',
                     page_validation_deferred=False)
        package.structure_audit = audit
        package.topology_locked = False
        if output_dir:
            atomic_write_json(Path(output_dir) / 'document_structure.json', audit)
        return audit
