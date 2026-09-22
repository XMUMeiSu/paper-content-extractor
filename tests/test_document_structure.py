import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from exam_pipeline.contracts import ExamPackage, Page, OCRBlock
from exam_pipeline.document_structure import DocumentStructureService


def page(index=1, text='1. Choose the correct word ( )'):
    return Page(index, 'original_{}.png'.format(index), 1000, 1400,
                [OCRBlock(text, [30, 100, 900, 135], .99),
                 OCRBlock('A. spring B. summer', [30, 180, 900, 215], .99)])


def proposal(index=1, number=1):
    anchor = '{}. Choose the correct word ( )'.format(number)
    refs = [{'page_index': index, 'anchor': anchor}]
    return {'title': 'English test', 'non_question_pages': [], 'sections': [
        {'section_id': 'reading', 'title': 'Reading', 'questions': [
            {'question_id': 'q{}'.format(number), 'number': number, 'text': anchor,
             'references': refs, 'items': [{'item_id': 'q{}'.format(number), 'label': str(number),
             'text': anchor + '\nA. spring B. summer', 'type': 'choice', 'references': copy.deepcopy(refs)}]}]}]}


def topology(index=1, numbers=(1,)):
    return {'title': 'English test', 'non_question_pages': [], 'sections': [
        {'title': 'Reading', 'questions': [
            {'number': number,
             'references': [{'page_index': index,
                             'anchor': '{}. Choose the correct word'.format(number)}],
             'items': [{'label': str(number), 'type': 'choice',
                        'references': [{'page_index': index,
                                        'anchor': '{}. Choose the correct word'.format(number)}]}]}
            for number in numbers]}]}


def page_text(index=1, numbers=(1,)):
    return {'page_index': index, 'questions': [
        {'question_id': 'q{}'.format(number),
         'text': '{}. Choose the correct word ( )'.format(number),
         'anchor': '{}. Choose the correct word ( )'.format(number),
         'items': [{'item_id': 'q{}'.format(number),
                    'text': '{}. Choose the correct word ( )\nA. spring B. summer'.format(number),
                    'anchor': '{}. Choose the correct word ( )'.format(number)}]}
        for number in numbers]}


def localization_proposal(index=1):
    return {'items': [{'item_id': 'q1', 'regions': [
        {'page_index': index, 'search_bbox': [10, 40, 980, 350]}],
        'confidence': .95}]}


def package():
    return ExamPackage('test', '', 'language', 'teacher', None, 1, [], [])


class DocumentStructureTests(unittest.TestCase):
    def test_lightweight_topology_uses_local_ids_then_page_text_enrichment(self):
        calls = []

        def request(prompt, paths, schema):
            calls.append(schema)
            return (page_text() if set(schema['properties']) == {'page_index', 'questions'}
                    else topology())

        image_only = page(); image_only.ocr = []
        pkg = package()
        audit = DocumentStructureService(request, 'fake').generate(
            pkg, [image_only], [image_only], validate_ocr=False)
        self.assertEqual(audit['status'], 'PROPOSED')
        self.assertEqual(audit['id_authority'], 'local_deterministic')
        self.assertEqual(len(calls), 2)
        question = pkg.sections[0].questions[0]
        self.assertEqual(question.question_id, 'q1')
        self.assertEqual(question.items[0].item_id, 'q1')
        self.assertIn('A. spring B. summer', question.items[0].question_text)

    def test_invalid_whole_document_json_recovers_pagewise(self):
        class InvalidJson(ValueError):
            code = 'INVALID_JSON'
            detail = 'JSON parse failed at char 42'
            response_audit = {'status': 'completed'}
            raw_text = '{"sections": ['

        calls = []

        def request(prompt, paths, schema):
            calls.append((prompt, schema))
            if len(calls) <= 2:
                raise InvalidJson()
            return (page_text() if set(schema['properties']) == {'page_index', 'questions'}
                    else topology())

        image_only = page(); image_only.ocr = []
        pkg = package()
        with tempfile.TemporaryDirectory() as tmp:
            audit = DocumentStructureService(request, 'fake').generate(
                pkg, [image_only], [image_only], output_dir=tmp, validate_ocr=False)
            self.assertEqual(audit['topology_source'], 'vlm_proposal')
            self.assertTrue(audit['page_recovery_attempts'])
            self.assertTrue((Path(tmp) / 'raw_topology_attempt_1.txt').exists())
            self.assertEqual(pkg.sections[0].questions[0].question_id, 'q1')

    def test_number_gap_retries_lightweight_topology(self):
        calls = []

        def request(prompt, paths, schema):
            calls.append(schema)
            if set(schema['properties']) == {'page_index', 'questions'}:
                numbers = (1, 3) if len([value for value in calls
                                        if set(value['properties']) != {'page_index', 'questions'}]) == 1 else (1, 2, 3)
                return page_text(numbers=numbers)
            topology_calls = len([value for value in calls
                                  if set(value['properties']) != {'page_index', 'questions'}])
            return topology(numbers=(1, 3) if topology_calls == 1 else (1, 2, 3))

        p1 = Page(1, 'original_1.png', 1000, 1400, [])
        pkg = package()
        audit = DocumentStructureService(request, 'fake').generate(
            pkg, [p1], [p1], validate_ocr=False)
        self.assertEqual(audit['status'], 'PROPOSED')
        self.assertEqual([question.question_num for question in pkg.sections[0].questions],
                         [1, 2, 3])
        self.assertIn('QUESTION_NUMBER_GAP',
                      [failure['code'] for failure in audit['attempts'][0]['failures']])

    def test_page_recovery_collapses_choice_options_and_merges_by_page_identity(self):
        initial = topology(numbers=(1, 2))
        recovered = topology(numbers=(1, 2))
        recovered['sections'][0]['title'] = 'Different visual heading'
        recovered['sections'][0]['questions'][0]['items'] = [
            {'label': label, 'type': 'choice',
             'references': [{'page_index': 1, 'anchor': label + '. option'}]}
            for label in ('A', 'B', 'C', 'D')]
        merged = DocumentStructureService._merge_page_topologies([initial, recovered])
        questions = [question for section in merged['sections']
                     for question in section['questions']]
        self.assertEqual([question['number'] for question in questions], [1, 2])
        self.assertEqual(len(questions[0]['items']), 1)
        self.assertEqual(questions[0]['items'][0]['type'], 'choice')

    def test_anchor_only_conflict_does_not_trigger_page_topology_recovery(self):
        raw = proposal()
        raw['sections'][0]['questions'][0]['references'][0]['anchor'] = 'not visible'
        raw['sections'][0]['questions'][0]['items'][0]['references'][0]['anchor'] = 'not visible'
        raw['sections'][0]['questions'][0]['items'][0]['text'] = 'unrelated visual text'
        image_only = page(); image_only.ocr = []
        pkg = package()
        service = DocumentStructureService(lambda *args: raw, 'fake')
        service.generate(pkg, [image_only], [image_only], validate_ocr=False)
        audit = service.revalidate(pkg, [page()], request=lambda *args: raw,
                                   original_pages=[image_only])
        self.assertNotIn('page_recovery_attempts', audit)
        self.assertIn('ITEM_ANCHOR_UNRESOLVED',
                      [failure['code'] for failure in audit['failures']])

    def test_visual_tree_can_be_proposed_before_ocr_validation(self):
        raw = proposal()
        image_only = page()
        image_only.ocr = []
        pkg = package()
        audit = DocumentStructureService(lambda *args: raw, 'fake').generate(
            pkg, [image_only], [image_only], validate_ocr=False)
        self.assertEqual(audit['status'], 'PROPOSED')
        self.assertTrue(audit['ocr_validation_deferred'])
        item = pkg.sections[0].questions[0].items[0]
        self.assertEqual(item.answer_regions[0].ocr_text, 'deferred_ocr_search_context')
        self.assertEqual(item.answer_regions[0].bbox, [0, 0, 1000, 1400])

    def test_post_ocr_conflict_gets_bounded_visual_retry(self):
        proposed = proposal()
        bad = copy.deepcopy(proposed)
        bad['sections'][0]['questions'][0]['number'] = 2
        bad['sections'][0]['questions'][0]['references'][0]['anchor'] = '2. Missing question'
        calls = []

        def request(prompt, paths, schema):
            calls.append(prompt)
            return bad if len(calls) == 1 else proposed

        visual_page = page(); visual_page.ocr = []
        ocr_page = page()
        pkg = package()
        service = DocumentStructureService(request, 'fake')
        service.generate(pkg, [visual_page], [visual_page], validate_ocr=False)
        audit = service.revalidate(pkg, [ocr_page], request=request,
                                   original_pages=[visual_page])
        self.assertEqual(audit['status'], 'COMPLETE')
        self.assertEqual(audit['ocr_validation']['retry_count'], 1)
        self.assertEqual(audit['selected_attempt'], 2)

    def test_whole_pages_primary_templates_supplementary_no_ocr_first_prompt(self):
        seen = []
        raw = proposal()
        second = proposal(3, 2)['sections'][0]['questions'][0]
        raw['sections'][0]['questions'].append(second)
        pages = [page(), page(3, '2. Choose the correct word ( )')]
        template = page(); template.path = 'template.png'
        def request(prompt, paths, schema):
            seen.append((prompt, paths)); return raw
        p = package()
        audit = DocumentStructureService(request, 'fake').generate(p, pages, pages, [template])
        self.assertEqual(audit['status'], 'COMPLETE')
        self.assertEqual(list(map(str, seen[0][1])), ['original_1.png', 'original_3.png', 'template.png'])
        self.assertNotIn('ocr_validation_evidence', seen[0][0])
        self.assertEqual(p.sections[0].section_id, 'reading')
        self.assertIsNone(p.sections[0].questions[0].items[0].standard_answer)
        self.assertEqual(p.sections[0].questions[1].items[0].stem_region.page_index, 3)

    def test_coverage_failure_retries_then_uses_corrected_vlm(self):
        pages = [page(), page(2, '2. Choose the correct word ( )')]
        bad = proposal(); good = copy.deepcopy(bad)
        good['sections'][0]['questions'].append(proposal(2, 2)['sections'][0]['questions'][0])
        calls = []
        def request(prompt, paths, schema):
            calls.append(prompt); return bad if len(calls) == 1 else good
        p = package(); audit = DocumentStructureService(request, 'fake').generate(p, pages, pages)
        self.assertEqual(audit['status'], 'COMPLETE'); self.assertEqual(audit['selected_attempt'], 2)
        self.assertIn('OCR_QUESTION_NOT_COVERED', calls[1])
        self.assertIn('ocr_validation_evidence', calls[1])

    def test_invalid_protocol_is_bounded_draft(self):
        for mutate in ('page', 'duplicate', 'answer', 'coordinate'):
            raw = proposal()
            q = raw['sections'][0]['questions'][0]
            if mutate == 'page': q['references'][0]['page_index'] = 99
            if mutate == 'duplicate': raw['sections'][0]['questions'].append(copy.deepcopy(q))
            if mutate == 'answer': q['items'][0]['standard_answer'] = 'B'
            if mutate == 'coordinate': q['items'][0]['bbox'] = [0, 0, 1000, 1000]
            p = package(); calls = []
            def request(*args): calls.append(1); return raw
            audit = DocumentStructureService(request, 'fake').generate(p, [page()], [page()])
            self.assertEqual(len(calls), 2)
            self.assertEqual(audit['status'], 'UNRESOLVED')
            self.assertEqual(audit['topology_source'], 'ocr_fallback_draft')
            self.assertFalse(p.topology_locked)

    def test_valid_vlm_topology_not_rewritten_on_ocr_disagreement(self):
        raw = proposal(); raw['sections'][0]['questions'][0]['items'][0]['references'][0]['anchor'] = 'unreadable anchor'; raw['sections'][0]['questions'][0]['items'][0]['text'] = 'unknown foreign prompt'
        p = package(); audit = DocumentStructureService(lambda *args: raw, 'fake').generate(p, [page()], [page()])
        self.assertEqual(audit['topology_source'], 'vlm'); self.assertEqual(audit['status'], 'UNRESOLVED')
        self.assertEqual(p.sections[0].section_id, 'reading')
        self.assertEqual(p.sections[0].questions[0].items[0].answer_regions, [])

    def test_continuation_uses_declared_page_ids_and_local_ocr(self):
        raw = proposal(); q = raw['sections'][0]['questions'][0]
        ref = {'page_index': 4, 'anchor': 'Continue explaining your reasoning'}
        q['references'].append(ref); q['items'][0]['references'].append(ref)
        p4 = Page(4, 'four.png', 1000, 1400, [OCRBlock(ref['anchor'], [40, 90, 850, 130], .99)])
        p = package(); audit = DocumentStructureService(lambda *a: raw, 'fake').generate(p, [page(), p4], [page(), p4])
        self.assertEqual(audit['status'], 'COMPLETE')
        item = p.sections[0].questions[0].items[0]
        self.assertTrue(item.is_cross_page)
        self.assertEqual([r.page_index for r in item.answer_regions], [1, 4])
        self.assertEqual(item.answer_regions[1].bbox[1], 90)

    def test_subquestions_require_coverage_and_distinct_local_anchors(self):
        p1 = page(); p1.ocr.extend([OCRBlock('(1) Explain the first reason', [30, 300, 900, 335], .99),
                                   OCRBlock('(2) Explain the second reason', [30, 500, 900, 535], .99)])
        _, failures, _ = DocumentStructureService().validate_and_convert(proposal(), [p1])
        self.assertIn('SUBQUESTION_COVERAGE_CONFLICT', [e['code'] for e in failures])

    def test_numbering_restart_across_sections_is_preserved(self):
        raw = proposal(); section = copy.deepcopy(proposal(2)['sections'][0]); section['section_id'] = 'writing'
        section['questions'][0]['question_id'] = 'writing_q1'; section['questions'][0]['items'][0]['item_id'] = 'writing_q1'
        raw['sections'].append(section)
        p = package(); audit = DocumentStructureService(lambda *a: raw, 'fake').generate(p, [page(), page(2)], [page(), page(2)])
        self.assertEqual(audit['status'], 'COMPLETE'); self.assertEqual(len(p.sections), 2)

    def test_backend_failure_does_not_leak_response_details(self):
        def request(*args): raise RuntimeError('secret provider body')
        p = package()
        with tempfile.TemporaryDirectory() as tmp:
            audit = DocumentStructureService(request, 'fake').generate(p, [page()], [page()], output_dir=tmp)
            self.assertNotIn('secret', json.dumps(audit))
            self.assertTrue((Path(tmp) / 'document_structure.json').exists())
            self.assertEqual(audit['failures'][0]['code'], 'STRUCTURE_BACKEND_ERROR')

    def test_repeated_number_on_same_page_cannot_hide_missing_question(self):
        p1 = page(); p1.ocr.append(OCRBlock('1. Explain another subject', [30, 600, 900, 635], .99))
        _, errors, _ = DocumentStructureService().validate_and_convert(proposal(), [p1])
        missing = [e for e in errors if e['code'] == 'OCR_QUESTION_NOT_COVERED']
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]['ocr_anchor_xy'], [30, 600])

    def test_choice_options_cannot_become_leaf_items(self):
        raw = proposal(); q = raw['sections'][0]['questions'][0]
        first = copy.deepcopy(q['items'][0]); first['label'] = 'A.'
        second = copy.deepcopy(first); second['label'] = 'B.'; second['item_id'] = 'q1_b'
        q['items'] = [first, second]
        _, errors, _ = DocumentStructureService().validate_and_convert(raw, [page()])
        self.assertIn('OPTIONS_ARE_NOT_SUBQUESTIONS', [e['code'] for e in errors])

    def test_option_loss_is_rejected(self):
        raw = proposal(); raw['sections'][0]['questions'][0]['items'][0]['text'] = 'Choose a word'
        _, errors, _ = DocumentStructureService().validate_and_convert(raw, [page()])
        self.assertIn('OPTION_COVERAGE_CONFLICT', [e['code'] for e in errors])

    def test_reading_order_conflict_is_not_silently_sorted(self):
        raw = proposal(); raw['sections'][0]['questions'].insert(0, proposal(2, 2)['sections'][0]['questions'][0])
        sections, errors, _ = DocumentStructureService().validate_and_convert(raw, [page(), page(2, '2. Choose the correct word ( )')])
        self.assertIn('READING_ORDER_CONFLICT', [e['code'] for e in errors])
        self.assertEqual(sections[0].questions[0].question_num, 2)

    def test_ocr_empty_keeps_visual_structure_unresolved(self):
        p1 = page(); p1.ocr = []
        p = package(); audit = DocumentStructureService(lambda *a: proposal(), 'fake').generate(p, [p1], [p1])
        self.assertEqual(audit['topology_source'], 'vlm')
        self.assertEqual(audit['status'], 'UNRESOLVED')
        self.assertIn('NO_OCR_NUMBER_EVIDENCE', [e['code'] for e in audit['failures']])

    def test_good_draft_survives_failed_retry(self):
        p1 = page(); p1.ocr = []; calls = []
        def request(*args):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('network')
            return proposal()
        p = package(); audit = DocumentStructureService(request, 'fake').generate(p, [p1], [p1])
        self.assertEqual(audit['selected_attempt'], 1)
        self.assertEqual(audit['topology_source'], 'vlm')
        self.assertEqual(audit['attempts'][1]['failures'][0]['code'], 'STRUCTURE_BACKEND_ERROR')

    def test_runtime_vlm_primary_called_once_and_student_inherits(self):
        import homework_extractor as runtime
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'; output = Path(tmp) / 'out'
            for role in ('teacher', 'student001'):
                folder = root / 'language' / role; folder.mkdir(parents=True)
                Image.new('RGB', (1000, 1400), 'white').save(folder / 'page_01.jpg')
            def make_pages(paths, *args):
                p = page(); p.path = str(paths[0]); return [p]
            def visual_response(*args):
                schema = args[-1]
                return localization_proposal() if set(schema.get('properties', {})) == {'items'} else proposal()
            with patch.object(runtime, 'make_pages', side_effect=make_pages), \
                 patch.object(runtime, 'call_doubao', side_effect=visual_response) as request, \
                 patch('exam_pipeline.subitems.FineGrainedItemSplitter.enrich_package', side_effect=AssertionError('must not rewrite VLM')), \
                 patch('exam_pipeline.golden.GoldenTemplateService.seed_regions_from_ocr', side_effect=AssertionError('must not reassign anchors')):
                result = runtime.process(root, output, 'fake', 'test-key', 'https://example.invalid', 'none', None,
                    False, False, 'eng', 1, local_ocr=False,
                    structure_vlm='auto', slot_semantics='none')
            self.assertEqual(result['errors'], [])
            self.assertEqual(request.call_count, 3)
            teacher = json.loads((output / 'language__teacher__teacher.json').read_text())
            student = json.loads((output / 'language__student__student001.json').read_text())
            self.assertEqual(teacher['structure_audit']['topology_source'], 'vlm')
            self.assertEqual(teacher['sections'][0]['section_id'], 'reading')
            self.assertEqual(student['sections'][0]['section_id'], 'reading')
            order = result['runtime']['pipeline_order']
            self.assertLess(order.index('question_tree_generation'), order.index('ocr_vlm_cross_validation'))
            phases = [entry['phase'] for entry in teacher['phase_trace']]
            self.assertLess(phases.index('question_tree_generation'), phases.index('ocr_vlm_cross_validation'))
            self.assertIn('slot_semantic_proposal', phases)
            self.assertIn('ocr_coordinate_grounding', phases)
