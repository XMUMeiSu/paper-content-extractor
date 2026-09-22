import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from exam_pipeline.contracts import (
    ExamItem, ExamPackage, ExamQuestion, ExamSection, Page, PageRegion, Slot,
)
from exam_pipeline.exam_tree import ExamTreeService
from exam_pipeline.visual_extraction import VisualExamExtractionService


def make_page(root, index=1, width=1000, height=1400):
    path = root / 'page_{:02d}.jpg'.format(index)
    Image.new('RGB', (width, height), 'white').save(path)
    return Page(
        index, str(path), width, height, [], document_id='doc',
        physical_page_id='doc:p{:04d}'.format(index), page_index=index,
    )


def make_package(role='teacher', item_count=1):
    items = []
    for index in range(1, item_count + 1):
        item = ExamItem(
            'q1_{}'.format(index), '1.({})'.format(index),
            'Compute part {}'.format(index), item_type='fill',
        )
        item.quality['structure_references'] = [
            {'page_index': 1, 'anchor': 'part {}'.format(index)}
        ]
        items.append(item)
    package = ExamPackage(
        'doc', 'Exam', 'math', role,
        'student001' if role == 'student' else None,
        1, [], [ExamSection('s1', 'Section', [
            ExamQuestion('q1', 1, 'Question', items)
        ])],
    )
    package.structure_audit = {
        'status': 'COMPLETE', 'topology_source': 'vlm',
        'validation_mode': 'vlm_schema_page_coverage', 'ocr_used': False,
    }
    return package


def item_response(page, item, *, text='42', bbox=None, label='formula_response',
                  content_kind='handwriting', legible=True, regions=None,
                  diagrams=None):
    if regions is None:
        regions = [{
            'answer_bbox': bbox or [500, 200, 680, 250],
            'transcription': text,
            'legible': legible,
            'content_kind': content_kind,
            'confidence': .94,
        }]
    return {
        'item_id': item.item_id,
        'physical_page_id': page.physical_page_id,
        'page_index': page.index,
        'question_region': [50, 100, 900, 350],
        'answer_layout': 'single',
        'slots': [{
            'index': 1,
            'label': label,
            'anchor_before': 'x=',
            'anchor_after': '',
            'regions': regions,
            'confidence': .95,
        }],
        'diagram_regions': diagrams or [],
        'confidence': .96,
    }


class VisualExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.page = make_page(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def run_service(self, package, responses):
        calls = []

        def request(prompt, images, schema):
            calls.append((prompt, images, schema))
            value = responses[min(len(calls) - 1, len(responses) - 1)]
            if isinstance(value, Exception):
                raise value
            return copy.deepcopy(value)

        summary = VisualExamExtractionService(request, 'test').extract(
            package, [self.page], self.root / 'visual'
        )
        return summary, calls

    def test_teacher_extracts_final_slot_and_answer_without_ocr(self):
        package = make_package()
        item = package.sections[0].questions[0].items[0]
        summary, calls = self.run_service(
            package, [{'items': [item_response(self.page, item)]}]
        )
        slot = item.slots[0]
        self.assertEqual(slot.expected_text, '42')
        self.assertEqual(slot.expected_bbox, [280, 500, 350, 680])
        self.assertEqual(slot.audit['coordinate_authority'],
                         'vlm_original_page_pixels')
        self.assertFalse(summary['ocr_used'])
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.page.ocr, [])

    def test_student_preserves_teacher_topology_but_reads_own_answer(self):
        package = make_package('student')
        item = package.sections[0].questions[0].items[0]
        item.expected_slot_count = 1
        item.semantic_slot_plan = [{
            'slot_id': item.item_id + ':slot:1', 'index': 1,
            'label': 'formula_response', 'anchor_before': 'x=',
            'anchor_after': '', 'expected_text': '42',
        }]
        self.run_service(
            package, [{'items': [item_response(self.page, item, text='41')]}]
        )
        slot = item.slots[0]
        self.assertEqual(slot.student_answer, '41')
        self.assertEqual(slot.expected_text, '42')
        self.assertEqual(item.expected_slot_count, 1)

    def test_stale_physical_page_identity_is_normalized_by_context(self):
        package = make_package()
        item = package.sections[0].questions[0].items[0]
        invalid = item_response(self.page, item)
        invalid['physical_page_id'] = 'wrong-page'
        valid = item_response(self.page, item)
        summary, calls = self.run_service(
            package, [{'items': [invalid]}, {'items': [valid]}]
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(summary['page_retries'], 0)
        self.assertEqual(item.slots[0].expected_text, '42')

    def test_invalid_page_index_retries_current_page(self):
        package = make_package()
        item = package.sections[0].questions[0].items[0]
        invalid = item_response(self.page, item)
        invalid['page_index'] = 2
        valid = item_response(self.page, item)
        summary, calls = self.run_service(
            package, [{'items': [invalid]}, {'items': [valid]}]
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(summary['page_retries'], 1)
        self.assertEqual(item.slots[0].expected_text, '42')

    def test_invalid_box_is_retried_then_marked_unresolved(self):
        package = make_package()
        item = package.sections[0].questions[0].items[0]
        invalid = item_response(self.page, item)
        invalid['slots'][0]['regions'][0]['answer_bbox'] = [500, 200, 1100, 250]
        summary, calls = self.run_service(
            package, [{'items': [invalid]}, {'items': [invalid]}]
        )
        self.assertEqual(len(calls), 3)
        self.assertEqual(summary['failed_pages'], 1)
        self.assertEqual(item.slots[0].geometry_status, 'MISSING')

    def test_overlapping_sibling_answers_do_not_discard_page(self):
        package = make_package(item_count=2)
        first, second = package.sections[0].questions[0].items
        overlap = {'items': [
            item_response(self.page, first, bbox=[450, 200, 650, 260]),
            item_response(self.page, second, bbox=[500, 210, 700, 270]),
        ]}
        corrected = {'items': [
            item_response(self.page, first, bbox=[450, 200, 650, 260]),
            item_response(self.page, second, bbox=[450, 500, 650, 560]),
        ]}
        summary, calls = self.run_service(package, [overlap, corrected])
        self.assertEqual(len(calls), 1)
        self.assertEqual(summary['page_retries'], 0)
        self.assertEqual(summary['accepted_items'], 2)

    def test_formula_blank_multiregion_and_diagram_are_preserved(self):
        formula = make_package()
        formula_item = formula.sections[0].questions[0].items[0]
        latex = r'\frac{x_1^2-1}{\sqrt{3}}'
        self.run_service(formula, [{'items': [item_response(
            self.page, formula_item, text=latex,
            diagrams=[[100, 400, 300, 600]],
        )]}])
        self.assertEqual(formula_item.slots[0].expected_text, latex)
        self.assertEqual(len(formula_item.diagrams), 1)
        self.assertEqual(formula_item.diagrams[0].bbox, [100, 560, 300, 840])

        blank = make_package()
        blank_item = blank.sections[0].questions[0].items[0]
        self.run_service(blank, [{'items': [item_response(
            self.page, blank_item, text='', content_kind='blank',
        )]}])
        self.assertEqual(blank_item.slots[0].content_status, 'BLANK')
        self.assertTrue(blank_item.slots[0].expected_bbox)

        multi = make_package()
        multi_item = multi.sections[0].questions[0].items[0]
        regions = [
            {'answer_bbox': [200, 400, 700, 500], 'transcription': 'line 1',
             'legible': True, 'content_kind': 'handwriting', 'confidence': .9},
            {'answer_bbox': [200, 520, 700, 620], 'transcription': 'line 2',
             'legible': True, 'content_kind': 'handwriting', 'confidence': .9},
        ]
        self.run_service(multi, [{'items': [item_response(
            self.page, multi_item, regions=regions,
        )]}])
        self.assertEqual(len(multi_item.slots), 2)
        self.assertEqual(len({slot.semantic_id for slot in multi_item.slots}), 1)

    def test_empty_region_gets_bounded_item_focus_retry(self):
        package = make_package()
        item = package.sections[0].questions[0].items[0]
        first = item_response(self.page, item, text='', content_kind='blank')
        first['slots'][0]['regions'] = []
        second = item_response(self.page, item, text='visible work')
        summary, calls = self.run_service(
            package, [{'items': [first]}, {'items': [second]}]
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(summary['page_retries'], 1)
        self.assertEqual(item.slots[0].recognized_text, 'visible work')

    def test_exam_tree_roundtrip_preserves_page_references(self):
        package = make_package()
        item = package.sections[0].questions[0].items[0]
        item.quality['structure_references'] = [
            {'page_index': 1, 'anchor': 'exact page phrase'}
        ]
        tree = ExamTreeService.compile(package)
        restored = ExamPackage(
            'other', '', 'math', 'teacher', None, 1, [self.page.path], []
        )
        ExamTreeService.apply_to_package(tree, restored)
        restored_item = restored.sections[0].questions[0].items[0]
        self.assertEqual(
            restored_item.quality['structure_references'],
            [{'page_index': 1, 'anchor': 'exact page phrase'}],
        )

    def test_vlm_only_pipeline_never_runs_ocr_backend(self):
        import homework_extractor as runtime
        from exam_pipeline.document_structure import DocumentStructureService

        dataset = self.root / 'dataset'
        output = self.root / 'output'
        for role in ('teacher', 'student001'):
            folder = dataset / 'math' / role
            folder.mkdir(parents=True)
            Image.new('RGB', (600, 800), 'white').save(folder / 'page_01.jpg')

        ocr_engines = []

        def ocr_spy(path, engine, language):
            ocr_engines.append(engine)
            if engine != 'none':
                raise AssertionError('OCR backend must be disabled in VLM-only mode')
            return []

        def generate(service, package, original_pages, validation_pages,
                     templates=(), output_dir=None, validate_ocr=True,
                     vlm_only=False):
            self.assertTrue(vlm_only)
            item = ExamItem('q1', '1', '1. Compute x', item_type='fill')
            item.quality['structure_references'] = [
                {'page_index': 1, 'anchor': 'Compute x'}
            ]
            package.sections = [ExamSection('s1', 'Section', [
                ExamQuestion('q1', 1, 'Compute x', [item])
            ])]
            package.structure_audit = {
                'status': 'COMPLETE', 'topology_source': 'vlm',
                'validation_mode': 'vlm_schema_page_coverage',
                'failures': [], 'ocr_used': False,
            }
            return package.structure_audit

        def extract(service, package, pages, output_dir):
            self.assertEqual(pages[0].ocr, [])
            item = package.sections[0].questions[0].items[0]
            expected = next((slot.expected_text for slot in item.slots
                             if slot.expected_text), '42')
            text = expected if package.document_type == 'teacher' else '41'
            semantic_id = item.item_id + ':slot:1'
            slot = Slot(1, 'semantic_region', item.item_id,
                        [200, 300, 250, 430], 1,
                        expected_text=expected,
                        geometry_status='ALIGNED', content_status='RECOGNIZED',
                        semantic_id=semantic_id, anchor_before='x=',
                        anchor_after='')
            slot.handwriting_bbox = list(slot.expected_bbox)
            slot.evidence_bbox = list(slot.expected_bbox)
            slot.recognition_bbox = list(slot.expected_bbox)
            slot.recognized_text = text
            slot.status = ('TEACHER_ANSWER_EXTRACTED'
                           if package.document_type == 'teacher'
                           else 'VLM_ANSWER_EXTRACTED')
            slot.review_status = 'AUTO_PASS'
            slot.audit = {
                'coordinate_authority': 'vlm_original_page_pixels',
                'topology_source': ('teacher_vlm' if package.document_type == 'teacher'
                                    else 'student_self'),
                'recognition': {'issues': [], 'warnings': [],
                                'agreement_type': 'VLM_DIRECT'},
            }
            slot.answer_fragments = [{
                'page_index': 1, 'page_file': pages[0].path,
                'bbox': [300, 200, 430, 250], 'text': text,
            }]
            if package.document_type == 'student':
                slot.student_answer = text
            item.slots = [slot]
            item.expected_slot_count = 1
            item.semantic_slot_plan = [{
                'slot_id': semantic_id, 'index': 1,
                'label': 'formula_response', 'anchor_before': 'x=',
                'anchor_after': '', 'expected_text': expected,
            }]
            item.slot_semantics_audit = {
                'status': 'ACCEPTED',
                'coordinate_authority': 'vlm_original_page_pixels',
            }
            item.cardinality_evidence = {
                'decision': 'VLM_DIRECT', 'resolved_count': 1,
            }
            item.stem_region = PageRegion(
                1, pages[0].path, [50, 100, 550, 300], .95,
                coordinate_role='vlm_question_region')
            answer_region = PageRegion(
                1, pages[0].path, [300, 200, 430, 250], .95,
                coordinate_role='vlm_final_answer_region')
            item.answer_regions = [answer_region]
            item.student_regions = ([copy.deepcopy(answer_region)]
                                    if package.document_type == 'student' else [])
            return {
                'page_calls': 1, 'page_retries': 0, 'failed_pages': 0,
                'items': 1, 'slots': 1, 'regions': 1,
                'accepted_items': 1, 'partial_items': 0,
                'coordinate_authority': 'vlm_original_page_pixels',
                'ocr_used': False,
            }

        with patch.object(runtime, 'ocr_page', side_effect=ocr_spy), \
             patch.object(DocumentStructureService, 'generate', generate), \
             patch.object(VisualExamExtractionService, 'extract', extract):
            manifest = runtime.process(
                dataset, output, 'fake-model', 'fake-key',
                'https://example.invalid', 'paddle', None,
                False, False, 'eng', 1, local_ocr=True,
                structure_vlm='doubao', slot_semantics='doubao',
            )

        self.assertEqual(manifest['errors'], [])
        self.assertEqual(ocr_engines, ['none', 'none'])
        self.assertEqual(manifest['runtime']['recognition_mode'], 'vlm_only')
        for result_path in output.glob('math__*.json'):
            import json
            result = json.loads(result_path.read_text(encoding='utf-8'))
            self.assertEqual(result['ocr'], [])
            self.assertFalse(result['ocr_used'])
            phases = {entry['phase'] for entry in result['phase_trace']}
            self.assertIn('visual_page_extraction', phases)
            self.assertNotIn('ocr_vlm_cross_validation', phases)


if __name__ == '__main__':
    unittest.main()
