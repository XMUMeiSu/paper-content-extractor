import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cv2
from exam_pipeline.contracts import Page, OCRBlock, ExamItem, ExamQuestion, ExamSection, ExamPackage, Slot, PageRegion
from exam_pipeline.result_contract import finalize_answers, valid_box
from exam_pipeline.reading_order import row_order, ordered_slots
from exam_pipeline.structuring import order_blocks_column_aware
from exam_pipeline.structure_recovery import build_sections, select_structure_pages, recover_structure
from exam_pipeline.answer_recognition import AnswerRecognizer, content_issues
from exam_pipeline.answer_parser import parse_student_answer
from exam_pipeline.exam_tree import ExamTreeService
from exam_pipeline.cohort_consensus import CohortConsensusBuilder
from exam_pipeline.teacher_answer_quality import TeacherAnswerQualityAssessor
from exam_pipeline.teacher_answers import TeacherAnswerExtractionService
from exam_pipeline.student_answer_localization import StudentAnswerLocalizationService
from exam_pipeline.subitems import FineGrainedItemSplitter
from exam_pipeline.quality import validate_item_regions


def package(items, role='student', subject='generic'):
    return ExamPackage('exam','',subject,role,None,2,['one.png','two.png'],
                       [ExamSection('s','',[ExamQuestion('q1',1,'Question',items)])])


def page(index=1, blocks=None):
    return Page(index, 'one.png' if index==1 else 'two.png', 1000,1400,blocks or [])


class ContractTests(unittest.TestCase):
    def test_multislot_nulls_and_authoritative_regions(self):
        a=Slot(1,'fill','q1',[100,50,140,120],1,student_answer='x',has_ink=True,
               handwriting_bbox=[100,50,140,120],geometry_status='ALIGNED',content_status='RECOGNIZED')
        b=Slot(2,'fill','q1',[],2,content_status='NOT_EVALUATED',geometry_status='MISSING')
        item=ExamItem('q1','q1',slots=[a,b],student_answer='contaminated stem',expected_slot_count=2)
        p=package([item]);finalize_answers(p,[page(),page(2)])
        self.assertEqual(item.student_answer,['x',None]);self.assertEqual(item.answer_status,'PARTIAL')
        self.assertEqual(item.student_regions[0].bbox,[50,100,120,140])
        self.assertEqual(item.student_regions[0].page_file,'one.png')
        self.assertEqual(p.extraction_status,'PARTIAL')

    def test_pages_are_ids_not_offsets(self):
        slot=Slot(1,'fill','q1',[100,50,140,120],2)
        item=ExamItem('q1','q1',slots=[slot])
        p=package([item]);service=StudentAnswerLocalizationService()
        with patch.object(service,'refine_student_regions',return_value=[]) as refine:
            service.localize(p,[page(),page(2)],[], 'none','eng')
        self.assertEqual(refine.call_args[0][:2],('two.png',2))
        regions=service._fallback_regions([{'expected_bbox':[100,50,140,120],'page_index':2}],None,'test')
        self.assertEqual(regions[0].bbox,[50,100,120,140])

    def test_missing_page_and_nonfinite(self):
        self.assertFalse(valid_box([0,0,float('nan'),40],page()))
        item=ExamItem('q1','q1',slots=[Slot(1,'fill','q1',[1,1,20,20],7,student_answer='x',content_status='RECOGNIZED')])
        p=package([item]);finalize_answers(p,[page()])
        self.assertIsNone(item.student_answer)
        self.assertIn('PAGE_UNAVAILABLE',[e['code'] for e in p.extraction_errors])

    def test_teacher_null_positions_are_not_compressed(self):
        slots=[Slot(1,'fill','q1',[1,1,20,20],1,status='TEACHER_ANSWER_NEEDS_REVIEW'),
               Slot(2,'fill','q1',[30,30,40,40],1,expected_text='b',status='TEACHER_ANSWER_EXTRACTED',
                    geometry_status='ALIGNED',content_status='RECOGNIZED')]
        item=ExamItem('q1','q1',slots=slots);p=package([item],'teacher')
        finalize_answers(p,[page()]);self.assertEqual(item.standard_answer,[None,'b'])
        from exam_pipeline.slots import _expected_text
        self.assertIsNone(_expected_text(item,0));self.assertEqual(_expected_text(item,1),'b')

    def test_tree_reconstructs_slot_with_correct_types(self):
        item=ExamItem('q1','q1','Question ____',item_type='fill',expected_slot_count=1,
            stem_region=PageRegion(1,'one.png',[10,10,400,30]),
            cardinality_evidence={'decision':'CONSENSUS_TWO_WAY','sources':{'layout':{'boxes':[[40,20,70,90]]}}})
        p=package([item],'teacher');tree=ExamTreeService.compile(p)
        ExamTreeService.apply_to_package(tree,p)
        slot=p.sections[0].questions[0].items[0].slots[0]
        self.assertEqual(slot.expected_bbox,[40,20,70,90]);self.assertEqual(slot.parent_item_id,'q1')
        self.assertEqual(slot.page_index,1)

    def test_conflict_tree_not_locked(self):
        item=ExamItem('q1','q1',expected_slot_count=1,cardinality_evidence={'decision':'CONFLICT'})
        tree=ExamTreeService.compile(package([item],'teacher'))
        self.assertEqual(ExamTreeService.finalize(tree,lock=True)['state'],'DRAFT')

    def test_valid_draft_cache_can_be_reused_without_claiming_lock(self):
        item=ExamItem('q1','q1',expected_slot_count=1,
                      slots=[Slot(1,'fill','q1',[10,10,40,40],1)])
        tree=ExamTreeService.compile(package([item],'teacher'))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tree.json'
            ExamTreeService.save(tree,path,lock=False)
            cached=ExamTreeService.load(path,expected_subject='generic',
                                        require_valid=True,allow_valid_draft=True)
        self.assertEqual(cached['state'],'DRAFT')
        self.assertEqual(cached['validation']['status'],'VALID')

    def test_real_overlap_is_reported(self):
        items=[]
        for n in (1,2):
            items.append(ExamItem('q'+str(n),'',slots=[Slot(1,'fill','q'+str(n),[10,10,40,40],1,
                handwriting_bbox=[10,10,40,40],geometry_status='ALIGNED',content_status='RECOGNIZED',student_answer='x')]))
        p=package(items);finalize_answers(p,[page()]);r=validate_item_regions(p,[page()])
        self.assertEqual(r['overlap_count'],1)


class StructureTests(unittest.TestCase):
    def test_single_column_options_stay_with_question(self):
        blocks=[OCRBlock('1. Long question across entire page',[30,100,950,125]),
                OCRBlock('A. one',[30,150,300,175]),OCRBlock('B. two',[550,150,850,175]),
                OCRBlock('2. Next question',[30,230,950,255])]
        self.assertEqual([b.text for b in order_blocks_column_aware(blocks,1000)],
                         [b.text for b in blocks])
        sections=build_sections([page(blocks=blocks)],'teacher')
        self.assertIn('B. two',sections[0].questions[0].question_title)
        self.assertEqual(len(sections[0].questions),2)

    def test_two_columns_with_heading(self):
        blocks=[OCRBlock('Full width heading',[20,10,980,35])]
        for x,prefix in [(30,'left'),(560,'right')]:
            for y in (100,200,300,400,500):
                blocks.append(OCRBlock(prefix+' body paragraph',[x,y,x+360,y+25]))
        ordered=order_blocks_column_aware(blocks,1000)
        self.assertTrue(ordered[1].text.startswith('left'));self.assertTrue(ordered[5].text.startswith('left'))
        self.assertTrue(ordered[6].text.startswith('right'))

    def test_slot_order_tolerates_baseline_jitter(self):
        slots=[Slot(1,'fill','q',[99,600,130,700]),Slot(2,'fill','q',[101,60,132,160])]
        self.assertEqual(ordered_slots(slots)[0].expected_bbox[1],60)

    def test_multisubject_multipage_coverage(self):
        for subject in ['math','language','chemistry']:
            for count in (1,2,4):
                pages=[page(n,[OCRBlock(str(n)+'. '+subject+' question',[30,100,900,130])]) for n in range(1,count+1)]
                p=package([], 'teacher', subject);p.sections=build_sections(pages[:1],'teacher')
                audit=recover_structure(p,pages,pages)
                self.assertEqual(audit['status'],'COMPLETE')
                self.assertEqual(len([q for s in p.sections for q in s.questions]),count)

    def test_damaged_template_falls_back(self):
        original=page(blocks=[OCRBlock('10. A long printed question',[30,100,950,130]),OCRBlock('11. Another question',[30,200,950,230])])
        damaged=page(blocks=[OCRBlock('10. fragments',[30,100,200,130])])
        selected,audit=select_structure_pages([original],[damaged])
        self.assertEqual(selected[0].ocr,original.ocr);self.assertFalse(audit[0]['template_accepted'])

    def test_no_fabricated_question_from_header(self):
        self.assertEqual(build_sections([page(blocks=[OCRBlock('Exam heading',[30,30,500,60])])],'teacher'),[])

    def test_inline_subquestions_split_without_fake_geometry(self):
        item=ExamItem('q1','q1','Given x. (1) Compute y; (2) Explain why.',
                     answer_regions=[PageRegion(1,'one.png',[30,100,950,600])])
        children=FineGrainedItemSplitter().split_item(item,'q1',1,[page()])
        self.assertEqual(len(children),2)
        self.assertIn('Given x.',children[1].question_text)
        self.assertEqual(children[0].quality['geometry_status'],'UNRESOLVED_SUBITEM_BOUNDARY')

    def test_zero_one_and_multiple_consensus_samples(self):
        image=np.full((60,120,3),255,np.uint8);image[20:30,20:90]=0
        for count in (0,1,2,3,7):
            mask,audit=CohortConsensusBuilder.consensus_mask([image.copy() for _ in range(count)])
            self.assertEqual(audit['status'],'INSUFFICIENT_SAMPLES' if count<2 else 'READY')


class RecognitionTests(unittest.TestCase):
    def test_reference_answer_cannot_select_ambiguous_observation(self):
        value,audit=parse_student_answer('B D','D','choice_mark','choice','')
        self.assertIsNone(value)

    def test_no_lexical_auto_correction(self):
        result=TeacherAnswerQualityAssessor.assess_answer('木太OIl','short_answer',.99)
        self.assertEqual(result['corrected_text'],'木太OIl')
        self.assertTrue(TeacherAnswerQualityAssessor.assess_answer('不','choice',.99)['needs_review'])
        self.assertEqual(TeacherAnswerExtractionService()._infer_item_type(ExamItem('q5','',item_type='fill')),'fill_blank')

    def test_formula_and_print_contamination(self):
        self.assertIn('UNBALANCED_EXPRESSION',content_issues('y=(x+1','', 'solve',.99))
        self.assertIn('PRINTED_STEM_CONTAMINATION',content_issues('该函数的开口方向是','该函数的开口方向是____','fill',.99))

    def test_recognition_ignores_legacy_mask_and_reads_original_crop(self):
        class FakeOCR:
            def recognize_mask(self,*a):return []
            def recognize_crop(self,*a,**kw):return [OCRBlock('B',[20,30,50,60],.99)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'page.png';cv2.imwrite(str(path),np.full((100,100,3),255,np.uint8))
            result=AnswerRecognizer(FakeOCR()).recognize(Page(1,str(path),100,100,[]),[25,15,65,55],
                                                        kind='choice',mask=np.zeros((20,20),np.uint8))
        self.assertEqual(result['status'],'RECOGNIZED');self.assertEqual(result['text'],'B')
        self.assertEqual(len(result['attempts']),1)
        self.assertEqual(result['attempts'][0]['source'],'original_crop')

    def test_formula_backend_is_used(self):
        class FakeOCR:
            def recognize_crop(self,*a,**kw):return [OCRBlock('x^',[20,30,50,60],.5)]
        class Formula:
            def analyze(self,prompt,paths):return {'transcription':'x^{2}'}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'p.png';cv2.imwrite(str(path),np.full((100,100,3),255,np.uint8))
            r=AnswerRecognizer(FakeOCR(),Formula()).recognize(Page(1,str(path),100,100,[]),[20,20,60,60],'x^2','fill')
        self.assertEqual(r['status'],'RECOGNIZED');self.assertEqual(r['text'],'x^{2}')


class RecoveryAndEvaluationTests(unittest.TestCase):
    def test_chinese_fullstop_and_dual_source_anchor(self):
        original=page(blocks=[OCRBlock('11。已知方程',[30,100,950,130],.9)])
        template=page(blocks=[OCRBlock('12. another printed question',[30,200,950,230],.9)])
        selected,audit=select_structure_pages([original],[template])
        ids={q.question_num for s in build_sections(selected,'teacher') for q in s.questions}
        self.assertEqual(ids,{11,12})

    def test_correct_ocr_does_not_need_correct_teacher_answer(self):
        from exam_pipeline.verification import SlotVerificationService
        slot=Slot(1,'choice','q1',[10,10,40,40],1,expected_text='D',
                  audit={'coordinate_authority':'ocr_boxes_only'})
        class Controller:
            def verify(self,*args):
                return dict(status='CONVERGED_SUCCESS',iterations_used=1,initial_bbox=[10,10,40,40],
                    final_bbox=[10,10,40,40],expected_text='D',recognized_text='',has_ink=True,
                    shrink_rate='0.0%',history=[],geometry_status='ALIGNED',content_status='OCR_UNCERTAIN',
                    semantic_status='NOT_EVALUATED',review_status='NEED_REVIEW')
        class Recognizer:
            def recognize(self,*args):
                return dict(text='B',status='RECOGNIZED',fragments=[],attempts=[],issues=[])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'p.png';cv2.imwrite(str(path),np.full((100,100,3),255,np.uint8))
            p=package([ExamItem('q1','q1','Choose',item_type='choice',slots=[slot])])
            service=SlotVerificationService(controller=Controller());service.recognizer=Recognizer()
            service.verify_package(p,[Page(1,str(path),100,100,[])])
        self.assertEqual(slot.student_answer,'B');self.assertEqual(slot.review_status,'AUTO_PASS')

    def test_crosspage_answer_fragments_aggregate(self):
        slots=[]
        for index in (1,2):
            slots.append(Slot(index,'free_response','q1',[10,10,50,50],index,
                student_answer='part'+str(index),handwriting_bbox=[10,10,50,50],
                content_status='RECOGNIZED',geometry_status='ALIGNED',semantic_id='q1:slot:1'))
        item=ExamItem('q1','q1',expected_slot_count=1,slots=slots)
        finalize_answers(package([item]),[page(),page(2)])
        self.assertEqual(item.student_answer,'part1\npart2');self.assertEqual(len(item.answer_parts[0]['fragments']),2)

    def test_evaluation_penalizes_missing_questions_and_wrong_pages(self):
        from exam_pipeline.evaluation import evaluate
        truth={'questions':[{'id':'q1','items':[{'id':'q1','answer':'B','slots':[
            {'id':'q1:slot:1','page_index':1,'bbox':[10,10,40,40],'text':'B'}]}]}, {'id':'q2','items':[]}]}
        prediction={'document_type':'student','sections':[{'questions':[{'question_id':'q1','items':[
            {'item_id':'q1','student_answer':'B','slots':[{'slot_idx':1,'page_index':2,
                'handwriting_bbox':[10,10,40,40],'student_answer':'B'}]}]}]}]}
        result=evaluate(prediction,truth)
        self.assertEqual(result['question_recall'],.5);self.assertEqual(result['slot_recall_iou50'],0)

class IntegrationTests(unittest.TestCase):
    def test_print_handwriting_template_stage_is_disabled(self):
        import homework_extractor as runtime
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'data'; output = Path(tmp) / 'out'
            folder = root / 'language' / 'teacher'; folder.mkdir(parents=True)
            Image.new('RGB', (600, 800), 'white').save(folder / 'page_01.jpg')

            def make_pages(paths, engine, ocr_map, language):
                return [Page(1, str(paths[0]), 600, 800, [
                    OCRBlock('1. Choose a word ( )', [30, 30, 500, 60], .99),
                    OCRBlock('A. spring B. summer', [30, 80, 500, 110], .99),
                ])]

            with patch.object(runtime, 'make_pages', side_effect=make_pages), \
                 patch('exam_pipeline.cohort_consensus.CohortConsensusBuilder.build',
                       side_effect=AssertionError('template stage must not run')):
                result = runtime.process(
                    root, output, 'unused', None, 'https://example.invalid', 'none', None,
                    False, False, 'eng', 1, local_ocr=True,
                    structure_vlm='none', slot_semantics='none')

            self.assertEqual(result['errors'], [])
            self.assertFalse((output / 'cohort_consensus').exists())
            self.assertNotIn('cohort_consensus',result['runtime'])

    def test_pipeline_without_models_or_cohort_never_claims_answers(self):
        import homework_extractor as runtime
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'data';output=Path(tmp)/'out'
            for role in ('teacher','student001'):
                folder=root/'language'/role;folder.mkdir(parents=True)
                Image.new('RGB',(600,800),'white').save(folder/'page_01.jpg')
            def make_pages(paths,engine,ocr_map,language):
                return [Page(n,str(path),1654,2338,[OCRBlock('1. Choose a word ( )',[100,100,1200,140],.99),
                    OCRBlock('A. spring B. summer',[100,180,1200,220],.99)]) for n,path in enumerate(paths,1)]
            with patch.object(runtime,'make_pages',side_effect=make_pages):
                result=runtime.process(root,output,'unused',None,'https://example.invalid','none',None,
                    False,False,'eng',1,local_ocr=False,structure_vlm='none')
            self.assertEqual(result['errors'],[])
            student=json.loads((output/'language__student__student001.json').read_text())
            self.assertEqual(student['schema_version'],'exam_package.v5')
            self.assertEqual(student['extraction_status'],'PARTIAL')
            for s in student['sections']:
                for q in s['questions']:
                    for item in q['items']:
                        answer=item['student_answer']
                        self.assertTrue(answer is None or isinstance(answer,list) and all(x is None for x in answer))

    def test_recognition_exception_is_bounded_and_audited(self):
        class BrokenOCR:
            calls=0
            def recognize_crop(self,*a,**k):self.calls+=1;raise RuntimeError('offline')
            def recognize_page(self,*a,**k):self.calls+=1;raise RuntimeError('offline')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'p.png';cv2.imwrite(str(path),np.full((100,100,3),255,np.uint8))
            ocr=BrokenOCR();r=AnswerRecognizer(ocr).recognize(Page(1,str(path),100,100,[]),[10,10,60,60])
            self.assertLessEqual(ocr.calls,2);self.assertEqual(r['status'],'UNRESOLVED')
            self.assertTrue(all('OCR_BACKEND_ERROR' in a['issues'] for a in r['attempts']))

class AdditionalGeneralizationTests(unittest.TestCase):
    def test_restarted_numbering_preserves_distinct_questions(self):
        from homework_extractor import reconcile_exam_package
        pages=[page(1,[OCRBlock('1. First section question',[30,100,900,130])]),
               page(2,[OCRBlock('1. Second section question',[30,100,900,130])])]
        p=package([],'teacher');p.sections=build_sections(pages,'teacher');reconcile_exam_package(p)
        questions=[q for s in p.sections for q in s.questions]
        self.assertEqual(len(questions),2);self.assertNotEqual(questions[0].question_id,questions[1].question_id)

    def test_long_answer_and_formula_not_truncated(self):
        text='The explanation repeats the given conditions. '+('x^2+2*x+1 = (x+1)^2\n'*10)
        value,_=parse_student_answer(text,None,'free_response','solve',text[:40])
        self.assertEqual(value,text.strip())

    def test_image_coverage_retry_recovers_missing_printed_line(self):
        from exam_pipeline.structure_recovery import recover_unread_lines
        class OCR:
            def recognize_crop(self,*a,**kw):
                return [OCRBlock('(3) Explain the result',[30,100,800,130],.99)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'p.png';im=np.full((500,1000,3),255,np.uint8)
            cv2.putText(im,'(3) Explain the result of the experiment',(30,130),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
            cv2.imwrite(str(path),im);p=Page(1,str(path),1000,500,[])
            audit=recover_unread_lines([p],OCR(),'paddle','eng')
            self.assertTrue(audit);self.assertTrue(p.ocr)

    def test_invalid_roi_stops_before_ocr_and_surfaces_document_error(self):
        ocr=unittest.mock.Mock()
        result=AnswerRecognizer(ocr).recognize(page(),[10,10,5,5])
        self.assertEqual(result['issues'],['INVALID_EXPECTED_BOX'])
        self.assertFalse(ocr.mock_calls)
        slot=Slot(1,'fill','q1',[10,10,5,5],1,student_answer='B',
                  content_status='RECOGNIZED',geometry_status='ALIGNED')
        p=package([ExamItem('q1','q1',slots=[slot])])
        finalize_answers(p,[page()])
        self.assertEqual(p.extraction_status,'PARTIAL')
        self.assertIsNone(p.sections[0].questions[0].items[0].student_answer)
        self.assertIn('INVALID_EXPECTED_BOX',[e['code'] for e in p.extraction_errors])

    def test_template_only_anchor_is_recoverable(self):
        original=page(1,[OCRBlock('1. Original question',[30,100,900,130])])
        selected=page(1,original.ocr+[OCRBlock('2. Template question',[30,300,900,330])])
        p=package([],'teacher');p.sections=build_sections([original],'teacher')
        audit=recover_structure(p,[selected],[original])
        self.assertEqual(audit['missing_anchors'],[])
        self.assertEqual(len([q for s in p.sections for q in s.questions]),2)

    def test_evaluation_preserves_crosspage_physical_occurrences(self):
        from exam_pipeline.evaluation import evaluate
        boxes=[{'id':'q1:slot:1','page_index':p,'bbox':[10,10,40,40],'text':str(p)} for p in (1,2)]
        truth={'questions':[{'id':'q1','items':[{'id':'q1','answer':'1\n2','slots':boxes}]}]}
        prediction={'document_type':'student','sections':[{'questions':[{'question_id':'q1','items':[
            {'item_id':'q1','student_answer':'1\n2','slots':[{'slot_idx':p,'semantic_id':'q1:slot:1',
                'page_index':p,'handwriting_bbox':[10,10,40,40],'student_answer':str(p)} for p in (1,2)]}]}]}]}
        result=evaluate(prediction,truth)
        self.assertEqual(result['slot_recall_iou50'],1)
        self.assertEqual(result['slot_precision_iou50'],1)

    def test_batch_evaluation_groups_dimensions_and_reports_missing_cases(self):
        from exam_pipeline.evaluation import evaluate_manifest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'p.json').write_text(json.dumps({'sections':[]}))
            (root/'t.json').write_text(json.dumps({'questions':[{'id':'q1','items':[]}]}))
            cases=[{'prediction':'p.json','truth':'t.json','subject':s,'layout':l,'pages':p,'cohort_samples':n}
                   for s,l,p,n in [('math','single',1,0),('language','dual',4,7)]]
            cases.append({'prediction':'missing.json','truth':'t.json'})
            (root/'matrix.json').write_text(json.dumps({'cases':cases}))
            result=evaluate_manifest(root/'matrix.json')
            self.assertEqual(result['status'],'PARTIAL')
            self.assertEqual(len(result['cases']),2)
            self.assertEqual(len(result['errors']),1)
            self.assertEqual(result['groups']['pages=4']['question_recall'],0)

if __name__=='__main__':unittest.main()
