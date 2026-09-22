import copy
import json
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from exam_pipeline.contracts import *
from exam_pipeline.localization import (ground_contexts, match_lines,
                                        partition_sibling_answer_domains)
from exam_pipeline.candidate_review import CandidateReviewer
from exam_pipeline.answer_vision import AnswerVisionClient
from exam_pipeline.answer_recognition import AnswerRecognizer
from exam_pipeline.stage_evaluation import evaluate_stages


class AutomaticLocalizationTests(unittest.TestCase):
    def package(self, index=1):
        i=ExamItem('q1','1','Choose the matching value',item_type='choice')
        i.quality['structure_references']=[{'page_index':index,'anchor':'Choose the matching value'}]
        return ExamPackage('e','','generic','student',None,1,[],[ExamSection('s','',[ExamQuestion('q1',1,'Choose the matching value',[i])])]),i

    def test_multiline_matching_uses_neighbor_lines(self):
        blocks=[OCRBlock('The plant grows several branches',[20,100,300,125]),
                OCRBlock('and each branch has leaves',[20,130,300,155]),
                OCRBlock('another unrelated prompt',[20,400,300,425])]
        match=match_lines('The plant grows several branches and each branch has leaves',blocks)
        self.assertEqual(match[1:],(0,2))

    def test_subitem_body_beats_handwritten_solution_marker(self):
        blocks=[
            OCRBlock('(1) Find the value range', [20,100,300,130]),
            OCRBlock('(1) solution: x=3', [20,160,260,200]),
        ]
        match=match_lines('(1) Find the value range',blocks)
        self.assertEqual(match[1:],(0,1))

    def test_large_answer_domain_starts_after_measured_stem(self):
        package,item=self.package();item.item_type='large_writing'
        item.quality['structure_references']=[{'page_index':1,'anchor':'(1) Find the value'}]
        page=Page(1,'a',500,700,[
            OCRBlock('1. Composite problem', [20,100,400,130]),
            OCRBlock('(1) Find the value', [30,150,250,180]),
            OCRBlock('student working', [30,220,260,260]),
        ])
        ground_contexts(package,[page])
        context=item.quality['localization']['contexts'][0]
        self.assertEqual(context['answer_search_domain'][1],180)
        self.assertGreater(context['answer_search_domain'][3],260)
        self.assertNotEqual(context['answer_search_domain'],context['bbox'])

    def test_horizontal_sibling_answer_bands_do_not_overlap(self):
        page=Page(1,'a',1000,1000,[])
        first=ExamItem('q1_1','(1)','first',stem_region=PageRegion(1,'a',[80,100,200,140]))
        second=ExamItem('q1_2','(2)','second',stem_region=PageRegion(1,'a',[580,100,700,140]))
        for item in (first,second):
            item.quality['localization']={'contexts':[{
                'page_index':1,'bbox':[0,80,1000,400],
                'answer_search_domain':[0,80,1000,400],
                'status':'QUESTION_ANCHORED'}]}
        question=ExamQuestion('q1',1,'composite',[first,second])
        partition_sibling_answer_domains(question,[page],{
            first.item_id:{'axis':'horizontal','confidence':.95},
            second.item_id:{'axis':'horizontal','confidence':.95}})
        left=first.quality['localization']['contexts'][0]['answer_search_domain']
        right=second.quality['localization']['contexts'][0]['answer_search_domain']
        self.assertLessEqual(left[2],right[0])
        self.assertEqual(first.quality['localization']['contexts'][0]['layout_axis'],'horizontal')

    def test_vertical_sibling_answer_bands_do_not_overlap(self):
        page=Page(1,'a',1000,1000,[])
        first=ExamItem('q1_1','(1)','first',item_type='large_writing',
                       stem_region=PageRegion(1,'a',[80,100,600,140]))
        second=ExamItem('q1_2','(2)','second',item_type='large_writing',
                        stem_region=PageRegion(1,'a',[80,420,600,460]))
        for item in (first,second):
            item.quality['localization']={'contexts':[{
                'page_index':1,'bbox':[0,80,1000,900],
                'answer_search_domain':[0,80,1000,900],
                'status':'QUESTION_ANCHORED'}]}
        question=ExamQuestion('q1',1,'composite',[first,second])
        partition_sibling_answer_domains(question,[page])
        upper=first.quality['localization']['contexts'][0]['answer_search_domain']
        lower=second.quality['localization']['contexts'][0]['answer_search_domain']
        self.assertLessEqual(upper[3],lower[1])
        self.assertEqual(upper[3],420)

    def test_vlm_regions_order_duplicate_ocr_sibling_anchors(self):
        page=Page(1,'a',1000,1000,[])
        items=[]
        for index in (1,2):
            item=ExamItem('q1_{}'.format(index),'({})'.format(index),'part',
                          item_type='large_writing',stem_region=PageRegion(1,'a',[50,80,800,120]))
            item.quality['localization']={'contexts':[{
                'page_index':1,'bbox':[0,50,1000,900],
                'answer_search_domain':[0,50,1000,900],
                'status':'QUESTION_ANCHORED'}]}
            items.append(item)
        question=ExamQuestion('q1',1,'composite',items)
        hints={
            'q1_1':{'axis':'vertical','confidence':.9,'coordinate_space':'full_page_normalized',
                    'regions':[{'page_index':1,'search_bbox':[50,150,900,400]}]},
            'q1_2':{'axis':'vertical','confidence':.9,'coordinate_space':'full_page_normalized',
                    'regions':[{'page_index':1,'search_bbox':[50,520,900,850]}]},
        }
        partition_sibling_answer_domains(question,[page],hints)
        first=items[0].quality['localization']['contexts'][0]
        second=items[1].quality['localization']['contexts'][0]
        self.assertLessEqual(first['answer_search_domain'][3],second['answer_search_domain'][1])
        self.assertEqual(first['anchor_source'],'vlm_proposal_order_hint')

    def test_context_does_not_jump_to_other_page(self):
        package,item=self.package(3)
        pages=[Page(3,'a',500,700,[OCRBlock('1. Choose the matching value',[30,200,450,230])]),
               Page(8,'b',500,700,[OCRBlock('1. Choose the matching value',[30,100,450,130])])]
        ground_contexts(package,pages)
        self.assertEqual([r.page_index for r in item.answer_regions],[3])
        self.assertEqual(item.stem_region.page_index,3)
        self.assertLess(item.answer_regions[0].bbox[1],188)

    def test_missing_ocr_retains_page_context_not_answer_box(self):
        package,item=self.package()
        ground_contexts(package,[Page(1,'a',500,700,[])])
        self.assertEqual(item.answer_regions,[])
        self.assertEqual(item.quality['localization']['status'],'CONTEXT_ONLY')
        self.assertEqual(item.quality['localization']['contexts'][0]['bbox'],[0,0,500,700])

    def test_crosspage_refs_stay_distinct(self):
        package,item=self.package()
        item.quality['structure_references'].append({'page_index':4,'anchor':'Continue the reasoning below'})
        pages=[Page(1,'a',500,700,[OCRBlock('1. Choose the matching value',[30,100,450,130])]),
               Page(4,'b',500,700,[OCRBlock('Continue the reasoning below',[30,100,450,130])])]
        ground_contexts(package,pages)
        self.assertEqual([r.page_index for r in item.answer_regions],[1,4])

    def test_review_rejects_printed_candidate_and_cannot_invent_coordinates(self):
        package,item=self.package();im=np.full((100,200,3),255,np.uint8)
        local={'status':'VERIFIED','bbox':[20,20,50,50],'alternatives':[{'bbox':[20,20,50,50],'source':'original_ink'}]}
        view={'image':im,'domain':[0,0,200,100],'page':Page(1,'',200,100,[])}
        semantic={'index':1,'label':'answer','anchor_before':'','anchor_after':''}
        with tempfile.TemporaryDirectory() as tmp:
            for raw,reason in [({'decision':'REJECT','candidate_ids':[],'reason':'printed stem'},'VISUAL_REJECT'),
                               ({'decision':'ACCEPT','candidate_ids':[2],'reason':'unknown'},'CANDIDATE_REVIEW_ERROR'),
                               ({'decision':'ACCEPT','candidate_ids':[1],'reason':'ok','bbox':[1,2,3,4]},'CANDIDATE_REVIEW_ERROR')]:
                result=CandidateReviewer(lambda *a:raw).review(item,semantic,view,local,Path(tmp)/'review.png')
                self.assertEqual(result['status'],'UNRESOLVED');self.assertEqual(result['reason'],reason)

    def test_review_cannot_accept_clipped_pixels(self):
        _,item=self.package();im=np.full((100,200,3),255,np.uint8)
        local={'status':'UNRESOLVED','alternatives':[{'bbox':[20,20,50,50],'source':'ink','touches_boundary':True}]}
        view={'image':im,'domain':[0,0,200,100],'page':Page(1,'',200,100,[])}
        semantic={'index':1,'label':'answer','anchor_before':'','anchor_after':''}
        with tempfile.TemporaryDirectory() as tmp:
            r=CandidateReviewer(lambda *a:{'decision':'ACCEPT','candidate_ids':[1],'reason':'visible'}).review(item,semantic,view,local,Path(tmp)/'r.png')
        self.assertEqual(r['reason'],'INK_TOUCHES_SEARCH_BOUNDARY')

    def test_visual_transcription_rescues_empty_ocr_without_reference_answer(self):
        class EmptyOCR:
            def recognize_crop(self,*a,**k):return []
            def recognize_page(self,*a,**k):return []
        seen=[]
        def request(prompt,paths,schema):
            seen.append(prompt);return {'transcription':'C','legible':True,'content_kind':'handwriting'}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a.png';cv2.imwrite(str(path),np.full((100,100,3),255,np.uint8))
            r=AnswerRecognizer(EmptyOCR(),AnswerVisionClient(request)).recognize(Page(1,str(path),100,100,[]),[10,10,80,80],kind='choice')
        self.assertEqual(r['status'],'RECOGNIZED');self.assertEqual(r['text'],'C')
        self.assertNotIn('standard_answer',seen[0])

    def test_visual_content_kind_is_diagnostic_not_a_veto(self):
        class OCR:
            def recognize_crop(self,*a,**k):return [OCRBlock('+x',[10,10,40,40],.99)]
        vision=AnswerVisionClient(lambda *a:{'transcription':'+x','legible':True,'content_kind':'printed'})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a.png';cv2.imwrite(str(path),np.full((100,100,3),255,np.uint8))
            r=AnswerRecognizer(OCR(),vision).recognize(Page(1,str(path),100,100,[]),[10,10,80,80],kind='fill')
        self.assertEqual(r['status'],'RECOGNIZED');self.assertEqual(r['text'],'+x')
        self.assertIn('VISUAL_CONTENT_KIND_PRINTED',r['warnings'])

    def test_stage_metrics_do_not_count_wrong_page_as_localized(self):
        item={'item_id':'q','student_answer':'C','quality':{'localization':{'contexts':[{'page_index':1,'bbox':[0,0,100,100]}]}},
              'slots':[{'slot_idx':1,'semantic_id':'q:slot:1','page_index':2,'expected_bbox':[10,10,40,40],'student_answer':'C'}]}
        pred={'document_type':'student','sections':[{'questions':[{'items':[item]}]}]}
        truth={'questions':[{'items':[{'id':'q','answer':'C','slots':[{'id':'q:slot:1','page_index':1,'bbox':[10,10,40,40],'text':'C'}]}]}]}
        r=evaluate_stages(pred,truth)
        self.assertEqual(r['context_recall'],1);self.assertEqual(r['localization_recall_iou50'],0)
        self.assertIsNone(r['transcription_exact_given_localized'])

    def test_quality_gate_preserves_context_evidence(self):
        from exam_pipeline.quality import validate_item_regions
        package,item=self.package()
        pages=[Page(1,'a',500,700,[OCRBlock('1. Choose the matching value',[30,200,450,230])])]
        ground_contexts(package,pages)
        validate_item_regions(package,pages)
        self.assertIn('localization',item.quality)
        self.assertIn('structure_references',item.quality)

    def test_automatic_relocalization_bounded_and_rejects_other_item_overlap(self):
        from exam_pipeline.automatic_repair import repair_contaminated_answers
        package,item=self.package()
        item.slots=[Slot(1,'semantic_region','q1',[10,10,30,30],audit={'recognition':{'issues':['PRINTED_STEM_CONTAMINATION']}})]
        other=ExamItem('q2','2',slots=[Slot(1,'semantic_region','q2',[40,40,60,60])])
        package.sections[0].questions.append(ExamQuestion('q2',2,'other',[other]))
        class Service:
            request=True
            calls=0
            def enrich_package(self,p,*args):
                self.calls+=1
                i=p.sections[0].questions[0].items[0]
                i.slots=[Slot(1,'semantic_region','q1',[40,40,60,60])]
                i.slot_semantics_audit={'status':'ACCEPTED'}
        service=Service()
        repair_contaminated_answers(package,[],service,lambda p:None,'unused')
        self.assertEqual(item.slots[0].expected_bbox,[])
        self.assertEqual(item.slot_semantics_audit['reason'],'SLOT_EVIDENCE_CONFLICT')
        repair_contaminated_answers(package,[],service,lambda p:None,'unused')
        self.assertEqual(service.calls,1)

    def test_coordinate_miss_is_not_retried_when_no_content_failure_exists(self):
        from exam_pipeline.automatic_repair import repair_contaminated_answers
        package,item=self.package()
        item.slot_semantics_audit={'attempts':[{'status':'PROPOSED'}]}
        item.slots=[Slot(1,'semantic_region','q1',[],1,
                         geometry_status='MISSING',
                         audit={'local_validation':{'reason':'OCR_SLOT_COORDINATES_UNRESOLVED'}})]
        class Service:
            request=True
            calls=0
            def enrich_package(self,*args,**kwargs): self.calls+=1
        service=Service()
        result=repair_contaminated_answers(package,[],service,lambda p:None,'unused')
        self.assertEqual(service.calls,0)
        self.assertEqual(result['status'],'NO_FAILURES')

    def test_retry_that_loses_geometry_does_not_replace_previous_result(self):
        from exam_pipeline.automatic_repair import repair_contaminated_answers
        package,item=self.package()
        item.slots=[Slot(1,'semantic_region','q1',[10,10,30,30],1,
                         geometry_status='ALIGNED', content_status='RECOGNIZED',
                         recognized_text='C',
                         audit={'recognition':{'issues':['PRINTED_STEM_CONTAMINATION']}})]
        item.slot_semantics_audit={'attempts':[{'status':'PROPOSED'}]}
        class Service:
            request=True
            calls=0
            def enrich_package(self,p,*args,**kwargs):
                self.calls+=1
                p.sections[0].questions[0].items[0].slots=[Slot(
                    1,'semantic_region','q1',[],1,geometry_status='MISSING',
                    content_status='NOT_EVALUATED')]
        service=Service()
        repair_contaminated_answers(package,[],service,lambda p:None,'unused')
        self.assertEqual(service.calls,1)
        self.assertEqual(item.slots[0].expected_bbox,[10,10,30,30])
        self.assertEqual(item.quality['automatic_relocalization']['status'],
                         'RETAINED_PREVIOUS_RESULT')

    def test_different_page_counts_and_column_layouts(self):
        for count in (1,2,4):
            package,item=self.package(count)
            pages=[Page(n,'p{}'.format(n),1000,1400,[OCRBlock('1. Choose the matching value',[30,200,450,230]),
                    OCRBlock('2. Right column question with text',[550,200,950,230]),
                    OCRBlock('Left column supporting passage',[30,260,450,290]),
                    OCRBlock('Right column supporting passage',[550,260,950,290]),
                    OCRBlock('Left column concluding passage',[30,320,450,350]),
                    OCRBlock('Right column concluding passage',[550,320,950,350])]) for n in range(1,count+1)]
            ground_contexts(package,pages)
            contexts=item.quality['localization']['contexts']
            self.assertEqual([c['page_index'] for c in contexts],[count])
            self.assertLess(contexts[0]['bbox'][2],550)

    def test_teacher_recognition_uses_ocr_grounded_box_without_ink_snap(self):
        from exam_pipeline.teacher_answers import TeacherAnswerExtractionService
        from unittest.mock import patch
        package,item=self.package();package.document_type='teacher'
        slot=Slot(1,'semantic_region',item.item_id,[10,10,80,80])
        slot.audit={'local_validation':{'support':'local_candidates_visually_reviewed'}}
        item.slots=[slot]
        class Recorder:
            box=None
            def recognize(self,page,box,*args):
                self.box=box
                return {'text':'C','status':'RECOGNIZED','issues':[],'fragments':[],'confidence':.99,'attempts':[]}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'image.png';cv2.imwrite(str(path),np.full((100,100,3),255,np.uint8))
            service=TeacherAnswerExtractionService();recorder=Recorder();service.recognizer=recorder
            service.extract(package,[Page(1,str(path),100,100,[])])
        self.assertEqual(recorder.box,[10,10,80,80])
        self.assertEqual(slot.handwriting_bbox,[10,10,80,80])
        self.assertEqual(slot.recognition_bbox,[10,10,80,80])
        self.assertEqual(slot.expected_text,'C')

    def test_visual_content_uses_one_primary_read(self):
        class OCR:
            def recognize_crop(self,*a,**k):return []
            def recognize_page(self,*a,**k):return []
        calls=[]
        def request(*args):
            calls.append(1)
            return {'transcription':'C' if len(calls)==1 else 'D','legible':True,'content_kind':'handwriting'}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a.png';cv2.imwrite(str(path),np.full((100,100,3),255,np.uint8))
            r=AnswerRecognizer(OCR(),AnswerVisionClient(request)).recognize(Page(1,str(path),100,100,[]),[10,10,80,80],kind='choice')
        self.assertEqual(len(calls),1)
        self.assertEqual(r['status'],'RECOGNIZED')
        self.assertEqual(r['text'],'C')

    def test_student_does_not_inherit_teacher_retry_budget(self):
        from exam_pipeline.golden import GoldenTemplateService
        from unittest.mock import patch
        teacher,item=self.package();teacher.document_type='teacher'
        item.quality['automatic_relocalization']={'attempts':1}
        item.quality['localization_retry_feedback']={'reason':'teacher problem'}
        candidate,_=self.package()
        pages=[Page(1,'a',500,700,[])]
        with patch.object(GoldenTemplateService,'_compute_student_bands',return_value={}):
            student=GoldenTemplateService().inherit_student_topology(teacher,candidate,pages,pages,'student','s')
        quality=student.sections[0].questions[0].items[0].quality
        self.assertNotIn('automatic_relocalization',quality)
        self.assertNotIn('localization_retry_feedback',quality)
