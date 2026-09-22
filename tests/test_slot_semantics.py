import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from exam_pipeline.contracts import Page, OCRBlock, Slot, ExamItem, ExamPackage, ExamSection, ExamQuestion, PageRegion
from exam_pipeline.slot_candidate_selection import VisualSlotSemanticService
from exam_pipeline.exam_tree import ExamTreeService
from exam_pipeline.result_contract import finalize_answers


def response(ids, index=1, label='axis of symmetry'):
    return {'index':index,'label':label,'anchor_before':'axis =','anchor_after':'',
            'candidate_ids':ids,'confidence':.95}


class SlotSemanticTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.pages=[]
        for index in (1,2):
            path=self.root/('p%d.png'%index)
            im=np.full((400,600,3),255,np.uint8)
            cv2.putText(im,'x=2',(210,140),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
            cv2.imwrite(str(path),im)
            self.pages.append(Page(index,str(path),600,400,[OCRBlock('x=2',[210,115,270,145],.99)]))
        self.item=ExamItem('q1','q1','Find the axis.',item_type='fill',
            stem_region=PageRegion(1,self.pages[0].path,[50,80,450,100]),
            answer_regions=[PageRegion(p.index,p.path,[50,80,550,200]) for p in self.pages],
            slots=[Slot(1,'fill','q1',[110,205,150,275],1)])
        self.package=ExamPackage('exam','','generic','teacher',None,2,[p.path for p in self.pages],
            [ExamSection('s','',[ExamQuestion('q1',1,'Question',[self.item])])])

    def run_service(self,request):
        return VisualSlotSemanticService(request,'test').enrich_package(self.package,self.pages,self.root/'audit')

    def test_select_ids_uses_only_local_coordinates(self):
        def request(prompt,paths,schema):
            self.assertTrue(all(p.is_file() for p in paths))
            self.assertNotIn('standard_answer',prompt)
            return {'slots':[response(['p1c1'])]}
        self.item.standard_answer='SECRET_REFERENCE'
        result=self.run_service(request)
        self.assertEqual(result['accepted'],1)
        self.assertEqual(self.item.slots[0].expected_bbox,[110,205,150,275])
        self.assertIsNone(self.item.slots[0].expected_text)
        self.assertEqual(self.item.semantic_slot_plan[0]['label'],'axis of symmetry')

    def test_model_coordinates_rejected_bounded(self):
        calls=[]
        def request(*args):
            calls.append(1);return {'slots':[dict(response(['p1c1']),bbox=[0,0,600,400])]}
        before=copy.deepcopy(self.item.slots)
        result=self.run_service(request)
        self.assertEqual(result['fallback'],1);self.assertEqual(len(calls),2)
        self.assertEqual(self.item.slots,before)
        self.assertEqual(self.item.slot_semantics_audit['reason'],'INVALID_SLOT_FIELDS')

    def test_unknown_ids_retried_without_mutating_coordinates(self):
        def request(*args):return {'slots':[response(['invented'])]}
        result=self.run_service(request)
        self.assertEqual(result['fallback'],1)
        self.assertEqual(self.item.slot_semantics_audit['reason'],'UNKNOWN_CANDIDATE_ID')

    def test_student_semantic_order_is_preserved(self):
        self.package.document_type='student'
        self.item.semantic_slot_plan=[{'slot_id':'q1:slot:1','index':1,'label':'vertex','anchor_before':'vertex','anchor_after':''}]
        self.run_service(lambda *args:{'slots':[response(['p1c1'],label='renamed')]})
        self.assertEqual(self.item.slots[0].audit['semantic_label'],'vertex')
        self.assertEqual(self.item.slots[0].semantic_id,'q1:slot:1')

    def test_student_cannot_change_teacher_count(self):
        self.package.document_type='student'
        self.item.semantic_slot_plan=[{'slot_id':'q1:slot:1'}, {'slot_id':'q1:slot:2'}]
        result=self.run_service(lambda *args:{'slots':[response(['p1c1'])]})
        self.assertEqual(result['fallback'],1)
        self.assertEqual(self.item.slot_semantics_audit['reason'],'TEACHER_TOPOLOGY_CONFLICT')

    def test_missing_candidate_triggers_image_retry_and_stays_missing(self):
        self.run_service(lambda *args:{'slots':[response([])]})
        self.assertEqual(len(self.item.slot_semantics_audit['attempts']),2)
        self.assertEqual(self.item.slot_semantics_audit['status'],'PARTIAL')
        self.assertEqual(self.item.slots[0].expected_bbox,[])
        self.assertTrue(any(c['source']=='image_component' for c in self.item.slot_semantics_audit['candidate_table']))
        finalize_answers(self.package,self.pages)
        self.assertEqual(self.package.extraction_status,'PARTIAL')

    def test_crosspage_fragments_preserve_one_semantic_identity_in_tree(self):
        def request(prompt,*args):
            context=json.loads(prompt.split('\n',1)[1])
            ids=[c['id'] for c in context['candidates'] if c['source']=='ocr']
            return {'slots':[response(ids)]}
        self.run_service(request)
        self.assertEqual(len(self.item.slots),2)
        self.assertEqual(len({s.semantic_id for s in self.item.slots}),1)
        self.package.structure_audit={'status':'COMPLETE'}
        tree=ExamTreeService.compile(self.package)
        clone=copy.deepcopy(self.package)
        ExamTreeService.apply_to_package(tree,clone)
        rebuilt=clone.sections[0].questions[0].items[0]
        self.assertEqual(len(rebuilt.slots),2)
        self.assertEqual(rebuilt.semantic_slot_plan,self.item.semantic_slot_plan)
        self.assertEqual(tree['sections'][0]['questions'][0]['items'][0]['localized_slot_count'],1)

    def test_backend_exception_is_bounded_and_explicit(self):
        def request(*args):raise RuntimeError('backend unavailable')
        result=self.run_service(request)
        self.assertEqual(result['fallback'],1)
        self.assertEqual(len(self.item.slot_semantics_audit['attempts']),2)
        self.assertEqual(self.item.slot_semantics_audit['reason'],'SEMANTIC_BACKEND_ERROR')

    def test_overlapping_detectors_for_same_logical_slot_are_merged_locally(self):
        self.run_service(lambda *args:{'slots':[response(['p1c1','p1c2'])]})
        self.assertEqual(self.item.slot_semantics_audit['status'],'ACCEPTED')
        self.assertEqual(len(self.item.slots),1)
        self.assertEqual(self.item.slots[0].expected_bbox,[110,205,150,275])

    def test_overlapping_detectors_for_different_slots_are_rejected(self):
        self.run_service(lambda *args:{'slots':[response(['p1c1']),response(['p1c2'],2)]})
        self.assertEqual(self.item.slot_semantics_audit['status'],'FALLBACK')
        self.assertEqual(self.item.slot_semantics_audit['reason'],'OVERLAPPING_SELECTION')

    def test_same_candidate_cannot_belong_to_two_slots(self):
        result=self.run_service(lambda *args:{'slots':[response(['p1c1']),response(['p1c1'],2)]})
        self.assertEqual(result['fallback'],1)
        self.assertEqual(self.item.slot_semantics_audit['reason'],'CANDIDATE_ASSIGNED_TWICE')

    def test_missing_semantic_region_is_not_an_invalid_coordinate(self):
        self.run_service(lambda *args:{'slots':[response([])]})
        finalize_answers(self.package,self.pages)
        codes=[e['code'] for e in self.package.extraction_errors]
        self.assertIn('SLOT_CANDIDATE_MISSING',codes)
        self.assertNotIn('INVALID_EXPECTED_BOX',codes)
        self.assertEqual(self.item.slots[0].geometry_status,'MISSING')
        self.assertTrue((self.root/'audit/item_0001.json').is_file())

    def test_local_selection_and_null_positions_survive_finalization(self):
        self.package.document_type='student'
        self.item.semantic_slot_plan=[{'slot_id':'q1:slot:1','index':1,'label':'axis','anchor_before':'axis','anchor_after':''},
            {'slot_id':'q1:slot:2','index':2,'label':'vertex','anchor_before':'vertex','anchor_after':''}]
        self.run_service(lambda *args:{'slots':[response(['p1c1']),response([],2,'vertex')]})
        slot=self.item.slots[0]
        slot.student_answer='x=2';slot.content_status='RECOGNIZED';slot.geometry_status='ALIGNED'
        slot.handwriting_bbox=list(slot.expected_bbox)
        finalize_answers(self.package,self.pages)
        self.assertEqual(self.item.student_answer,['x=2',None])
        self.assertEqual([p['slot_id'] for p in self.item.answer_parts],['q1:slot:1','q1:slot:2'])

    def test_choice_options_are_not_independent_slots(self):
        self.item.item_type='choice'
        self.run_service(lambda *args:{'slots':[response(['p1c1']),response([],2)]})
        self.assertEqual(self.item.slot_semantics_audit['status'],'FALLBACK')
        self.assertEqual(self.item.slot_semantics_audit['reason'],'SINGLE_RESPONSE_REQUIRED_OPTIONS_ARE_NOT_SLOTS')

    def test_free_response_is_one_slot_even_when_multiple_regions(self):
        self.item.item_type='large_writing'
        self.run_service(lambda *args:{'slots':[response(['p1c1']),response([],2)]})
        self.assertEqual(self.item.slot_semantics_audit['status'],'FALLBACK')

    def test_partial_plan_survives_retry_backend_failure(self):
        calls=[]
        def request(*args):
            calls.append(1)
            if len(calls)==1:return {'slots':[response([])]}
            raise RuntimeError('offline')
        self.run_service(request)
        self.assertEqual(self.item.slot_semantics_audit['status'],'PARTIAL')
        self.assertEqual(len(self.item.semantic_slot_plan),1)
        self.assertEqual(self.item.slots[0].expected_bbox,[])

    def test_low_or_nonfinite_confidence_cannot_be_accepted(self):
        result=self.run_service(lambda *args:{'slots':[dict(response(['p1c1']),confidence=float('nan'))]})
        self.assertEqual(result['fallback'],1)

    def test_disabled_backend_is_visible_in_output(self):
        self.run_service(None);finalize_answers(self.package,self.pages)
        self.assertIn('SLOT_SEMANTICS_UNRESOLVED',[e['code'] for e in self.package.extraction_errors])

    def test_main_pipeline_calls_semantics_before_answer_extraction(self):
        from exam_pipeline.slot_semantics import VisualSlotSemanticService
        import homework_extractor as runtime
        from exam_pipeline.teacher_answers import TeacherAnswerExtractionService
        dataset=self.root/'data'
        folder=dataset/'generic'/'teacher';folder.mkdir(parents=True)
        cv2.imwrite(str(folder/'page_01.png'),np.full((400,600,3),255,np.uint8))
        def pages(paths,*args):
            return [Page(1,str(paths[0]),600,400,[OCRBlock('1. Find x = ____',[50,80,450,100],.99)])]
        calls=[]
        def semantic(service,package,pages,output,**kwargs):
            calls.append('semantics');return {'accepted':1,'partial':0,'fallback':0}
        def extract(service,*args,**kwargs):
            self.assertEqual(calls,['semantics']);calls.append('answers')
            return {'total_slots':0,'answers_extracted':0}
        with patch.object(runtime,'make_pages',side_effect=pages), patch.object(VisualSlotSemanticService,'enrich_package',semantic), patch.object(TeacherAnswerExtractionService,'extract',extract):
            result=runtime.process(dataset,self.root/'out','unused',None,'https://example.invalid','none',None,False,False,'eng',1,
                local_ocr=False,structure_vlm='none',slot_semantics='none')
        self.assertEqual(result['errors'],[]);self.assertEqual(calls,['semantics','answers'])

if __name__=='__main__':unittest.main()
