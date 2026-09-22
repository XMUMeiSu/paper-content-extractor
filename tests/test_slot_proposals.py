import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import cv2,numpy as np
from exam_pipeline.contracts import *
from exam_pipeline.slot_semantics import VisualSlotSemanticService
from exam_pipeline.slot_evidence import (LocalSlotEvidenceValidator,
                                         assign_unique_candidates)
from exam_pipeline.exam_tree import ExamTreeService
from exam_pipeline.question_localization import VisualQuestionLocalizer


def proposal(box=[300,200,600,600],page=1):
    return {'slots':[{'index':1,'label':'axis','anchor_before':'axis =','anchor_after':'',
                      'regions':[{'page_index':page,'search_bbox':box}], 'confidence':.95}]}


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.path=self.root/'page.png'
        image=np.full((300,500,3),255,np.uint8)
        cv2.putText(image,'26',(220,140),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        cv2.imwrite(str(self.path),image);self.image=image
        self.page=Page(1,str(self.path),500,300,[OCRBlock('26',[220,115,260,145],.99)])
        self.item=ExamItem('q1','q1','Compute the result',item_type='fill',
            answer_regions=[PageRegion(1,str(self.path),[100,80,400,200])],
            stem_region=PageRegion(1,str(self.path),[100,80,400,100]))
        self.package=ExamPackage('exam','','generic','teacher',None,1,[str(self.path)],
            [ExamSection('s','',[ExamQuestion('q1',1,'question',[self.item])])])

    def run_service(self,request,validator=None):
        review=lambda *args: {'decision':'ACCEPT','candidate_ids':[1],'reason':'visible answer'}
        service=VisualSlotSemanticService(request,'test',review_request=review)
        if validator:service.validator=validator
        # Explicit domain verifies crop-relative scaling, not page-relative.
        with patch('exam_pipeline.slot_semantics.QuestionLayoutService.corridors',return_value={'q1':[80,100,200,400]}):
            return service.enrich_package(self.package,[self.page],self.root/'audit')

    def test_visual_call_precedes_independent_local_detection(self):
        order=[]
        class Validator:
            def validate(inner,*args,**kwargs):
                order.append('local')
                return {'status':'VERIFIED','bbox':[220,115,260,145],'support':'ocr_and_pixels'}
        def request(prompt,paths,schema):
            order.append('visual')
            context=json.loads(prompt.split('\n',1)[1])
            self.assertNotIn('candidates',context)
            self.assertNotIn('standard_answer',context)
            self.assertTrue(paths[0].is_file())
            return proposal()
        self.run_service(request,Validator())
        self.assertEqual(order,['visual','local'])
        slot=self.item.slots[0]
        self.assertEqual(slot.expected_bbox,[115,220,145,260])
        self.assertNotEqual(slot.audit['proposal_bbox'],[220,115,260,145])

    def test_pre_ocr_proposal_keeps_its_original_question_domain(self):
        calls=[]
        def request(*args):
            calls.append(1);return proposal()
        service=VisualSlotSemanticService(request,'test')
        service.reviewer=None
        with patch('exam_pipeline.slot_semantics.QuestionLayoutService.corridors',
                   return_value={'q1':[80,100,200,400]}):
            service.propose_package(self.package,[self.page],self.root/'proposals')
        self.item.quality['localization']={'contexts':[{
            'page_index':1,'bbox':[150,100,350,180],'status':'QUESTION_ANCHORED'}]}
        seen=[]
        class Validator:
            def validate(inner,item,page,image,rough,domain,*args,**kwargs):
                seen.append([round(value) for value in rough])
                seen.append(list(domain))
                return {'status':'VERIFIED','bbox':[220,115,260,145],'support':'pixels'}
        service.validator=Validator()
        service.enrich_package(self.package,[self.page],self.root/'validated',reuse_proposals=True)
        self.assertEqual(calls,[1])
        self.assertEqual(seen[0],[190,104,280,152])
        self.assertEqual(seen[1],[150,100,350,180])

    def test_large_answer_view_uses_answer_corridor_not_question_bbox(self):
        self.item.item_type = 'large_writing'
        self.item.quality['localization'] = {'contexts': [{
            'page_index': 1, 'bbox': [100, 80, 400, 120],
            'question_context': [100, 80, 400, 120],
            'answer_search_domain': [20, 120, 480, 290],
            'printed_exclusion_regions': [[100, 80, 400, 120]],
            'status': 'QUESTION_ANCHORED'}]}
        views = VisualSlotSemanticService._views(self.item, [self.page], None)
        self.assertEqual(views[1]['domain'], [20, 120, 480, 290])
        self.assertEqual(views[1]['question_context'], [100, 80, 400, 120])
        self.assertEqual(views[1]['printed_exclusion_regions'], [[100, 80, 400, 120]])

    def test_page_batch_keeps_all_items_together(self):
        second = copy.deepcopy(self.item)
        second.item_id = 'q2'
        second.item_name = '2'
        second.question_text = 'Second question'
        second.quality = copy.deepcopy(self.item.quality)
        self.item.quality['localization'] = {'contexts': [{
            'page_index': 1, 'bbox': [100, 80, 400, 200],
            'answer_search_domain': [80, 80, 420, 220],
            'question_context': [100, 80, 400, 200],
            'printed_exclusion_regions': [], 'status': 'QUESTION_ANCHORED'}]}
        second.quality['localization'] = {'contexts': [{
            'page_index': 1, 'bbox': [100, 80, 400, 200],
            'answer_search_domain': [80, 80, 420, 220],
            'question_context': [100, 80, 400, 200],
            'printed_exclusion_regions': [], 'status': 'QUESTION_ANCHORED'}]}
        self.package.sections[0].questions.append(ExamQuestion(
            'q2', 2, 'Second question', [second]))
        calls = []
        def request(prompt, paths, schema):
            payload = json.loads(prompt.split('\n', 1)[1])
            calls.append(payload)
            return {'items': [
                {'item_id': 'q1','answer_layout':{'axis':'single','confidence':.9},
                 'slots': [dict(proposal()['slots'][0], label='blank_response')]},
                {'item_id': 'q2','answer_layout':{'axis':'single','confidence':.9},
                 'slots': [dict(proposal()['slots'][0], label='blank_response')]},
            ]}
        service = VisualSlotSemanticService(request, 'doubao')
        service.propose_package(self.package, [self.page], self.root / 'batch')
        self.assertEqual(len(calls), 1)
        self.assertEqual({entry['item_id'] for entry in calls[0]['items']}, {'q1', 'q2'})

    def test_overlapping_sibling_vlm_regions_are_partitioned_by_anchor_bands(self):
        second=copy.deepcopy(self.item)
        second.item_id='q1_2';second.item_name='(2)';second.question_text='Second part'
        second.stem_region=PageRegion(1,self.page.path,[100,190,400,215])
        self.item.item_id='q1_1';self.item.item_name='(1)'
        self.item.stem_region=PageRegion(1,self.page.path,[100,80,400,105])
        for item in (self.item,second):
            item.quality['localization']={'contexts':[{
                'page_index':1,'bbox':[50,60,450,290],
                'answer_search_domain':[50,60,450,290],
                'question_context':[50,60,450,290],
                'printed_exclusion_regions':[], 'status':'QUESTION_ANCHORED'}]}
        self.package.sections[0].questions[0].items=[self.item,second]
        def request(prompt,paths,schema):
            return {'items':[
                {'item_id':'q1_1','answer_layout':{'axis':'vertical','confidence':.95},
                 'slots':[dict(proposal([100,100,900,900])['slots'][0],label='working_response')]},
                {'item_id':'q1_2','answer_layout':{'axis':'vertical','confidence':.95},
                 'slots':[dict(proposal([100,100,900,900])['slots'][0],label='working_response')]},
            ]}
        service=VisualSlotSemanticService(request,'doubao')
        service.propose_package(self.package,[self.page],self.root/'sibling-batch')
        first_domain=self.item.quality['localization']['contexts'][0]['answer_search_domain']
        second_domain=second.quality['localization']['contexts'][0]['answer_search_domain']
        self.assertLessEqual(first_domain[3],second_domain[1])
        self.assertTrue(any(warning['warning']=='OVERLAPPING_SIBLING_VLM_REGIONS_PARTITIONED'
                            for warning in self.item.quality['slot_protocol_warnings']))

    def test_printed_exclusion_region_blocks_question_candidate(self):
        page = copy.deepcopy(self.page)
        page.ocr = [OCRBlock('Compute the result', [100, 80, 400, 120], .99)]
        item = copy.deepcopy(self.item)
        item.item_type = 'large_writing'
        item.stem_region = PageRegion(1, page.path, [100, 80, 400, 120])
        result = LocalSlotEvidenceValidator().validate(
            item, page, self.image, [90, 70, 410, 130], [0, 70, 500, 290],
            printed_exclusion_regions=[[100, 80, 400, 120]])
        self.assertEqual(result['status'], 'UNRESOLVED')
        self.assertIn('PRINTED_EXCLUSION_OVERLAP', result['candidates'][0]['rejection_reasons'])

    def test_printed_exclusion_is_geometric_even_for_unrelated_text(self):
        page=copy.deepcopy(self.page)
        page.ocr=[OCRBlock('unrelated symbols',[100,80,400,120],.99)]
        result=LocalSlotEvidenceValidator().validate(
            self.item,page,self.image,[90,70,410,130],[0,70,500,290],
            printed_exclusion_regions=[[100,80,400,120]])
        self.assertEqual(result['status'],'UNRESOLVED')
        self.assertIn('PRINTED_EXCLUSION_OVERLAP',result['candidates'][0]['rejection_reasons'])

    def test_merged_prompt_answer_box_uses_local_high_resolution_ocr(self):
        page=copy.deepcopy(self.page)
        page.ocr=[OCRBlock('prompt 26',[80,80,260,130],.75)]
        calls=[]
        class LocalOCR:
            def recognize_crop_high_resolution(inner,path,bbox,**kwargs):
                calls.append((list(bbox),kwargs['scale']))
                return [OCRBlock('prompt',[82,82,145,122],.99),
                        OCRBlock('26',[180,88,225,124],.98)]
        validator=LocalSlotEvidenceValidator(ocr_service=LocalOCR())
        result=validator.validate(
            self.item,page,self.image,[70,70,270,140],[100,70,300,160],
            printed_exclusion_regions=[[80,80,150,130]])
        self.assertEqual(result['status'],'VERIFIED')
        self.assertEqual(result['bbox'],[180,88,225,124])
        self.assertEqual(len(calls),1)
        self.assertEqual(result['local_ocr_redetection'][0]['status'],'REDETECTED')
        original=next(candidate for candidate in result['candidates']
                      if candidate['bbox']==[80,80,260,130])
        self.assertIn('MERGED_PRINTED_AND_ANSWER_OCR_BOX',original['rejection_reasons'])

    def test_long_answer_selects_one_spatially_continuous_cluster(self):
        item=copy.deepcopy(self.item);item.item_type='large_writing';item.stem_region=None
        page=Page(1,self.page.path,500,600,[
            OCRBlock('line a',[30,100,220,125],.92),
            OCRBlock('line b',[35,140,240,165],.91),
            OCRBlock('other a',[300,400,470,425],.94),
            OCRBlock('other b',[300,445,475,470],.93),
        ])
        result=LocalSlotEvidenceValidator().validate(
            item,page,self.image,[0,50,500,500],[0,50,500,500])
        self.assertEqual(result['status'],'VERIFIED')
        self.assertEqual(len(result['spatial_clusters']),2)
        self.assertEqual(result['selected_boxes'],[[30,100,220,125],[35,140,240,165]])

    def test_answer_like_semantic_label_is_not_retained_as_answer_content(self):
        raw=proposal();raw['slots'][0]['label']='26'
        self.run_service(lambda *args:raw)
        self.assertEqual(self.item.semantic_slot_plan[0]['label'],'blank_response')
        self.assertEqual(self.item.quality['slot_protocol_warnings'][0]['warning'],
                         'ANSWER_LIKE_LABEL_REPLACED_WITH_SEMANTIC_ROLE')

    def test_previous_question_option_cannot_win_current_fill_slot(self):
        page=copy.deepcopy(self.page)
        page.ocr=[
            OCRBlock('A.1+2x=100',[20,82,180,105],.99),
            OCRBlock('26',[430,108,470,145],.93),
            OCRBlock('4. Compute the result',[30,110,390,145],.98),
        ]
        self.item.stem_region=PageRegion(1,page.path,[30,110,390,145])
        result=LocalSlotEvidenceValidator().validate(
            self.item,page,self.image,[0,75,490,155],[0,70,500,170],
            foreign_texts=['Choose one\nA. 1+2x=100'])
        self.assertEqual(result['status'],'VERIFIED')
        self.assertEqual(result['selected_text'],['26'])
        option=next(candidate for candidate in result['candidates']
                    if candidate['text'].startswith('A.'))
        self.assertFalse(option['eligible'])
        self.assertIn('OTHER_ITEM_PRINTED_TEXT',option['rejection_reasons'])

    def test_long_ocr_line_with_small_search_overlap_is_rejected(self):
        page=copy.deepcopy(self.page)
        page.ocr=[OCRBlock('A very long printed question line',[0,100,450,130],.99)]
        result=LocalSlotEvidenceValidator().validate(
            self.item,page,self.image,[400,80,480,130],[0,0,500,200])
        self.assertEqual(result['status'],'UNRESOLVED')
        self.assertIn('LONG_LINE_PARTIAL_INTERSECTION',
                      result['candidates'][0]['rejection_reasons'])

    def test_global_assignment_uses_each_ocr_box_once(self):
        first={'bbox':[10,10,30,30],'text':'A','score':.90,'eligible':True}
        second={'bbox':[40,10,60,30],'text':'B','score':.80,'eligible':True}
        competing={'bbox':[10,10,30,30],'text':'A','score':.89,'eligible':True}
        assigned=assign_unique_candidates([
            {'page_index':1,'candidates':[first,second]},
            {'page_index':1,'candidates':[competing]},
        ])
        self.assertEqual(assigned[0]['text'],'B')
        self.assertEqual(assigned[1]['text'],'A')

    def test_visual_question_context_cannot_expand_ocr_boundary(self):
        self.item.quality['localization']={'status':'GROUNDED','contexts':[{
            'page_index':1,'bbox':[100,80,400,200],'status':'QUESTION_ANCHORED',
            'limits':[80,60,420,220]}]}
        stem=self.item.stem_region
        raw={'items':[{'item_id':'q1','regions':[{
            'page_index':1,'search_bbox':[0,0,1000,1000]}], 'confidence':.95}]}
        service=VisualQuestionLocalizer(lambda *args:raw,'test')
        service.localize(self.package,[self.page],self.root/'question-localization')
        context=self.item.quality['localization']['contexts'][0]
        self.assertEqual(context['bbox'],[100,80,400,200])
        self.assertEqual(context['status'],'QUESTION_CONTEXT_CROSS_VALIDATED')
        self.assertIs(self.item.stem_region,stem)

    def test_ocr_coordinates_replace_visual_search_hint(self):
        result=self.run_service(lambda *args:proposal())
        self.assertEqual(result['accepted'],1)
        slot=self.item.slots[0]
        self.assertTrue(slot.expected_bbox)
        self.assertEqual(slot.audit['local_validation']['status'],'VERIFIED')
        self.assertNotEqual(slot.expected_bbox,[104,190,152,280])

    def test_blank_image_cannot_promote_model_box(self):
        cv2.imwrite(str(self.path),np.full((300,500,3),255,np.uint8));self.page.ocr=[]
        self.run_service(lambda *args:proposal())
        self.assertEqual(self.item.slot_semantics_audit['status'],'PARTIAL')
        self.assertEqual(self.item.slots[0].expected_bbox,[])

    def test_isolated_punctuation_is_not_answer_evidence(self):
        im=np.full((300,500,3),255,np.uint8);cv2.circle(im,(250,130),2,(0,0,0),-1)
        cv2.imwrite(str(self.path),im);self.page.ocr=[]
        self.run_service(lambda *args:proposal())
        self.assertEqual(self.item.slots[0].expected_bbox,[])

    def test_printed_similarity_is_warning_not_coordinate_veto(self):
        im=np.full((300,500,3),255,np.uint8)
        cv2.putText(im,'Compute the result',(100,130),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),1)
        page=copy.deepcopy(self.page);page.ocr=[OCRBlock('Compute the result',[100,112,290,134],.99)]
        v=LocalSlotEvidenceValidator().validate(self.item,page,im,[95,105,300,140],[50,80,400,200])
        self.assertEqual(v['status'],'VERIFIED')
        self.assertIn('OCR_BOX_MAY_INCLUDE_PRINTED_CONTEXT',v['warnings'])

    def test_low_semantic_confidence_is_warning_when_ocr_box_is_valid(self):
        raw=proposal();raw['slots'][0]['confidence']=.25
        result=self.run_service(lambda *args:raw)
        self.assertEqual(result['accepted'],1)
        local=self.item.slots[0].audit['local_validation']
        self.assertIn('LOW_VLM_SEMANTIC_CONFIDENCE',local['warnings'])

    def test_unavailable_vlm_clears_inherited_coordinates(self):
        self.item.slots=[Slot(1,'semantic_region','q1',[100,120,150,260],1)]
        self.item.expected_slot_count=1
        service=VisualSlotSemanticService(None,'none')
        result=service.enrich_package(self.package,[self.page],self.root/'unavailable')
        self.assertEqual(result['fallback'],1)
        self.assertEqual(self.item.slots[0].expected_bbox,[])
        self.assertEqual(self.item.slots[0].audit['topology_source'],'missing_placeholder')
        self.assertEqual(self.item.slot_semantics_audit['reason'],'SEMANTIC_BACKEND_UNAVAILABLE')

    def test_invalid_normalized_coordinates_bounded(self):
        calls=[]
        def request(*args):calls.append(1);return proposal([0,0,float('nan'),900])
        result=self.run_service(request)
        self.assertEqual(len(calls),2);self.assertEqual(result['fallback'],1)
        self.assertFalse(self.item.slots)

    def test_invalid_page_cannot_map_to_other_image(self):
        self.run_service(lambda *args:proposal(page=2))
        self.assertEqual(self.item.slot_semantics_audit['reason'],'UNKNOWN_PROPOSAL_PAGE')

    def test_model_final_bbox_field_is_rejected(self):
        raw=proposal();raw['slots'][0]['bbox']=[1,2,3,4]
        self.run_service(lambda *args:raw)
        self.assertEqual(self.item.slot_semantics_audit['reason'],'INVALID_SLOT_FIELDS')

    def test_local_coordinate_failure_does_not_retry_vlm_for_pixels(self):
        calls=[]
        class Validator:
            def validate(inner,*args,**kwargs):return {'status':'UNRESOLVED','reason':'NO_OCR_COORDINATE_EVIDENCE'}
        def request(prompt,*args):
            calls.append(json.loads(prompt.split('\n',1)[1]));return proposal()
        self.run_service(request,Validator())
        self.assertEqual(len(calls),1)
        self.assertEqual(self.item.slot_semantics_audit['reason'],'OCR_SLOT_COORDINATES_UNRESOLVED')

    def test_student_identity_preserved_and_reference_passed_to_local_only(self):
        self.package.document_type='student'
        self.item.semantic_slot_plan=[{'slot_id':'q1:slot:1','index':1,'label':'vertex','anchor_before':'vertex','anchor_after':''}]
        self.run_service(lambda *args:proposal())
        self.assertEqual(self.item.slots[0].semantic_id,'q1:slot:1')
        self.assertEqual(self.item.slots[0].audit['semantic_label'],'vertex')

    def test_conflicting_regions_cannot_be_shared_by_two_logical_slots(self):
        raw=proposal();second=copy.deepcopy(raw['slots'][0]);second['index']=2;second['label']='second';raw['slots'].append(second)
        self.run_service(lambda *args:raw)
        self.assertEqual(self.item.slot_semantics_audit['status'],'PARTIAL')
        self.assertEqual(self.item.slots[1].expected_bbox,[])

    def test_cached_ocr_on_blank_page_cannot_verify_a_slot(self):
        import hashlib
        self.page.file_fingerprint=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.page.ocr_source_fingerprint=self.page.file_fingerprint
        cv2.imwrite(str(self.path),np.full((300,500,3),255,np.uint8))
        self.run_service(lambda *args:proposal())
        self.assertEqual(self.item.slots[0].expected_bbox,[])
        self.assertEqual(self.item.slots[0].audit['local_validation']['reason'],
                         'PAGE_FINGERPRINT_MISMATCH')

    def test_reference_image_does_not_change_ocr_coordinate_grounding(self):
        result=LocalSlotEvidenceValidator().validate(self.item,self.page,self.image,
            [210,105,280,155],[100,80,400,200],np.full_like(self.image,255))
        self.assertEqual(result['status'],'VERIFIED')
        self.assertEqual(result['coordinate_authority'],'ocr_boxes_only')
        self.assertNotIn('reference_used',result)

    def test_student_uses_own_anchor_domain(self):
        views=VisualSlotSemanticService._views(self.item,[self.page],[80,100,200,400],(1,[90,120,190,380]))
        self.assertEqual(views[1]['domain'],[120,90,380,190])

    def test_student_anchor_jump_to_other_question_is_not_used(self):
        views=VisualSlotSemanticService._views(self.item,[self.page],[80,100,200,400],(1,[220,120,280,380]))
        self.assertEqual(views[1]['domain'],[100,80,400,200])

    def test_overlapping_proposals_same_slot_merge_measured_regions(self):
        raw=proposal();raw['slots'][0]['regions'].append(copy.deepcopy(raw['slots'][0]['regions'][0]))
        self.run_service(lambda *args:raw)
        self.assertEqual(len(self.item.slots),1)
        self.assertEqual(self.item.slots[0].audit['local_validation']['support'],'merged_ocr_coordinates')

    def test_crosspage_proposals_keep_identity_and_local_page(self):
        second=copy.deepcopy(self.page);second.index=2
        self.item.answer_regions.append(PageRegion(2,second.path,[100,80,400,200]))
        raw=proposal();raw['slots'][0]['regions'].append({'page_index':2,'search_bbox':[300,200,600,600]})
        service=VisualSlotSemanticService(lambda *args:raw,'test',review_request=lambda *args: {'decision':'ACCEPT','candidate_ids':[1],'reason':'visible answer'})
        with patch('exam_pipeline.slot_semantics.QuestionLayoutService.corridors',return_value={'q1':[80,100,200,400]}):
            service.enrich_package(self.package,[self.page,second],self.root/'crosspage')
        self.assertEqual([s.page_index for s in self.item.slots],[1,2])
        self.assertEqual(len({s.semantic_id for s in self.item.slots}),1)
        self.assertTrue(all(s.expected_bbox for s in self.item.slots))

    def test_pipeline_skips_old_candidate_engine_when_visual_enabled(self):
        import homework_extractor as runtime
        from exam_pipeline.slots import MultiSlotTopologyEngine
        dataset=self.root/'dataset';teacher=dataset/'test'/'teacher';teacher.mkdir(parents=True)
        cv2.imwrite(str(teacher/'page_01.png'),self.image)
        def make_pages(paths,*args):
            return [Page(1,str(paths[0]),500,300,[OCRBlock('1. Compute result',[50,50,350,80],.99)])]
        def semantic(service,package,pages,output,**kwargs):
            self.assertIsNotNone(service.request)
            self.assertIn('reference_pages',kwargs)
            return {'accepted':0,'partial':1,'fallback':0}
        with patch.object(runtime,'make_pages',side_effect=make_pages), patch.object(VisualSlotSemanticService,'enrich_package',semantic), patch.object(MultiSlotTopologyEngine,'enrich_package',side_effect=AssertionError('legacy detector before model')):
            result=runtime.process(dataset,self.root/'pipeline','model','dummy','https://example.invalid','none',None,
                False,False,'eng',1,local_ocr=False,structure_vlm='none',slot_semantics='doubao')
        self.assertEqual(result['errors'],[])

    def test_tree_roundtrip_preserves_validated_geometry(self):
        self.run_service(lambda *args:proposal())
        self.package.structure_audit={'status':'COMPLETE'}
        tree=ExamTreeService.compile(self.package)
        result=copy.deepcopy(self.package);ExamTreeService.apply_to_package(tree,result)
        self.assertEqual(result.sections[0].questions[0].items[0].slots[0].expected_bbox,self.item.slots[0].expected_bbox)

if __name__=='__main__':unittest.main()
