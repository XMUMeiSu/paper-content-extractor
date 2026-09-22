"""Document-level OCR evidence fusion and bounded structural recovery."""
import copy
import re
from statistics import median
from pathlib import Path
from .contracts import ExamItem, ExamQuestion, ExamSection, PageRegion
from .reading_order import column_groups, row_order

# OCR frequently drops the punctuation after a question number.  A bounded
# whitespace form is accepted as a structural anchor; ordinary body numbers
# remain excluded by the left-margin and reading-order checks below.
TOP = re.compile(r'^\s*(\d{1,3})\s*(?:[.、。．:：)）](?!\d)|\s+)(.*)$')
SUB = re.compile(r'[（(]\s*(\d{1,2})\s*[）)]')


def anchors(page):
    candidates = []
    for group in column_groups(page.ocr, page.width or 1654):
        if not group:
            continue
        left = min(b.bbox[0] for b in group if len(b.bbox) == 4)
        # Restrict numbered anchors to the text-column margin, not formula operands.
        for block in group:
            match = TOP.match(block.text or '')
            if match and len(block.bbox) == 4 and block.bbox[0] <= left + (page.width or 1654)*.12:
                candidates.append((int(match.group(1)), block))
    return candidates


def select_structure_pages(originals, templates):
    """Use templates only when their OCR preserves original printed evidence."""
    template_map = {p.index: p for p in templates}
    chosen, audit = [], []
    for page in originals:
        template = template_map.get(page.index)
        original_ids = {n for n, _ in anchors(page)}
        template_ids = {n for n, _ in anchors(template)} if template else set()
        original_chars = sum(len(b.text) for b in page.ocr)
        template_chars = sum(len(b.text) for b in template.ocr) if template else 0
        coverage = len(original_ids & template_ids)/len(original_ids) if original_ids else 1.0
        retention = template_chars/max(1, original_chars)
        accepted = bool(template and template.ocr and coverage == 1.0 and retention >= .45)
        selected = copy.deepcopy(template if accepted else page)
        # Recover independently visible number anchors even from a rejected
        # template, without copying its damaged body text.
        if template:
            known = {n for n, _ in anchors(selected)}
            for number, block in anchors(template if not accepted else page):
                if number not in known and (block.confidence is None or block.confidence >= .6):
                    selected.ocr.append(copy.deepcopy(block))
                    known.add(number)
        chosen.append(selected)
        audit.append({'page_index': page.index, 'source': 'template' if accepted else 'original',
                      'original_question_numbers': sorted(original_ids),
                      'template_question_numbers': sorted(template_ids),
                      'anchor_coverage': coverage, 'text_retention': round(retention, 4),
                      'template_accepted': accepted,
                      'fallback_reason': None if accepted else 'TEMPLATE_EVIDENCE_INCOMPLETE'})
    return chosen, audit


def infer_kind(text):
    if len(set(re.findall(r'(?:^|\s)([A-H])[.、．)]', text))) >= 2:
        return 'choice'
    if len(SUB.findall(text)) >= 2:
        return 'solve'
    if re.search(r'(?:图象|图像|表达式|选出|下列|以下|选择|Which|Choose).*[（(][^）)]{0,3}[）)]\s*', text, re.I):
        return 'choice'
    if re.search(r'_{2,}|填空|大小关系|值为|个数是|开口方向|坐标是|写出一个|取值范围为|Fill in', text, re.I):
        return 'fill'
    return 'solve'


def build_sections(pages, role):
    """Row-aware fallback. No fabricated question when a page has no anchor."""
    sections = []
    used_ids = set()
    previous_question = None
    for page in pages:
        for column_index, group in enumerate(column_groups(page.ocr, page.width or 1654)):
            if not group:
                continue
            candidate = {id(b): n for n, b in anchors(page)}
            starts = [(i, candidate[id(b)], b) for i, b in enumerate(group) if id(b) in candidate]
            if not starts:
                # Explicit subquestion continuation only; headers are not continuations.
                if previous_question and SUB.match(group[0].text.strip()):
                    old = previous_question.items[-1]
                    box = [min(b.bbox[0] for b in group), min(b.bbox[1] for b in group),
                           max(b.bbox[2] for b in group), max(b.bbox[3] for b in group)]
                    old.answer_regions.append(PageRegion(page.index, page.path, box, None, ''))
                    old.question_text += '\n' + '\n'.join(b.text for b in group)
                    old.is_cross_page = True
                continue
            section = ExamSection('page_{}_column_{}'.format(page.index, column_index+1), '', [])
            for position, (start, number, anchor) in enumerate(starts):
                end = starts[position+1][0] if position+1 < len(starts) else len(group)
                blocks = group[start:end]
                for adjacent in group[:start]:
                    if adjacent.bbox[0] > anchor.bbox[2] and abs((adjacent.bbox[1]+adjacent.bbox[3]-anchor.bbox[1]-anchor.bbox[3])/2) < (anchor.bbox[3]-anchor.bbox[1])*.5:
                        if adjacent not in blocks:
                            blocks.insert(1, adjacent)
                text = '\n'.join(b.text.strip() for b in blocks if b.text.strip())
                kind = infer_kind(text)
                submarkers = list(SUB.finditer(text))
                seen = set()
                for marker in submarkers:
                    number_text = marker.group(1)
                    if number_text in seen and len(seen) >= 2:
                        text = text[:marker.start()].rstrip()
                        break
                    seen.add(number_text)
                left = min(b.bbox[0] for b in blocks)
                right = max(b.bbox[2] for b in blocks)
                bottom = (starts[position+1][2].bbox[1]-4 if position+1 < len(starts)
                          else (page.height or 2338)-20)
                # One coarse region is a search domain, never a verified answer.
                region = PageRegion(page.index, page.path,
                                    [max(0,left-12),anchor.bbox[1],min(page.width or 1654,right+12),
                                     max(anchor.bbox[3],bottom)], None, '')
                question_id = 'q{}'.format(number)
                if question_id in used_ids:
                    question_id += '_p{}_c{}_{}'.format(page.index,column_index+1,position+1)
                used_ids.add(question_id)
                item = ExamItem(question_id, '第{}题'.format(number), text,
                                item_type=kind, answer_regions=[region],
                                student_regions=[copy.deepcopy(region)] if role == 'student' else [],
                                stem_region=PageRegion(page.index,page.path,list(anchor.bbox),anchor.confidence,anchor.text))
                question = ExamQuestion(question_id, number, text, [item])
                section.questions.append(question)
                previous_question = question
            sections.append(section)
    return sections


def recover_structure(package, selected_pages, original_pages):
    """One evidence-driven repair pass; missing anchors are never silently dropped."""
    evidence = {(p.index,n) for p in original_pages for n,_ in anchors(p)}
    evidence |= {(p.index,n) for p in selected_pages for n,_ in anchors(p)}
    def covered():
        return {(r.page_index,q.question_num) for s in package.sections for q in s.questions
                for i in q.items for r in ([i.stem_region] if i.stem_region else i.answer_regions)}
    before = sorted(evidence-covered())
    if before:
        selected_map = {p.index:p for p in selected_pages}
        for page_index, number in before:
            fallback = build_sections([selected_map[page_index]], package.document_type)
            for section in fallback:
                for question in section.questions:
                    if question.question_num == number and (page_index,number) not in covered():
                        package.sections.append(ExamSection('recovered_{}_{}'.format(page_index,number), '', [question]))
    missing = sorted(evidence-covered())
    covered_pages = {p for p,n in covered()}
    empty_pages = [p.index for p in original_pages if p.ocr and p.index not in covered_pages]
    package.structure_audit.update({'expected_anchors': [list(x) for x in sorted(evidence)],
        'missing_before_repair': [list(x) for x in before], 'missing_anchors': [list(x) for x in missing],
        'unexplained_pages': empty_pages, 'repair_attempts': int(bool(before)),
        'coverage_scope': 'detected_question_anchors_only',
        'status': 'COMPLETE' if evidence and not missing and not empty_pages else 'UNRESOLVED'})
    return package.structure_audit


def recover_unread_lines(pages, ocr, engine, language, max_regions=4):
    """Bounded second OCR pass on long ink rows missed by page detection.

    The retry is driven by image coverage, not expected question numbers. It
    can recover inline subquestions or a detached question number without
    inventing a missing sequence member.
    """
    import cv2
    import numpy as np
    audit=[]
    if engine == 'none':
        return audit
    for page in pages:
        image=cv2.imread(page.path)
        if image is None:
            audit.append({'page_index':page.index,'code':'PAGE_UNAVAILABLE'})
            continue
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
        lines=cv2.morphologyEx(ink,cv2.MORPH_CLOSE,np.ones((3,25),np.uint8))
        _,_,stats,_=cv2.connectedComponentsWithStats(lines,8)
        candidates=[]
        for x,y,w,h,area in stats[1:]:
            if w < image.shape[1]*.35 or h < 12 or h > max(40, image.shape[0]*.06):
                continue
            union=np.zeros((h,w),np.uint8)
            for b in page.ocr:
                if len(b.bbox)!=4:continue
                bx1,by1,bx2,by2=[int(v) for v in b.bbox]
                rx1,ry1=max(0,bx1-x),max(0,by1-y)
                rx2,ry2=min(w,bx2-x),min(h,by2-y)
                if rx2>rx1 and ry2>ry1:union[ry1:ry2,rx1:rx2]=1
            covered=float(union.sum())/max(1,w*h)
            if covered < .45:
                candidates.append((covered,[int(x),int(y),int(x+w),int(y+h)]))
        for coverage,box in sorted(candidates,key=lambda v:v[0])[:max_regions]:
            try:
                blocks=ocr.recognize_crop(Path(page.path),box,padding=12,engine=engine,language=language)
                added=0
                for b in blocks:
                    if len(b.bbox)!=4 or (b.confidence is not None and b.confidence<.5):continue
                    if any(b.text==old.text and abs(b.bbox[1]-old.bbox[1])<15 for old in page.ocr if len(old.bbox)==4):continue
                    page.ocr.append(b);added+=1
                audit.append({'page_index':page.index,'bbox':box,'reason':'UNCOVERED_INK_ROW','added_blocks':added})
            except Exception as exc:
                audit.append({'page_index':page.index,'bbox':box,'code':'OCR_RETRY_FAILED','error_type':type(exc).__name__})
    return audit
