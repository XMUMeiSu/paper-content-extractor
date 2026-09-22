"""Ground-truth based extraction evaluation; internal statuses are not accuracy.

CLI: python -m exam_pipeline.evaluation prediction.json truth.json
Truth schema: {"questions": [{"id": "q1", "items": [{"id": "q1",
"answer": "B", "slots": [{"id": "q1:slot:1", "page_index": 1,
"bbox": [x1,y1,x2,y2], "text": "B"}]}]}]}.
"""
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict


def normalize_text(text):
    return (re.sub(r"\s+", "", str(text or ""))
            .replace("−", "-").replace("（", "(").replace("）", ")"))


def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda x:max(0,x[2]-x[0])*max(0,x[3]-x[1])
    return inter/max(1,area(a)+area(b)-inter)


def equal_answer(a,b):
    if isinstance(a,list) or isinstance(b,list):
        return isinstance(a,list) and isinstance(b,list) and len(a)==len(b) and all(equal_answer(x,y) for x,y in zip(a,b))
    return a is not None and b is not None and normalize_text(a)==normalize_text(b)


def evaluate(prediction, truth):
    questions={q['question_id']:q for s in prediction.get('sections',[]) for q in s.get('questions',[])}
    items={i['item_id']:i for q in questions.values() for i in q.get('items',[])}
    expected_questions=truth['questions']; counts={'questions_expected':len(expected_questions),
        'questions_found':sum(q['id'] in questions for q in expected_questions),
        'answers_expected':0,'answers_exact':0,'slots_expected':0,'slots_matched':0,
        'slot_text_exact':0,'slots_predicted':sum(len(i.get('slots',[])) for i in items.values())}
    overlaps=[]
    for q in expected_questions:
        for t in q.get('items',[]):
            item=items.get(t['id'],{});counts['answers_expected']+=1
            key='standard_answer' if prediction.get('document_type')=='teacher' else 'student_answer'
            counts['answers_exact']+=int(equal_answer(item.get(key),t.get('answer')))
            slots = list(item.get('slots', []))
            used = set()
            for ts in t.get('slots',[]):
                counts['slots_expected']+=1
                candidates=[]
                for index, ps in enumerate(slots):
                    identity=ps.get('semantic_id') or '{}:slot:{}'.format(t['id'],ps['slot_idx'])
                    box=ps.get('handwriting_bbox')
                    if index in used or identity != ts['id'] or ps.get('page_index')!=ts['page_index'] or len(box or [])!=4:
                        continue
                    xy=[box[1],box[0],box[3],box[2]]
                    candidates.append((iou(xy,ts['bbox']),index,ps))
                if not candidates:continue
                overlap,index,ps=max(candidates,key=lambda value:value[0])
                used.add(index);overlaps.append(overlap)
                counts['slots_matched']+=int(overlap>=.5)
                counts['slot_text_exact']+=int(equal_answer(ps.get('student_answer') if key=='student_answer' else ps.get('expected_text'),ts.get('text')))
    ratio=lambda a,b:counts[a]/counts[b] if counts[b] else None
    from .stage_evaluation import evaluate_stages
    return {'stages': evaluate_stages(prediction, truth), 'counts':counts,'question_recall':ratio('questions_found','questions_expected'),
        'answer_exact_match':ratio('answers_exact','answers_expected'),
        'slot_recall_iou50':ratio('slots_matched','slots_expected'),
        'slot_precision_iou50':ratio('slots_matched','slots_predicted'),
        'slot_text_exact_match':ratio('slot_text_exact','slots_expected'),
        'mean_matched_pair_iou':sum(overlaps)/len(overlaps) if overlaps else None}


def evaluate_manifest(path):
    """Evaluate outputs by dataset dimensions without hiding missing cases.

    Manifest: {"cases": [{"prediction": "out.json", "truth": "truth.json",
    "subject": "math", "layout": "single_column", "pages": 2,
    "cohort_samples": 3}]}. Paths are relative to the manifest.
    """
    path = Path(path)
    manifest = json.loads(path.read_text())
    cases, errors = [], []
    groups = defaultdict(list)
    for index, case in enumerate(manifest['cases']):
        try:
            prediction = json.loads((path.parent / case['prediction']).read_text())
            truth = json.loads((path.parent / case['truth']).read_text())
            metrics = evaluate(prediction, truth)
            dimensions = {key: case.get(key) for key in ('subject','layout','pages','cohort_samples')}
            cases.append({'index': index, **dimensions, 'metrics': metrics})
            for key, value in dimensions.items():
                groups['{}={}'.format(key, value)].append(metrics['counts'])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({'index': index, 'code': 'EVALUATION_CASE_FAILED',
                           'error_type': type(exc).__name__})
    aggregated = {}
    for label, rows in groups.items():
        counts = {key: sum(row[key] for row in rows) for key in rows[0]}
        ratio = lambda a,b: counts[a]/counts[b] if counts[b] else None
        aggregated[label] = {'case_count': len(rows), 'counts': counts,
                            'question_recall': ratio('questions_found','questions_expected'),
                            'answer_exact_match': ratio('answers_exact','answers_expected'),
                            'slot_recall_iou50': ratio('slots_matched','slots_expected'),
                            'slot_precision_iou50': ratio('slots_matched','slots_predicted')}
    return {'status': 'PARTIAL' if errors else 'COMPLETE', 'cases': cases,
            'groups': aggregated, 'errors': errors}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prediction', nargs='?');parser.add_argument('truth', nargs='?')
    parser.add_argument('--manifest', help='Batch matrix manifest; see evaluate_manifest docstring')
    args=parser.parse_args()
    if args.manifest:
        result = evaluate_manifest(args.manifest)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result['errors']:
            raise SystemExit(1)
        return
    if not args.prediction or not args.truth:
        parser.error('provide prediction and truth, or --manifest')
    with open(args.prediction) as f:prediction=json.load(f)
    with open(args.truth) as f:truth=json.load(f)
    print(json.dumps(evaluate(prediction,truth),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
