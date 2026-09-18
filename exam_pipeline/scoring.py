"""Score-tree conservation and optional downstream objective grading."""
import re
from typing import Any, Dict, List
from .contracts import ExamPackage


def _items(package):
    return [item for section in package.sections for question in section.questions for item in question.items]


def balance_score_tree(package: ExamPackage) -> List[str]:
    warnings: List[str] = []
    prior_status = str(package.score_audit.get("status", ""))
    for section in package.sections:
        for question in section.questions:
            values = [float(item.item_score) for item in question.items if item.item_score is not None]
            derived = sum(values) if values else None
            if derived is not None and question.question_score is not None and abs(float(question.question_score) - derived) > 0.01:
                warnings.append(f"{question.question_id} 分值由 {question.question_score:g} 修复为 {derived:g}")
            if derived is not None:
                question.question_score = derived
        q_values = [float(question.question_score) for question in section.questions
                    if question.question_score is not None]
        derived_section = sum(q_values) if q_values else None
        if derived_section is not None and section.section_score is not None and abs(float(section.section_score) - derived_section) > 0.01:
            warnings.append(f"{section.section_id} 分值修复为 {derived_section:g}")
        if derived_section is not None:
            section.section_score = derived_section
    section_values = [float(section.section_score) for section in package.sections
                      if section.section_score is not None]
    derived_total = sum(section_values) if section_values else None
    if derived_total is not None:
        if package.total_score is not None and abs(float(package.total_score) - derived_total) > 0.01:
            if package.declared_total_score is None:
                package.declared_total_score = package.total_score
            warnings.append(f"总分由 {package.total_score:g} 修复为 {derived_total:g}")
        package.total_score = derived_total
    status = "REPAIRED" if warnings or prior_status == "REPAIRED" else "OK"
    package.score_audit.update({"status": status, "derived_total_score": derived_total})
    return warnings


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def grade_objective_items(package: ExamPackage) -> Dict[str, int]:
    auto_graded = 0
    needs_review = 0
    for item in _items(package):
        kind = str(item.item_type or "").lower()
        is_choice = any(token in kind for token in (
            "choice", "single", "judgment", "选择", "判断"
        ))
        if is_choice and item.standard_answer is not None and item.student_answer is not None:
            correct = _normalized(item.standard_answer) == _normalized(item.student_answer)
            item.is_correct = correct
            item.student_score = float(item.item_score or 0.0) if correct else 0.0
            item.eval_status = "auto_graded"
            auto_graded += 1
        else:
            item.student_score = None
            item.is_correct = None
            item.eval_status = "needs_review"
            needs_review += 1
    return {"auto_graded": auto_graded, "needs_review": needs_review}
