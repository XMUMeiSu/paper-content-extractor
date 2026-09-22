"""Teacher answer quality assessment and auto-correction strategies."""
from typing import Any, Dict, List, Optional, Tuple
import re


class TeacherAnswerQualityAssessor:
    """评估教师手写答案提取的质量，并提供自动修正建议。

    生产级流水线需要自动识别哪些答案可以直接使用，哪些需要人工确认。
    """

    # 高置信度答案模式（选择题）
    CHOICE_PATTERNS = {
        'single': re.compile(r'^[A-H]$'),
        'multiple': re.compile(r'^[A-H]{2,4}$'),
        'mark': re.compile(r'^[✓✔√☑勾]$'),
    }

    # 常见 OCR 错误映射
    OCR_CORRECTIONS = {
        # 选择题常见错误
        'A.': 'A', 'B.': 'B', 'C.': 'C', 'D.': 'D',
        'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D',
        '|A': 'A', '|B': 'B', '|C': 'C', '|D': 'D',

        # 数字常见错误
        'O': '0', 'o': '0', 'l': '1', 'I': '1',

        # 中文常见错误
        '不叫以': '不可以', '木': '不', '太': '大',
    }

    @classmethod
    def assess_answer(cls, text: str, item_type: str,
                     ocr_confidence: Optional[float] = None,
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """评估单个答案的质量。

        Args:
            text: OCR 识别的文本
            item_type: 题目类型 (choice/fill_blank/short_answer)
            ocr_confidence: OCR 引擎返回的置信度
            context: 题目上下文（题干、选项等）

        Returns:
            {
                'confidence': float,  # 0.0-1.0，综合置信度
                'quality': str,  # 'high'/'medium'/'low'
                'auto_corrected': bool,  # 是否自动修正
                'corrected_text': str,  # 修正后的文本
                'needs_review': bool,  # 是否需要人工确认
                'issues': List[str],  # 发现的问题
            }
        """
        issues = []
        corrected_text = text
        auto_corrected = False

        # 1. 清理空白字符
        corrected_text = corrected_text.strip()

        # 2. 应用 OCR 常见错误修正
        # Only normalize choice presentation; never change lexical content.
        if item_type == 'choice':
            import re
            match = re.fullmatch(r'\s*([A-Ha-h])\s*[.、]?\s*', corrected_text)
            if match:
                corrected_text = match.group(1).upper()
                auto_corrected = corrected_text != text.strip()

        # 3. 根据题目类型评估
        if item_type == 'choice':
            return cls._assess_choice_answer(
                corrected_text, ocr_confidence, issues, auto_corrected, text
            )
        elif item_type == 'fill_blank':
            return cls._assess_fill_blank_answer(
                corrected_text, ocr_confidence, issues, auto_corrected, text, context
            )
        else:
            return cls._assess_short_answer(
                corrected_text, ocr_confidence, issues, auto_corrected, text
            )

    @classmethod
    def _assess_choice_answer(cls, text: str, ocr_conf: Optional[float],
                             issues: List[str], auto_corrected: bool,
                             original_text: str) -> Dict[str, Any]:
        """评估选择题答案。"""
        # 单选题
        if cls.CHOICE_PATTERNS['single'].match(text):
            confidence = 0.95 if ocr_conf and ocr_conf > 0.8 else 0.85
            return {
                'confidence': confidence,
                'quality': 'high',
                'auto_corrected': auto_corrected,
                'corrected_text': text,
                'needs_review': False,
                'issues': issues,
            }

        # 多选题
        if cls.CHOICE_PATTERNS['multiple'].match(text):
            confidence = 0.90 if ocr_conf and ocr_conf > 0.7 else 0.75
            return {
                'confidence': confidence,
                'quality': 'high',
                'auto_corrected': auto_corrected,
                'corrected_text': text,
                'needs_review': False,
                'issues': issues,
            }

        if cls.CHOICE_PATTERNS['mark'].match(text):
            return {
                'confidence': 0.85 if ocr_conf is None or ocr_conf >= .65 else 0.75,
                'quality': 'high',
                'auto_corrected': auto_corrected,
                'corrected_text': text,
                'needs_review': False,
                'issues': issues,
            }

        # 不符合选择题格式
        issues.append(f"选择题答案格式异常: '{text}'")
        return {
            'confidence': 0.3,
            'quality': 'low',
            'auto_corrected': auto_corrected,
            'corrected_text': text,
            'needs_review': True,
            'issues': issues,
        }

    @classmethod
    def _assess_fill_blank_answer(cls, text: str, ocr_conf: Optional[float],
                                 issues: List[str], auto_corrected: bool,
                                 original_text: str,
                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """评估填空题答案。"""
        # 检查长度合理性
        if len(text) == 0:
            issues.append("答案为空")
            return {
                'confidence': 0.0,
                'quality': 'low',
                'auto_corrected': False,
                'corrected_text': text,
                'needs_review': True,
                'issues': issues,
            }

        # 短答案（1-3字）：高置信度
        if len(text) <= 3:
            confidence = 0.85 if ocr_conf and ocr_conf > 0.7 else 0.70
            needs_review = confidence < 0.75
            quality = 'high' if confidence >= 0.80 else 'medium'
            return {
                'confidence': confidence,
                'quality': quality,
                'auto_corrected': auto_corrected,
                'corrected_text': text,
                'needs_review': needs_review,
                'issues': issues,
            }

        # 中等长度（4-10字）
        if len(text) <= 10:
            confidence = 0.75 if ocr_conf and ocr_conf > 0.6 else 0.60
            needs_review = confidence < 0.70 or auto_corrected
            quality = 'medium'
            return {
                'confidence': confidence,
                'quality': quality,
                'auto_corrected': auto_corrected,
                'corrected_text': text,
                'needs_review': needs_review,
                'issues': issues,
            }

        # 长答案（>10字）：需要人工确认
        issues.append(f"长答案需要确认: {len(text)}字")
        return {
            'confidence': 0.50,
            'quality': 'medium',
            'auto_corrected': auto_corrected,
            'corrected_text': text,
            'needs_review': True,
            'issues': issues,
        }

    @classmethod
    def _assess_short_answer(cls, text: str, ocr_conf: Optional[float],
                            issues: List[str], auto_corrected: bool,
                            original_text: str) -> Dict[str, Any]:
        """评估简答题答案。"""
        if len(text) == 0:
            issues.append("答案为空")
            return {
                'confidence': 0.0,
                'quality': 'low',
                'auto_corrected': False,
                'corrected_text': text,
                'needs_review': True,
                'issues': issues,
            }

        # 简答题通常需要人工确认
        confidence = 0.60 if ocr_conf and ocr_conf > 0.5 else 0.40
        return {
            'confidence': confidence,
            'quality': 'medium',
            'auto_corrected': auto_corrected,
            'corrected_text': text,
            'needs_review': True,
            'issues': issues,
        }

    @classmethod
    def batch_assess(cls, answers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量评估答案质量，生成 HITL 任务。

        Args:
            answers: List of {item_id, text, item_type, ocr_confidence, ...}

        Returns:
            {
                'total': int,
                'high_confidence': int,  # 可以自动使用
                'needs_review': int,  # 需要人工确认
                'auto_corrected': int,  # 自动修正数量
                'hitl_tasks': List[Dict],  # HITL 确认任务
                'summary': Dict,
            }
        """
        results = []
        hitl_tasks = []

        for answer in answers:
            assessment = cls.assess_answer(
                answer['text'],
                answer.get('item_type', 'fill_blank'),
                answer.get('ocr_confidence'),
                answer.get('context')
            )

            result = {**answer, **assessment}
            results.append(result)

            if assessment['needs_review']:
                hitl_tasks.append({
                    'item_id': answer['item_id'],
                    'original_text': answer['text'],
                    'corrected_text': assessment['corrected_text'],
                    'confidence': assessment['confidence'],
                    'issues': assessment['issues'],
                    'item_type': answer.get('item_type'),
                })

        high_conf = sum(1 for r in results if r['quality'] == 'high' and not r['needs_review'])
        needs_review = sum(1 for r in results if r['needs_review'])
        auto_corrected = sum(1 for r in results if r['auto_corrected'])

        return {
            'total': len(results),
            'high_confidence': high_conf,
            'needs_review': needs_review,
            'auto_corrected': auto_corrected,
            'results': results,
            'hitl_tasks': hitl_tasks,
            'summary': {
                'auto_accept_rate': f"{high_conf/len(results)*100:.1f}%" if results else "0%",
                'review_rate': f"{needs_review/len(results)*100:.1f}%" if results else "0%",
            }
        }
