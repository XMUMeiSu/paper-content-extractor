"""Extract clean student answers from noisy OCR text."""

import re
from typing import Optional, Tuple
from difflib import SequenceMatcher


def parse_student_answer(
    recognized_text: str,
    expected_text: Optional[str],
    slot_type: str,
    item_type: str,
    question_text: str = "",
) -> Tuple[Optional[str], dict]:
    """
    Extract clean student answer from OCR text that may contain printed content.

    Args:
        recognized_text: Raw OCR output from the slot region
        expected_text: Teacher's standard answer for reference
        slot_type: Type of slot (choice_mark, fill_blank, etc.)
        item_type: Type of item (choice, fill, etc.)
        question_text: Full question text to filter out

    Returns:
        Tuple of (clean_answer, audit_info)
    """
    audit = {"raw_text": recognized_text, "method": "unknown"}

    if not recognized_text or recognized_text.strip() == "":
        return None, {**audit, "method": "empty_input"}

    text = recognized_text.strip()

    # Choice questions: extract single letter
    if slot_type in ("choice_mark", "single_choice") or item_type in ("choice", "single", "mcq"):
        answer, choice_audit = _extract_choice_answer(text, expected_text, question_text)
        return answer, {**audit, **choice_audit}

    # Fill-in-the-blank: extract short answer
    if slot_type in ("fill_blank", "blank_line") or item_type in ("fill", "blank", "cloze"):
        answer, fill_audit = _extract_fill_answer(text, expected_text, question_text)
        return answer, {**audit, **fill_audit}

    # Large answer areas: keep full text but clean it
    if slot_type in ("free_response", "essay", "proof") or item_type in ("composition", "essay", "proof"):
        answer, essay_audit = _extract_essay_answer(text, question_text)
        return answer, {**audit, **essay_audit}

    # Default: basic cleaning
    answer = _basic_clean(text, question_text)
    return answer, {**audit, "method": "basic_clean"}


def _extract_choice_answer(text: str, expected: Optional[str], question: str) -> Tuple[Optional[str], dict]:
    """Extract single choice letter (A/B/C/D/E/F/G/H) from text."""
    audit = {"method": "choice_extraction"}

    # Remove question text fragments
    if question:
        cleaned = _remove_question_fragments(text, question)
    else:
        cleaned = text

    # Find all uppercase letters A-H
    letters = re.findall(r'\b([A-H])\b', cleaned, re.IGNORECASE)

    if not letters:
        # Try finding with punctuation: A. B) C、
        letters = re.findall(r'([A-H])[.、)）]', cleaned, re.IGNORECASE)

    if not letters:
        # Try finding without word boundary
        letters = re.findall(r'([A-H])', cleaned, re.IGNORECASE)

    # Normalize to uppercase
    letters = [l.upper() for l in letters]

    # Filter out letters that appear in the question
    if question:
        question_letters = set(re.findall(r'\b([A-H])\b', question, re.IGNORECASE))
        letters = [l for l in letters if l.upper() not in question_letters]

    if not letters:
        audit["candidate_count"] = 0
        return None, audit

    # If multiple candidates, prefer the one matching expected answer
    audit["candidate_count"] = len(letters)
    audit["candidates"] = letters

    if expected and expected.upper() in letters:
        audit["matched_expected"] = True
        return expected.upper(), audit

    # Return the first candidate (most likely to be student's answer)
    audit["matched_expected"] = False
    return letters[0], audit


def _extract_fill_answer(text: str, expected: Optional[str], question: str) -> Tuple[Optional[str], dict]:
    """Extract fill-in-the-blank answer from text."""
    audit = {"method": "fill_extraction"}

    # Remove question text fragments
    if question:
        cleaned = _remove_question_fragments(text, question)
    else:
        cleaned = text

    # Remove common question patterns
    cleaned = re.sub(r'^[\d一二三四五六七八九十]+[.、．）\)]', '', cleaned)  # Remove numbering
    cleaned = re.sub(r'答[:：]?', '', cleaned)  # Remove "答："
    cleaned = re.sub(r'解[:：]?', '', cleaned)  # Remove "解："

    # If text is still very long after cleaning, likely wrong OCR region
    if len(cleaned) > 100:
        audit["warning"] = "text_too_long_after_cleaning"
        # Try to extract numbers or short words as fallback
        numbers = re.findall(r'-?\d+(?:\.\d+)?', cleaned)
        if numbers:
            answer = '; '.join(numbers[:3])
            audit["fallback"] = "numbers_only"
            return answer, audit

        # Extract first 10 chars as degraded fallback
        answer = cleaned[:10].strip()
        audit["fallback"] = "truncated"
        return answer if answer else None, audit

    # If text is already short, use it directly
    if len(cleaned) <= 30:
        answer = cleaned.strip()
        audit["short_text"] = True
        return answer if answer else None, audit

    # Try to extract structured content

    # Pattern 1: Number (integer or decimal)
    numbers = re.findall(r'-?\d+(?:\.\d+)?', cleaned)
    if numbers and len(numbers) <= 3:
        answer = '; '.join(numbers)
        audit["pattern"] = "numbers"
        return answer, audit

    # Pattern 2: Mathematical expression (preserve operators)
    math_expr = re.search(r'[-+]?\d+(?:\.\d+)?(?:\s*[+\-*/×÷]\s*\d+(?:\.\d+)?)*', cleaned)
    if math_expr:
        answer = math_expr.group(0).strip()
        audit["pattern"] = "math_expression"
        return answer, audit

    # Pattern 3: Short phrase (no more than 20 chars, not a full sentence)
    sentences = re.split(r'[。！？.!?]', cleaned)
    for sent in sentences:
        sent = sent.strip()
        if 1 <= len(sent) <= 20 and not re.search(r'[，,、]', sent):
            audit["pattern"] = "short_phrase"
            return sent, audit

    # Pattern 4: Text between quotes
    quoted = re.findall(r'["""\'\'](.*?)["""\'\']', cleaned)
    if quoted:
        answer = quoted[0].strip()
        audit["pattern"] = "quoted_text"
        return answer, audit

    # Fallback: return cleaned text (may still be noisy)
    answer = cleaned[:50].strip()  # Limit to 50 chars
    audit["pattern"] = "truncated"
    return answer if answer else None, audit



def _extract_essay_answer(text: str, question: str) -> Tuple[Optional[str], dict]:
    """Extract essay/proof answer, preserving full content but removing question."""
    audit = {"method": "essay_extraction"}

    if question:
        cleaned = _remove_question_fragments(text, question)
    else:
        cleaned = text

    # Remove common leading patterns
    cleaned = re.sub(r'^[\d一二三四五六七八九十]+[.、．）\)]', '', cleaned)
    cleaned = re.sub(r'^(答|解|证明|证)[:：]?', '', cleaned)

    answer = cleaned.strip()
    audit["length"] = len(answer)
    return answer if answer else None, audit


def _remove_question_fragments(text: str, question: str) -> str:
    """Remove fragments of question text from the recognized text."""
    # 1. Remove common header patterns
    garbage_patterns = [
        r'\d{4}[—\-]\d{4}学年',  # Academic year
        r'作业\d+',              # Homework number
        r'[年级班座]号?[:：]',    # Grade, class, seat number
        r'姓名[:：]',             # Name
        r'得分[:：]',             # Score
        r'命题人[:：]',           # Question setter
        r'审核人[:：]',           # Reviewer
        r'CHOOL',                # Common OCR error for SCHOOL
        r'[A-Z]{3,}',            # Long uppercase sequences (likely headers)
    ]

    for pattern in garbage_patterns:
        text = re.sub(pattern, '', text)

    # 2. Normalize spaces
    text = re.sub(r'\s+', '', text)
    question = re.sub(r'\s+', '', question)

    # 3. Find longest common substring
    matcher = SequenceMatcher(None, question, text)
    match = matcher.find_longest_match(0, len(question), 0, len(text))

    if match.size > 15:  # Lower threshold for more aggressive removal
        # Remove the matching part from text
        before = text[:match.b]
        after = text[match.b + match.size:]
        text = (before + after).strip()

    # 4. Remove question markers
    question_markers = ['下列', '以下', '正确的是', '错误的是', '下面', '如图', '已知']
    for marker in question_markers:
        if marker in question:
            text = text.replace(marker, '')

    # 5. If text is still very long (>200 chars), likely contains question
    if len(text) > 200:
        # Try to extract shortest meaningful segment
        lines = text.split('\n')
        if lines:
            shortest_line = min(lines, key=len)
            if len(shortest_line) < len(text) / 2:
                text = shortest_line

    return text.strip()

    result = []
    i = 0
    while i < len(text):
        # Check if next 5 chars are similar to question
        chunk = text[i:i+5]
        if len(chunk) < 3:
            result.append(text[i])
            i += 1
            continue

        chunk_trigrams = set(chunk[j:j+3] for j in range(len(chunk)-2))
        similarity = len(chunk_trigrams & question_words) / max(len(chunk_trigrams), 1)

        if similarity > 0.6:  # High similarity, skip
            i += 1
        else:
            result.append(text[i])
            i += 1

    return ''.join(result).strip()


def _basic_clean(text: str, question: str = "") -> str:
    """Basic text cleaning: trim, normalize spaces."""
    cleaned = text.strip()

    # Remove question if present
    if question and len(question) > 10:
        cleaned = _remove_question_fragments(cleaned, question)

    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)

    return cleaned


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.

    - Remove spaces, punctuation
    - Convert math symbols: × → *, ÷ → /
    - Convert numbers: 1.0 → 1
    """
    if not answer:
        return ""

    # Remove spaces
    normalized = re.sub(r'\s+', '', answer)

    # Convert math symbols
    normalized = normalized.replace('×', '*').replace('÷', '/')
    normalized = normalized.replace('﹣', '-').replace('—', '-')

    # Normalize decimal: 1.0 → 1
    normalized = re.sub(r'\.0+\b', '', normalized)

    # Remove common punctuation
    normalized = re.sub(r'[,，、;；。.！!？?]', '', normalized)

    return normalized.lower()
