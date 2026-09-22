"""Extract clean student answers from noisy OCR text."""

import re
from typing import Optional, Tuple
from difflib import SequenceMatcher

CHOICE_MARKS = {"✓", "✔", "√", "☑", "勾", "check", "tick"}


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
    if (slot_type in ("choice_mark", "single_choice") or "choice" in str(slot_type).casefold()
            or item_type in ("choice", "single", "mcq")):
        answer, choice_audit = _extract_choice_answer(text, expected_text, question_text)
        return answer, {**audit, **choice_audit}

    # The recognition layer already rejects contamination. Preserve valid
    # transcription verbatim, including copied givens, units and proof steps.
    return text, {**audit, "method": "verbatim_transcription"}


def _extract_choice_answer(text: str, expected: Optional[str], question: str) -> Tuple[Optional[str], dict]:
    """Extract single choice letter (A/B/C/D/E/F/G/H) from text."""
    audit = {"method": "choice_extraction"}

    # Remove question text fragments
    if question:
        cleaned = _remove_question_fragments(text, question)
    else:
        cleaned = text

    if cleaned.strip().casefold() in CHOICE_MARKS:
        audit.update(candidate_count=1, candidates=["✓"], marker=True)
        return "✓", audit

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

    if not letters:
        audit["candidate_count"] = 0
        return None, audit

    # Keep candidates only to detect ambiguity; reference text is ignored.
    audit["candidate_count"] = len(letters)
    audit["candidates"] = letters

    # Ambiguous observations stay unresolved; teacher truth is not OCR evidence.
    candidates = list(dict.fromkeys(letters))
    audit["matched_expected"] = False
    if len(candidates) != 1:
        audit["warning"] = "ambiguous_choice"
        return None, audit
    return candidates[0], audit



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

    # Preserve a complete observation. Never turn a long contaminated crop
    # into a plausible answer by selecting its first numbers or truncating it.
    audit["method"] = "verbatim_fill"
    return cleaned.strip() or None, audit


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
