"""Evidence-only answer recognition with dedicated choice and formula routes."""
import re
import tempfile
from pathlib import Path

from .ocr import OCRService
from .reading_order import row_order
from .roi import yxyx_to_xyxy


WARNING_ISSUES = {"LOW_OCR_CONFIDENCE"}
CHOICE_MARKS = {"✓", "✔", "√", "☑", "勾", "check", "tick"}
_VISUAL_UNSET = object()


def normalize_text(text):
    return (re.sub(r"\s+", "", str(text or ""))
            .replace("−", "-").replace("（", "(").replace("）", ")"))


def _comparison_value(text, formula=False):
    value = normalize_text(text)
    if formula:
        value = value.replace("$", "").replace("\\left", "").replace("\\right", "")
    return value


def normalize_choice(text):
    """Normalize only an observed choice token; never infer an option from a mark."""
    value = normalize_text(text).replace("［", "[").replace("］", "]")
    if value.casefold() in CHOICE_MARKS:
        return "✓"
    match = re.fullmatch(r"[\(\[]?([A-Ha-h])[\)\]]?[.。、]?", value)
    if match:
        return match.group(1).upper()
    match = re.fullmatch(r"[\(\[]?([A-Ha-h](?:[,，、/][A-Ha-h])+)[\)\]]?", value)
    if match:
        return "".join(re.findall(r"[A-H]", match.group(1).upper()))
    if re.fullmatch(r"[A-Ha-h]{2,8}", value):
        return value.upper()
    return value


def _balanced(value):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    escaped = False
    for char in value:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False
    return not stack


def _latex_group(value, offset):
    """Return contents and next offset for one required LaTeX group."""
    while offset < len(value) and value[offset].isspace():
        offset += 1
    if offset >= len(value) or value[offset] != "{":
        return None, offset
    depth = 0
    for index in range(offset, len(value)):
        if value[index] == "{":
            depth += 1
        elif value[index] == "}":
            depth -= 1
            if depth == 0:
                return value[offset + 1:index], index + 1
    return None, offset


def formula_structure_issues(text):
    """Validate visible formula structure without comparing with a reference answer."""
    value = str(text or "").strip()
    compact = normalize_text(value)
    issues = []
    if not _balanced(value):
        issues.append("UNBALANCED_EXPRESSION")
    for token in ("^", "_"):
        for match in re.finditer(re.escape(token), compact):
            tail = compact[match.end():]
            if not tail or tail.startswith("{}") or tail[0] in ")]}=+-*/^_":
                issues.append("INCOMPLETE_SCRIPT")
                break
    fraction_commands = tuple(re.finditer(r"\\(?:d?frac|tfrac)", value))
    fraction_count = len(fraction_commands)
    offset = 0
    complete_fractions = 0
    while True:
        match = re.search(r"\\(?:d?frac|tfrac)", value[offset:])
        if not match:
            break
        start = offset + match.start()
        command_end = offset + match.end()
        numerator, next_offset = _latex_group(value, command_end)
        denominator, end_offset = _latex_group(value, next_offset)
        if numerator and denominator:
            complete_fractions += 1
            offset = end_offset
        else:
            offset = max(start + len("\\frac"), next_offset + 1)
    if fraction_count != complete_fractions or re.search(r"(?<!\\)frac\s*[({]", value):
        issues.append("MALFORMED_FRACTION")
    if re.search(r"(?:[=+\-*/]|\\times|\\div)\s*[)\]}]", compact):
        issues.append("MISSING_OPERAND")
    if re.search(r"(?<![A-Za-z])(?:rac\}|xt\})", compact):
        issues.append("MALFORMED_FORMULA")
    return list(dict.fromkeys(issues))


def formula_structure(text):
    """Auditable structural signature for a formula transcription."""
    value = str(text or "")
    issues = formula_structure_issues(value)
    return {
        "balanced_delimiters": "UNBALANCED_EXPRESSION" not in issues,
        "fractions_complete": "MALFORMED_FRACTION" not in issues,
        "scripts_complete": "INCOMPLETE_SCRIPT" not in issues,
        "operands_complete": "MISSING_OPERAND" not in issues,
        "minus_count": normalize_text(value).count("-"),
        "has_minus_sign": "-" in normalize_text(value),
        "issues": issues,
    }


def _is_choice(kind):
    value = str(kind or "").casefold()
    return any(token in value for token in ("choice", "single", "multiple", "mcq", "judgment", "选择", "判断"))


def _is_fill(kind):
    value = str(kind or "").casefold()
    return any(token in value for token in ("fill", "blank", "cloze", "completion", "填空"))


def _is_formula(question, kind, observations=""):
    value = " ".join((str(question or ""), str(kind or ""), str(observations or "")))
    if any(token in str(kind or "").casefold() for token in ("formula", "equation", "calculation", "公式", "方程")):
        return True
    return bool(re.search(r"[=√∑∫^]|\\(?:frac|sqrt|sum|int)|[A-Za-z]\s*[+−-]\s*\d", value))


def content_issues(text, question="", kind="", confidence=None, formula=False):
    issues = []
    value = normalize_choice(text) if _is_choice(kind) else normalize_text(text)
    if not value:
        return ["EMPTY_OCR"]
    if confidence is not None and confidence < .65:
        issues.append("LOW_OCR_CONFIDENCE")
    if _is_choice(kind) and not (re.fullmatch(r"[A-H]{1,8}", value) or value == "✓"):
        issues.append("INVALID_CHOICE")
    # Long copied prose and short blank labels are print evidence. Mathematical
    # tokens such as x and y remain valid even when they also occur in the stem.
    stem = normalize_text(question)
    prose = re.findall(r"[\u4e00-\u9fff]{3,}|[A-Za-z ]{18,}", str(text or ""))
    if any(normalize_text(fragment) in stem for fragment in prose):
        issues.append("PRINTED_STEM_CONTAMINATION")
    if value.endswith(("=", "+", "-", "/", "^", "\\")):
        issues.append("TRUNCATED_EXPRESSION")
    if re.fullmatch(r"[（(]\d+[）)]", value):
        issues.append("SUBQUESTION_MARKER_ONLY")
    if formula or _is_formula(question, kind, text):
        issues.extend(formula_structure_issues(text))
    elif not _balanced(str(text or "")):
        issues.append("UNBALANCED_EXPRESSION")
    return list(dict.fromkeys(issues))


def remove_printed_horizontal_rules(crop):
    """Remove long, thin printed rules while retaining handwriting components."""
    import cv2
    import numpy as np

    if crop is None or getattr(crop, "size", 0) == 0:
        return crop, {"applied": False, "removed_pixels": 0}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    width = ink.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(width * .28)), 1))
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    filtered = np.zeros_like(lines)
    count = 0
    for contour in cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= max(15, int(width * .25)) and h <= max(4, int(ink.shape[0] * .08)):
            cv2.drawContours(filtered, [contour], -1, 255, -1)
            count += int(cv2.countNonZero(filtered[y:y + h, x:x + w]))
    filtered = cv2.dilate(filtered, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    handwriting = cv2.bitwise_and(ink, cv2.bitwise_not(filtered))
    result = 255 - handwriting
    if crop.ndim == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result, {"applied": bool(count), "removed_pixels": int(count)}


class AnswerRecognizer:
    """Never receive expected answers. Retries change evidence, not the answer."""

    def __init__(self, ocr=None, formula_client=None):
        self.ocr = ocr or OCRService()
        self.formula_client = formula_client
        self.batch_metrics = {
            "batch_calls": 0, "batched_slots": 0, "single_slot_fallbacks": 0,
            "formula_singletons": 0, "formula_batches": 0, "batch_failures": 0,
        }

    def _vision(self, route, prompt, paths):
        method = getattr(self.formula_client, "analyze_" + route, None)
        return method(prompt, paths) if method else self.formula_client.analyze(prompt, paths)

    def recognize(self, page, box, question="", kind="", mask=None,
                  engine="paddle", language="chi_sim+eng",
                  visual_result=_VISUAL_UNSET, visual_source=None):
        import cv2
        from .result_contract import valid_box

        # Retained in the public signature for compatibility. Production no
        # longer consumes a handwriting/ink mask.
        del mask

        if not valid_box(box, page, "yxyx"):
            return {"text": "", "status": "UNRESOLVED", "fragments": [],
                    "attempts": [], "issues": ["INVALID_EXPECTED_BOX"], "warnings": []}
        if engine == "none" and self.formula_client is None:
            return {"text": "", "status": "UNRESOLVED", "fragments": [],
                    "attempts": [], "issues": ["OCR_DISABLED"], "warnings": []}

        xy = yxyx_to_xyxy(box)
        choice = _is_choice(kind)
        attempts, candidates = [], []

        def record(blocks, source, absolute=True, text_override=None,
                   visual_result=None, formula=False):
            ordered = row_order(blocks)
            raw = text_override if text_override is not None else "\n".join(b.text for b in ordered)
            text = normalize_choice(raw) if choice else str(raw or "").strip()
            confidence = min((b.confidence for b in ordered if b.confidence is not None), default=None)
            context = question + "\n" + "\n".join(
                b.text for b in page.ocr if len(re.findall(r"[\u4e00-\u9fff]", b.text)) >= 8)
            found = content_issues(text, context, kind, confidence, formula=formula)
            warnings = [issue for issue in found if issue in WARNING_ISSUES]
            issues = [issue for issue in found if issue not in WARNING_ISSUES]
            if visual_result is not None:
                if visual_result.get("legible", True) is False:
                    issues.append("VISUAL_ILLEGIBLE")
                content_kind = visual_result.get("content_kind", "uncertain")
                if content_kind not in {"handwriting", "uncertain"}:
                    warnings.append("VISUAL_CONTENT_KIND_" + str(content_kind).upper())
                # Formula syntax checks are diagnostics for a visual
                # transcription. They trigger review/repair telemetry but do
                # not erase an otherwise legible model result.
                if formula:
                    soft = {"UNBALANCED_EXPRESSION", "INCOMPLETE_SCRIPT",
                            "MALFORMED_FRACTION", "MISSING_OPERAND",
                            "MALFORMED_FORMULA", "TRUNCATED_EXPRESSION"}
                    warnings.extend(issue for issue in issues if issue in soft)
                    issues = [issue for issue in issues if issue not in soft]
            fragments = [{"page_index": page.index, "page_file": page.path,
                          "bbox": list(b.bbox), "text": b.text} for b in ordered] if absolute else []
            if text and not fragments:
                fragments = [{"page_index": page.index, "page_file": page.path,
                              "bbox": xy, "text": text}]
            candidate = {"text": text, "confidence": confidence,
                         "issues": list(dict.fromkeys(issues)), "warnings": list(dict.fromkeys(warnings)),
                         "source": source, "fragments": fragments,
                         "visual": visual_result is not None}
            candidates.append(candidate)
            attempts.append({"source": source, "issues": candidate["issues"],
                             "warnings": candidate["warnings"], "confidence": confidence})
            return candidate

        def attempt(source, function, absolute=True, formula=False):
            try:
                return record(function(), source, absolute, formula=formula)
            except Exception as exc:
                attempts.append({"source": source, "issues": ["OCR_BACKEND_ERROR"],
                                 "warnings": [], "error_type": type(exc).__name__})
                return None

        image = cv2.imread(page.path)
        crop = None
        if image is not None:
            x1, y1, x2, y2 = [int(v) for v in xy]
            crop = image[max(0, y1 - 4):min(image.shape[0], y2 + 4),
                         max(0, x1 - 4):min(image.shape[1], x2 + 4)]

        usable = lambda c: c and not c["issues"]
        if engine != "none":
            # Slot coordinates are already grounded to current-page OCR. Reuse
            # those blocks before launching another detector over the crop.
            def existing_blocks():
                selected = []
                for block in page.ocr:
                    if len(block.bbox or []) != 4:
                        continue
                    bx1, by1, bx2, by2 = block.bbox
                    overlap = max(0, min(xy[2], bx2) - max(xy[0], bx1)) * max(
                        0, min(xy[3], by2) - max(xy[1], by1))
                    block_area = max(1, (bx2 - bx1) * (by2 - by1))
                    if overlap / block_area >= .5:
                        selected.append(block)
                return selected

            page_blocks = existing_blocks()
            if page_blocks:
                attempt("current_page_ocr", lambda: page_blocks)
            if not any(usable(c) for c in candidates):
                attempt("original_crop", lambda: self.ocr.recognize_crop(
                    Path(page.path), xy, padding=4, engine=engine, language=language))
        if (engine != "none" and not any(usable(c) for c in candidates)
                and hasattr(self.ocr, "recognize_line")
                and (xy[3] - xy[1]) <= 180 and (xy[2] - xy[0]) <= 500):
            attempt("recognition_only", lambda: self.ocr.recognize_line(
                Path(page.path), xy, engine=engine, language=language))

        formula = _is_formula(question, kind, " ".join(c["text"] for c in candidates)) and not choice
        visual_candidates = []

        def visual_read(source, route, prompt):
            if crop is None or not crop.size:
                return None
            try:
                with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                    factor = max(1., min(4., 128. / max(1, min(crop.shape[:2]))))
                    enlarged = cv2.resize(crop, None, fx=factor, fy=factor,
                                           interpolation=cv2.INTER_CUBIC)
                    canvas = cv2.copyMakeBorder(enlarged, 24, 24, 24, 24,
                                                cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    cv2.imwrite(tmp.name, canvas)
                    result = self._vision(route, prompt, [Path(tmp.name)])
                value = result.get("transcription", result.get("_paddleocr_vl_ocr_text", ""))
                candidate = record([], source, False, str(value or ""), result, formula=formula)
                visual_candidates.append(candidate)
                return candidate
            except Exception as exc:
                attempts.append({"source": source, "issues": ["OCR_BACKEND_ERROR"],
                                 "warnings": [], "error_type": type(exc).__name__})
                return None

        # VLM transcription is the primary content result. OCR supplies the
        # physical box and an auxiliary reading; one visual pass avoids the
        # former two-pass consensus cost and false conflict vetoes.
        need_visual = bool(self.formula_client) or visual_result is not _VISUAL_UNSET
        if need_visual:
            if choice:
                prompt = ("Read the visible answer response as a multiple-choice mark. Return only the observed "
                          "A-H letter(s), or the literal check mark if no letter is written. Exclude printed "
                          "brackets and option labels. Never map a check mark to an option letter.")
                route, prefix = "choice", "choice_vision"
            elif formula:
                prompt = ("Transcribe only the visible answer formula in LaTeX. Preserve every minus sign, "
                          "superscript, subscript, numerator, denominator and parenthesis. Do not solve, complete, "
                          "or algebraically normalize the expression.")
                route, prefix = "formula", "formula_vision"
            else:
                prompt = ("Transcribe only the visible answer in this crop. Ignore printed question text "
                          "and blank rules. Preserve line order and do not solve or correct the response.")
                route, prefix = "handwriting", "handwriting_vision"
            if visual_result is _VISUAL_UNSET:
                visual_read(prefix, route, prompt)
            elif isinstance(visual_result, dict):
                value = visual_result.get(
                    "transcription", visual_result.get("_paddleocr_vl_ocr_text", ""))
                candidate = record(
                    [], visual_source or (prefix + "_batch"), False,
                    str(value or ""), visual_result, formula=formula,
                )
                visual_candidates.append(candidate)
            else:
                attempts.append({"source": visual_source or (prefix + "_batch"),
                                 "issues": ["INVALID_TRANSCRIPTION_RESPONSE"],
                                 "warnings": []})

        if not any(usable(c) for c in candidates) and crop is not None and crop.size:
            try:
                with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                    enlarged = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(tmp.name, enlarged)
                    record(self.ocr.recognize_page(Path(tmp.name), engine, language),
                           "upscaled_crop", False, formula=formula)
            except Exception as exc:
                attempts.append({"source": "upscaled_crop", "issues": ["OCR_BACKEND_ERROR"],
                                 "warnings": [], "error_type": type(exc).__name__})

        valid = [c for c in candidates if usable(c)]
        strong_local = [c for c in valid if not c["visual"] and not c["warnings"]]
        valid_visual = [c for c in visual_candidates if usable(c)]
        issues, warnings = [], []
        best = None
        agreement_type = "NONE"

        invalid_visual = [c for c in visual_candidates if c["issues"]]
        visual_illegible = any("VISUAL_ILLEGIBLE" in c["issues"] for c in invalid_visual)
        visual_values = [_comparison_value(c["text"], formula) for c in valid_visual]
        visual_stable = len(visual_values) == 1

        if valid_visual:
            best = valid_visual[0]
            different_local = [candidate for candidate in strong_local
                               if _comparison_value(candidate["text"], formula)
                               != _comparison_value(best["text"], formula)]
            if different_local:
                warnings.append("OCR_VLM_DISAGREEMENT")
                agreement_type = "VLM_PRIMARY_OCR_WARNING"
            else:
                agreement_type = "VLM_PRIMARY"
            warnings.extend(issue for candidate in valid for issue in candidate["warnings"])
        elif visual_illegible and not strong_local:
            issues.append("VISUAL_ILLEGIBLE")
            best = valid[0] if valid else (invalid_visual[0] if invalid_visual else None)
            agreement_type = "VISUAL_QUALITY_INSUFFICIENT"
        elif strong_local:
            values = {_comparison_value(c["text"], formula) for c in strong_local}
            best = strong_local[0]
            if len(values) > 1:
                warnings.append("OCR_INTERNAL_DISAGREEMENT")
                agreement_type = "OCR_PRIMARY_WITH_WARNING"
            else:
                agreement_type = "LOCAL_OCR_AGREEMENT"
        elif valid:
            best = valid_visual[-1] if valid_visual else valid[0]
            warnings.extend(best["warnings"])
            agreement_type = "LOW_CONFIDENCE_ONLY"
        else:
            best = min(candidates, key=lambda c: (len(c["issues"]), -float(c["confidence"] or 0))) \
                if candidates else None
            issues.extend(best["issues"] if best else ["EMPTY_OCR"])

        if best:
            warnings.extend(best["warnings"])
        issues = list(dict.fromkeys(issues))
        warnings = list(dict.fromkeys(warnings))
        response = {
            "text": best["text"] if best else "",
            "status": "RECOGNIZED" if best and not issues else "UNRESOLVED",
            "fragments": best["fragments"] if best else [],
            "issues": issues,
            "warnings": warnings,
            "confidence": best["confidence"] if best else None,
            "attempts": attempts,
            "agreement_type": agreement_type,
            # Keep the legacy route label for consumers while exposing the
            # dedicated prompt path separately.
            "formula_route": "vision" if formula and self.formula_client else
                             "local_ocr_only" if formula else None,
            "vision_route": ("formula" if formula else "choice" if choice else "handwriting")
                if need_visual else None,
            "coordinate_source": "ocr_box",
        }
        if choice:
            response["choice_evidence"] = {
                "normalized": normalize_choice(response["text"]),
                "visual_passes": len(visual_candidates),
                "independent_visual_agreement": False,
                "agreement_type": agreement_type,
            }
        if formula:
            response["formula_checks"] = formula_structure(response["text"])
        return response

    def recognize_many(self, entries, engine="paddle", language="chi_sim+eng",
                       batch_size=10):
        """Recognize page-local slots with one VLM request per route batch.

        ``entries`` contain ``key``, ``slot_id``, ``page``, ``box``,
        ``question`` and ``kind``. Formula entries retain a dedicated formula
        prompt batch. A malformed batch falls back only to its own slots, and
        an omitted ID falls back only to that ID.
        """
        import cv2

        entries = list(entries)
        if not entries:
            return {}
        analyze_batch = getattr(self.formula_client, "analyze_batch", None)
        if not callable(analyze_batch):
            return {
                entry["key"]: self.recognize(
                    entry["page"], entry["box"], entry.get("question", ""),
                    entry.get("kind", ""), None, engine, language)
                for entry in entries
            }

        results = {}
        groups = {}
        for entry in entries:
            route = ("choice" if _is_choice(entry.get("kind")) else
                     "formula" if _is_formula(entry.get("question"), entry.get("kind"))
                     else "handwriting")
            group_key = (entry["page"].index, route)
            groups.setdefault(group_key, []).append(entry)

        for (_, route), grouped in groups.items():
            for offset in range(0, len(grouped), max(1, min(12, int(batch_size)))):
                batch = grouped[offset:offset + max(1, min(12, int(batch_size)))]
                batch_results = None
                try:
                    with tempfile.TemporaryDirectory(prefix="answer_batch_") as tmp:
                        paths = []
                        request_entries = []
                        for image_index, entry in enumerate(batch, 1):
                            image = cv2.imread(entry["page"].path)
                            if image is None:
                                raise ValueError("ANSWER_PAGE_UNAVAILABLE")
                            x1, y1, x2, y2 = [int(v) for v in yxyx_to_xyxy(entry["box"])]
                            crop = image[max(0, y1 - 4):min(image.shape[0], y2 + 4),
                                         max(0, x1 - 4):min(image.shape[1], x2 + 4)]
                            if not crop.size:
                                raise ValueError("INVALID_EXPECTED_BOX")
                            factor = max(1., min(4., 128. / max(1, min(crop.shape[:2]))))
                            enlarged = cv2.resize(crop, None, fx=factor, fy=factor,
                                                  interpolation=cv2.INTER_CUBIC)
                            canvas = cv2.copyMakeBorder(
                                enlarged, 24, 24, 24, 24, cv2.BORDER_CONSTANT,
                                value=(255, 255, 255))
                            path = Path(tmp) / f"{image_index:02d}.png"
                            if not cv2.imwrite(str(path), canvas):
                                raise OSError("ANSWER_CROP_WRITE_FAILED")
                            paths.append(path)
                            request_entries.append({"slot_id": str(entry["slot_id"])})
                        self.batch_metrics["batch_calls"] += 1
                        self.batch_metrics["batched_slots"] += len(batch)
                        if route == "formula":
                            self.batch_metrics["formula_batches"] += 1
                        batch_results = analyze_batch(request_entries, paths, route=route)
                except Exception:
                    self.batch_metrics["batch_failures"] += 1

                for entry in batch:
                    visual = (batch_results or {}).get(str(entry["slot_id"]))
                    if visual is None:
                        self.batch_metrics["single_slot_fallbacks"] += 1
                        results[entry["key"]] = self.recognize(
                            entry["page"], entry["box"], entry.get("question", ""),
                            entry.get("kind", ""), None, engine, language)
                    else:
                        results[entry["key"]] = self.recognize(
                            entry["page"], entry["box"], entry.get("question", ""),
                            entry.get("kind", ""), None, engine, language,
                            visual_result=visual, visual_source=route + "_vision_batch")
        return results


__all__ = ["AnswerRecognizer", "content_issues", "formula_structure",
           "formula_structure_issues", "normalize_choice", "normalize_text",
           "remove_printed_horizontal_rules"]
