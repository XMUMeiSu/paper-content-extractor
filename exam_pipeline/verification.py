"""Blank gate, pure-ink snapping and iterative momentum verification."""
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from .contracts import ExamPackage, Page, Slot
from .ocr import OCRService
from .roi import yxyx_to_xyxy
from .answer_parser import parse_student_answer


def bbox_iou(a, b):
    y1, x1, y2, x2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, y2-y1) * max(0, x2-x1)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1]); ab = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / max(1, aa+ab-inter)


def levenshtein_distance(left: str, right: str) -> int:
    a, b = str(left or ""), str(right or "")
    if len(a) < len(b): a, b = b, a
    previous = list(range(len(b)+1))
    for row, ca in enumerate(a, 1):
        current = [row]
        for column, cb in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[column]+1,
                               previous[column-1] + (ca != cb)))
        previous = current
    return previous[-1]


class BlankInkGate:
    MIN_PIXELS = 10
    @classmethod
    def evaluate(cls, mask, minimum_pixels=None):
        import cv2
        count = int(cv2.countNonZero(mask)) if mask is not None else 0
        threshold = int(minimum_pixels or cls.MIN_PIXELS)
        return {"has_ink": count >= threshold, "pixel_count": count,
                "threshold": threshold}


class UniversalInkSnapper:
    HORIZONTAL_KERNEL = (25, 1)
    BREATHING_PADDING = 3

    @classmethod
    def extract(cls, image: Any, expected_bbox: Sequence[int], reference_image=None,
                policy=None, reference_kind: str = "teacher",
                reference_confidence: Optional[float] = None,
                prefer_colored_ink: bool = False):
        import cv2
        from .ink_separation import NoBlankInkSeparator
        height, width = image.shape[:2]
        y1, x1, y2, x2 = [int(round(v)) for v in expected_bbox[:4]]
        y1, y2 = sorted((max(0, y1), min(height, y2)))
        x1, x2 = sorted((max(0, x1), min(width, x2)))
        crop = image[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            gate = {"has_ink": False, "pixel_count": 0, "threshold": 10}
            return {"bbox": None, "mask": None, "gate": gate, "component_count": 0}
        reference_crop = None
        if reference_image is not None and reference_image.shape[:2] == image.shape[:2]:
            reference_crop = reference_image[y1:y2, x1:x2]
        separation = NoBlankInkSeparator.separate(
            crop, reference_crop, policy, reference_kind, reference_confidence,
            prefer_colored_ink
        )
        pure = separation["handwriting_mask"]
        minimum_pixels = policy.parameter(
            "minimum_ink_pixels", BlankInkGate.MIN_PIXELS
        ) if policy else BlankInkGate.MIN_PIXELS
        gate = BlankInkGate.evaluate(pure, minimum_pixels)
        if not gate["has_ink"]:
            return {"bbox": None, "mask": pure, "gate": gate, "component_count": 0,
                    "separation": {key: separation[key] for key in (
                        "mode", "confidence", "reference_used", "reference_kind", "metrics"
                    )}}
        _, _, stats, _ = cv2.connectedComponentsWithStats(pure, connectivity=8)
        crop_h, crop_w = pure.shape[:2]
        components = []
        for bx, by, bw, bh, area in stats[1:]:
            bx, by, bw, bh, area = map(int, (bx, by, bw, bh, area))
            if bw < 4 or bh < 5 or area < 10: continue
            if bw > .85*crop_w and bh > .85*crop_h: continue
            if bh > 35 and bw <= 3: continue
            if abs(by + .5*bh - .5*crop_h) > max(28, .45*crop_h): continue
            components.append((bx, by, bw, bh, area))
        if not components:
            return {"bbox": None, "mask": pure, "gate": gate, "component_count": 0,
                    "separation": {key: separation[key] for key in (
                        "mode", "confidence", "reference_used", "reference_kind", "metrics"
                    )}}
        lx1 = min(c[0] for c in components); ly1 = min(c[1] for c in components)
        lx2 = max(c[0]+c[2] for c in components); ly2 = max(c[1]+c[3] for c in components)
        padding = int(policy.parameter(
            "breathing_padding_px", cls.BREATHING_PADDING
        )) if policy else cls.BREATHING_PADDING
        bbox = [max(0, y1+ly1-padding), max(0, x1+lx1-padding),
                min(height, y1+ly2+padding), min(width, x1+lx2+padding)]
        return {"bbox": bbox, "mask": pure, "gate": gate,
                "component_count": len(components), "ink_area": sum(c[4] for c in components),
                "separation": {key: separation[key] for key in (
                    "mode", "confidence", "reference_used", "reference_kind", "metrics"
                )}, "_handwriting_mask": pure}


class IterativeVerificationController:
    MAX_ITERATIONS = 3
    DETECTED_WEIGHT = .65
    CURRENT_WEIGHT = .35
    IOU_THRESHOLD = .90

    def __init__(self, policy=None):
        self.policy = policy

    @classmethod
    def _damp(cls, current, detected, corridor):
        result = [int(round(cls.CURRENT_WEIGHT*current[i] + cls.DETECTED_WEIGHT*detected[i])) for i in range(4)]
        result[0] = max(corridor[0], result[0]); result[2] = min(corridor[1], result[2])
        return result

    def verify(self, image, slot: Slot, reference_image=None,
               recognize: Optional[Callable[[Sequence[int]], str]] = None, observed_text: str = "",
               reference_kind: str = "teacher",
               reference_confidence: Optional[float] = None):
        initial = list(slot.expected_bbox)
        probe = list(initial)
        evidence = str((slot.audit or {}).get("evidence", ""))
        if "residual_local_registration" in evidence:
            # Residual discovery already returns a tight student-ink box. Give
            # the independent verifier context around it; otherwise a glyph
            # can fill >85% of the crop and be rejected as a border artifact.
            height, width = image.shape[:2]
            padding = max(10, min(24, int(round(
                .18 * max(initial[2]-initial[0], initial[3]-initial[1])
            ))))
            probe = [
                max(0, initial[0]-padding), max(0, initial[1]-padding),
                min(height, initial[2]+padding), min(width, initial[3]+padding),
            ]
        detection = UniversalInkSnapper.extract(
            image, probe, reference_image, self.policy,
            reference_kind, reference_confidence
        )
        gate = detection["gate"]
        if not gate["has_ink"]:
            return {"status": "BLANK_UNANSWERED", "iterations_used": 0, "initial_bbox": initial,
                    "final_bbox": None, "expected_text": slot.expected_text, "recognized_text": "",
                    "has_ink": False, "shrink_rate": "0.0%", "history": [], "gate": gate,
                    "component_count": detection.get("component_count", 0),
                    "geometry_status": "ALIGNED", "content_status": "BLANK",
                    "semantic_status": "NOT_EVALUATED", "review_status": "NEED_REVIEW",
                    "separation": detection.get("separation", {}),
                    "_handwriting_mask": detection.get("mask")}
        if not detection.get("bbox"):
            return {"status": "ANOMALY_ESCALATED", "iterations_used": 0, "initial_bbox": initial,
                    "final_bbox": None, "expected_text": slot.expected_text,
                    "recognized_text": str(observed_text or ""), "has_ink": True,
                    "shrink_rate": "0.0%", "history": [], "gate": gate, "component_count": 0,
                    "reason": "ink present but no valid handwriting component",
                    "geometry_status": "FAILED", "content_status": "OCR_UNCERTAIN",
                    "semantic_status": "NOT_EVALUATED", "review_status": "NEED_REVIEW",
                    "separation": detection.get("separation", {}),
                    "_handwriting_mask": detection.get("mask")}
        detected = list(detection["bbox"])
        recognized = str(observed_text or "")
        if recognize:
            local = str(recognize(detected) or "")
            if local: recognized = local
        expected = str(slot.expected_text or "")
        distance = levenshtein_distance(recognized, expected) if expected else 0
        semantic_ok = bool(expected) and distance <= 1
        line_height = max(1, initial[2]-initial[0])
        corridor = (max(0, round(initial[0]-.5*line_height)),
                    min(image.shape[0], round(initial[2]+.5*line_height)))
        current, history, status = initial, [], "ANOMALY_ESCALATED"
        max_iterations = int(self.policy.parameter(
            "max_iterations", self.MAX_ITERATIONS
        )) if self.policy else self.MAX_ITERATIONS
        for iteration in range(1, max_iterations+1):
            updated = self._damp(current, detected, corridor)
            iou = bbox_iou(current, updated)
            history.append({"iteration": iteration, "input_bbox": list(current), "detected_bbox": detected,
                            "output_bbox": list(updated), "iou": round(iou, 4),
                            "max_displacement": max(abs(updated[i]-current[i]) for i in range(4)),
                            "recognized_text": recognized, "semantic_distance": distance})
            current = updated
            # Localization is a geometric decision. OCR/answer agreement is
            # reported independently and must never turn a good bbox into a
            # geometry anomaly.
            if iou >= self.IOU_THRESHOLD:
                status = "CONVERGED_SUCCESS"; break
        initial_area = max(1, (initial[2]-initial[0])*(initial[3]-initial[1]))
        final_area = max(0, (current[2]-current[0])*(current[3]-current[1]))
        shrink = max(0., min(100., (1-final_area/initial_area)*100))
        geometry_status = "ALIGNED" if status == "CONVERGED_SUCCESS" else "UNCERTAIN"
        if not recognized:
            content_status = "OCR_UNCERTAIN"
        else:
            content_status = "RECOGNIZED"
        semantic_status = ("MATCH" if semantic_ok else
                           "MISMATCH" if expected and recognized else "NOT_EVALUATED")
        review_status = ("AUTO_PASS" if geometry_status == "ALIGNED"
                         and semantic_status == "MATCH" else "NEED_REVIEW")
        reason = ""
        if geometry_status != "ALIGNED":
            reason = "geometry did not converge"
        elif not expected:
            reason = "missing expected text"
        elif not recognized:
            reason = "OCR returned no text"
        elif not semantic_ok:
            reason = "recognized text differs from expected text"
        return {"status": status, "iterations_used": len(history), "initial_bbox": initial,
                "final_bbox": current, "detected_bbox": detected, "expected_text": slot.expected_text,
                "recognized_text": recognized, "has_ink": True, "shrink_rate": f"{shrink:.1f}%",
                "history": history, "gate": gate, "component_count": detection.get("component_count", 0),
                "semantic_distance": distance,
                "geometry_status": geometry_status, "content_status": content_status,
                "semantic_status": semantic_status, "review_status": review_status,
                "reason": reason,
                "separation": detection.get("separation", {}),
                "_handwriting_mask": detection.get("_handwriting_mask")}


class SlotVerificationService:
    def __init__(self, controller=None, ocr_service=None, policy=None):
        self.policy = policy
        self.controller = controller or IterativeVerificationController(policy)
        self.ocr = ocr_service or OCRService()

    def verify_package(self, package: ExamPackage, pages: Sequence[Page], reference_pages=(),
                       engine: str = "paddle", language: str = "chi_sim+eng",
                       reference_context: Optional[Dict[str, Any]] = None,
                       debug_dir: Optional[Path] = None,
                       independent_page: bool = False):
        import cv2
        pageVerts = {page.index: page for page in pages}
        ref_map = {page.index: page for page in reference_pages}
        images = {index: cv2.imread(page.path) for index, page in pageVerts.items()}
        references = {index: cv2.imread(page.path) for index, page in ref_map.items()}
        registration = {index: meta for index, meta in enumerate(package.registration or [], 1)}
        records: Dict[str, List[Dict[str, Any]]] = {}
        counts = {"CONVERGED_SUCCESS": 0, "BLANK_UNANSWERED": 0, "ANOMALY_ESCALATED": 0}
        geometry_counts = {"ALIGNED": 0, "UNCERTAIN": 0, "FAILED": 0, "MISSING": 0}
        content_counts = {"RECOGNIZED": 0, "OCR_UNCERTAIN": 0, "BLANK": 0,
                          "NOT_EVALUATED": 0}
        iterations, shrink_rates = [], []
        for section in package.sections:
            for question in section.questions:
                for item in question.items:
                    for slot in item.slots:
                        page = pageVerts.get(slot.page_index); image = images.get(slot.page_index)
                        if page is None or image is None:
                            continue
                        registration_meta = registration.get(slot.page_index)
                        student_self_located = (slot.audit or {}).get("topology_source") == "student_self"
                        missing_slot = (slot.audit or {}).get("topology_source") == "missing_placeholder"
                        if missing_slot:
                            result = {
                                "status": "ANOMALY_ESCALATED", "iterations_used": 0,
                                "initial_bbox": list(slot.expected_bbox), "final_bbox": None,
                                "expected_text": slot.expected_text, "recognized_text": "",
                                "has_ink": False, "shrink_rate": "0.0%", "history": [],
                                "gate": {"has_ink": False, "pixel_count": 0, "threshold": 10},
                                "component_count": 0, "geometry_status": "MISSING",
                                "content_status": "NOT_EVALUATED",
                                "semantic_status": "NOT_EVALUATED",
                                "review_status": "NEED_REVIEW",
                                "reason": "expected slot was not found on the student page",
                            }
                        elif (not independent_page and registration_meta
                                and registration_meta.get("status") != "REGISTERED"
                                and not student_self_located):
                            result = {
                                "status": "ANOMALY_ESCALATED", "iterations_used": 0,
                                "initial_bbox": list(slot.expected_bbox), "final_bbox": None,
                                "expected_text": slot.expected_text, "recognized_text": "",
                                "has_ink": False, "shrink_rate": "0.0%", "history": [],
                                "gate": {"has_ink": False, "pixel_count": 0, "threshold": 10},
                                "component_count": 0,
                                "geometry_status": "FAILED", "content_status": "NOT_EVALUATED",
                                "semantic_status": "NOT_EVALUATED", "review_status": "NEED_REVIEW",
                                "reason": f"registration unavailable: {registration_meta.get('status')}",
                            }
                        else:
                            result = None
                        def recognize(box, page_path=Path(page.path)):
                            if engine == "none": return ""
                            blocks = self.ocr.recognize_crop(page_path, yxyx_to_xyxy(box), padding=3,
                                                             engine=engine, language=language)
                            return "".join(block.text for block in sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))).strip()
                        observed = str(item.student_answer or "") if len(item.slots) == 1 else ""
                        if result is None:
                            reference = references.get(slot.page_index)
                            if (independent_page or (registration_meta
                                    and registration_meta.get("status") != "REGISTERED"
                                    and student_self_located)):
                                reference = None
                            root_context = reference_context or {}
                            context = (root_context.get("pages", {}).get(slot.page_index, {})
                                       if root_context.get("pages") else root_context)
                            result = self.controller.verify(
                                image, slot, reference, recognize, observed,
                                str(context.get("kind", "teacher")),
                                context.get("confidence"),
                            )
                        mask = result.pop("_handwriting_mask", None)
                        if debug_dir and mask is not None:
                            debug_path = (Path(debug_dir) / f"page_{slot.page_index:02d}"
                                          / f"{item.item_id}__slot_{slot.slot_idx}.png")
                            debug_path.parent.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(debug_path), mask)
                            result.setdefault("separation", {})["handwriting_mask_path"] = str(debug_path)
                        slot.handwriting_bbox = result["final_bbox"]; slot.recognized_text = result["recognized_text"]
                        slot.has_ink = result["has_ink"]; slot.status = result["status"]
                        slot.geometry_status = result.get("geometry_status", "UNCERTAIN")
                        slot.content_status = result.get("content_status", "NOT_EVALUATED")
                        slot.review_status = result.get("review_status", "NEED_REVIEW")
                        slot.iterations_used = result["iterations_used"]; slot.shrink_rate = result["shrink_rate"]
                        slot.history = result["history"]
                        slot.audit["ink_separation"] = result.get("separation", {})

                        # Parse clean student answer from noisy OCR text
                        if slot.recognized_text and slot.has_ink:
                            clean_answer, parse_audit = parse_student_answer(
                                recognized_text=slot.recognized_text,
                                expected_text=slot.expected_text,
                                slot_type=slot.slot_type,
                                item_type=item.item_type,
                                question_text=question.question_title or ""
                            )
                            slot.student_answer = clean_answer
                            slot.audit["answer_parsing"] = parse_audit
                        else:
                            slot.student_answer = None

                        if self.policy:
                            slot.audit["knowledge_rules"] = sorted(set(
                                slot.audit.get("knowledge_rules", [])
                                + self.policy.rule_ids("verification")
                            ))
                        record = {"question_id": question.question_id, "item_id": item.item_id,
                                  "slot_index": slot.slot_idx, **result}
                        records.setdefault(f"{package.exam_id}:page_{slot.page_index:02d}", []).append(record)
                        counts[result["status"]] += 1
                        geometry_counts[slot.geometry_status] = geometry_counts.get(slot.geometry_status, 0) + 1
                        content_counts[slot.content_status] = content_counts.get(slot.content_status, 0) + 1
                        if result["status"] == "CONVERGED_SUCCESS":
                            iterations.append(result["iterations_used"])
                            shrink_rates.append(float(result["shrink_rate"].rstrip("%")))
        total = sum(counts.values())
        summary = {"total_audit_slots": total, "converged_success": counts["CONVERGED_SUCCESS"],
                   "blank_unanswered": counts["BLANK_UNANSWERED"], "anomaly_escalated": counts["ANOMALY_ESCALATED"],
                   "geometry": geometry_counts, "content": content_counts,
                   "avg_iterations_to_converge": round(sum(iterations)/len(iterations), 2) if iterations else 0.,
                   "avg_area_shrink_rate": f"{sum(shrink_rates)/len(shrink_rates):.1f}%" if shrink_rates else "0.0%",
                   "max_iter_limit": int(self.policy.parameter(
                       "max_iterations", 3
                   )) if self.policy else 3,
                   "knowledge_base": self.policy.audit() if self.policy else None}
        return {"summary": summary, "pages": records, "bbox_format": "yxyx",
                "canonical_canvas": {"width": 1654, "height": 2338}}


__all__ = ["BlankInkGate", "UniversalInkSnapper", "IterativeVerificationController",
           "SlotVerificationService", "bbox_iou", "levenshtein_distance"]
