"""Handwriting isolation when no pristine blank exam template exists."""
from __future__ import annotations

from typing import Any, Dict, Optional


class NoBlankInkSeparator:
    """Build a conservative handwriting mask from a filled page.

    The default path classifies ink from the current page only.  Registered
    cohort/teacher references remain optional compatibility modes.
    """

    @staticmethod
    def _binary_ink(image):
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )[1]

    @staticmethod
    def _colored_ink(image):
        import cv2
        import numpy as np
        if image.ndim != 3:
            return np.zeros(image.shape[:2], dtype="uint8")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Saturated pen colors (red/blue/green) are annotations or handwriting,
        # while ordinary black printed text has low saturation.
        return cv2.inRange(hsv, (0, 38, 25), (179, 255, 255))

    @staticmethod
    def _line_grid_mask(binary):
        import cv2
        import numpy as np
        height, width = binary.shape[:2]
        horizontal = cv2.morphologyEx(
            binary, cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_RECT, (max(25, int(width * 0.08)), 1)
            ),
        )
        # Do not erase vertical structures inside a tight answer crop: Chinese
        # characters and digits legitimately contain long vertical strokes.
        # Registered template consensus handles printed grid borders instead.
        vertical = np.zeros_like(binary)
        return cv2.bitwise_or(horizontal, vertical)

    @staticmethod
    def _reconstruct_seeded_components(source, seeds):
        import cv2
        import numpy as np
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            source, connectivity=8
        )
        result = np.zeros_like(source)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 6:
                continue
            component = labels == label
            seed_pixels = int(cv2.countNonZero(
                cv2.bitwise_and(seeds, seeds, mask=component.astype("uint8"))
            ))
            # Registration/JPEG noise commonly leaves one or two residual
            # pixels on print. Require a meaningful unique-ink fraction.
            if seed_pixels >= max(3, int(round(area * 0.08))):
                result[component] = 255
        return result

    @classmethod
    def _single_page_classify(cls, image, ink_without_lines, colored_ink):
        """Split print/handwriting using only visual regularity in one crop.

        Printed glyph runs normally have a repeated height and baseline.  The
        classifier suppresses only those high-support regular runs and obvious
        tiny edge labels.  Ambiguous black components stay in the handwriting
        mask so answers are not silently erased; the quality gate receives the
        resulting confidence and can route them to HITL.
        """
        import cv2
        import numpy as np

        black = cv2.subtract(ink_without_lines, colored_ink)
        joined = cv2.morphologyEx(
            black, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1
        )
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            joined, connectivity=8
        )
        height, width = black.shape[:2]
        components = []
        for label in range(1, count):
            x, y, w, h, area = map(int, stats[label])
            if area < 3:
                continue
            components.append({
                "label": label, "x": x, "y": y, "w": w, "h": h,
                "area": area, "cy": float(centroids[label][1]),
            })

        printed_labels = set()
        eligible = [c for c in components if 4 <= c["h"] <= max(8, int(.32 * height))]
        eligible.sort(key=lambda c: c["cy"])
        rows = []
        for component in eligible:
            best = None
            for row in rows:
                tolerance = max(3.0, .45 * row["median_height"])
                if abs(component["cy"] - row["baseline"]) <= tolerance:
                    best = row
                    break
            if best is None:
                rows.append({
                    "items": [component], "baseline": component["cy"],
                    "median_height": float(component["h"]),
                })
            else:
                best["items"].append(component)
                ys = sorted(item["cy"] for item in best["items"])
                hs = sorted(item["h"] for item in best["items"])
                best["baseline"] = ys[len(ys) // 2]
                best["median_height"] = float(hs[len(hs) // 2])

        regular_rows = 0
        for row in rows:
            items = row["items"]
            if len(items) < 5:
                continue
            heights = np.asarray([item["h"] for item in items], dtype=np.float32)
            median = max(1.0, float(np.median(heights)))
            height_cv = float(np.std(heights) / median)
            span = max(item["x"] + item["w"] for item in items) - min(item["x"] for item in items)
            if height_cv <= .34 and span >= max(35, int(.18 * width)):
                regular_rows += 1
                printed_labels.update(item["label"] for item in items)

        # Small punctuation/labels touching a crop edge are much more likely to
        # belong to the printed prompt than to the answer body.
        edge_margin = max(2, int(round(min(height, width) * .025)))
        for component in components:
            edge = (component["x"] <= edge_margin
                    or component["x"] + component["w"] >= width - edge_margin)
            if edge and component["area"] <= 90 and component["h"] <= max(16, int(.18 * height)):
                printed_labels.add(component["label"])

        printed = np.zeros_like(black)
        for label in printed_labels:
            printed[labels == label] = 255
        # Keep only pixels present in the original (prevents closing artifacts).
        printed = cv2.bitwise_and(printed, black)
        handwriting = cv2.subtract(ink_without_lines, printed)
        handwriting = cv2.bitwise_or(handwriting, colored_ink)
        classified_pixels = int(cv2.countNonZero(printed))
        black_pixels = max(1, int(cv2.countNonZero(black)))
        evidence = classified_pixels / black_pixels
        confidence = 0.68 if regular_rows else (0.62 if evidence >= .08 else 0.56)
        return printed, handwriting, confidence, {
            "component_count": len(components),
            "regular_print_rows": regular_rows,
            "classified_print_ratio": round(evidence, 4),
        }

    @classmethod
    def separate(cls, student_image: Any, reference_image: Optional[Any] = None,
                 policy=None, reference_kind: str = "teacher",
                 reference_confidence: Optional[float] = None,
                 prefer_colored_ink: bool = False) -> Dict[str, Any]:
        import cv2
        import numpy as np
        student_ink = cls._binary_ink(student_image)
        line_grid = cls._line_grid_mask(student_ink)
        student_without_lines = cv2.subtract(student_ink, line_grid)
        student_color = cls._colored_ink(student_image)

        usable_reference = (
            reference_image is not None
            and reference_image.shape[:2] == student_image.shape[:2]
        )
        soft_residual_metrics = {}
        colored_pixels = int(cv2.countNonZero(student_color))
        if prefer_colored_ink and colored_pixels >= 10:
            printed = cv2.subtract(student_ink, student_color)
            handwriting = student_color
            mode, confidence = "teacher_colored_ink", 0.96
        elif usable_reference:
            reference_ink = cls._binary_ink(reference_image)
            reference_color = cls._colored_ink(reference_image)
            # Teacher red/colored answers must never become part of the print
            # template. A small dilation also removes their antialiased fringe.
            reference_clean = cv2.subtract(
                reference_ink,
                cv2.dilate(reference_color, np.ones((3, 3), np.uint8), iterations=1),
            )
            tolerance = int(policy.parameter(
                "print_alignment_tolerance_px", 1
            )) if policy else 1
            if reference_kind == "cohort":
                tolerance = max(tolerance, 3)
            kernel_size = max(1, 2 * tolerance + 1)
            nearby_reference = cv2.dilate(
                reference_clean,
                np.ones((kernel_size, kernel_size), np.uint8),
                iterations=1,
            )
            printed = cv2.bitwise_and(student_ink, nearby_reference)
            binary_unique = cv2.subtract(student_without_lines, printed)

            # Continuous darkness residual retains pen pressure/antialiasing.
            # Suppress energy on strong reference-print edges, where a tiny
            # registration shift otherwise creates convincing false ink.
            student_gray = (cv2.cvtColor(student_image, cv2.COLOR_BGR2GRAY)
                            if student_image.ndim == 3 else student_image)
            reference_gray = (cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
                              if reference_image.ndim == 3 else reference_image)
            darkness_delta = np.maximum(
                reference_gray.astype(np.float32) - student_gray.astype(np.float32),
                0.0,
            )
            sobel_x = cv2.Sobel(reference_gray, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(reference_gray, cv2.CV_32F, 0, 1, ksize=3)
            edge_strength = np.clip(
                cv2.magnitude(sobel_x, sobel_y) / 100.0, 0.0, 1.0
            )
            residual_energy = darkness_delta * np.exp(-2.5 * edge_strength)
            residual_threshold = 24.0 if reference_kind == "cohort" else 30.0
            soft_seed = np.where(residual_energy >= residual_threshold, 255, 0).astype(np.uint8)
            soft_seed = cv2.bitwise_and(soft_seed, student_without_lines)
            seeds = cv2.bitwise_or(binary_unique, soft_seed)
            seeds = cv2.bitwise_or(seeds, student_color)
            handwriting = cls._reconstruct_seeded_components(
                student_without_lines, seeds
            )
            handwriting = cv2.bitwise_or(handwriting, student_color)
            soft_residual_metrics = {
                "residual_method": "continuous_darkness_edge_suppressed",
                "residual_threshold": residual_threshold,
                "soft_seed_pixels": int(cv2.countNonZero(soft_seed)),
                "mean_residual_energy": round(float(residual_energy.mean()), 4),
            }
            if reference_kind == "cohort":
                mode = "cohort_consensus_residual"
                confidence = float(reference_confidence or 0.84)
            else:
                mode, confidence = "teacher_consensus_residual", 0.82
        else:
            printed, handwriting, confidence, single_page_metrics = cls._single_page_classify(
                student_image, student_without_lines, student_color
            )
            mode = "single_page_visual_classifier"

        return {
            "handwriting_mask": handwriting,
            "printed_mask": printed,
            "line_grid_mask": line_grid,
            "student_ink_mask": student_ink,
            "mode": mode,
            "confidence": confidence,
            "reference_used": bool(usable_reference),
            "reference_kind": reference_kind if usable_reference else "none",
            "metrics": {
                "student_ink_pixels": int(cv2.countNonZero(student_ink)),
                "printed_pixels": int(cv2.countNonZero(printed)),
                "line_grid_pixels": int(cv2.countNonZero(line_grid)),
                "handwriting_pixels": int(cv2.countNonZero(handwriting)),
                "colored_ink_pixels": colored_pixels,
                **soft_residual_metrics,
                **(single_page_metrics if not usable_reference
                   and not (prefer_colored_ink and colored_pixels >= 10) else {}),
            },
        }


__all__ = ["NoBlankInkSeparator"]
