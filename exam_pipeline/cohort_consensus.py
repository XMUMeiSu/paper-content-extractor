"""Batch-level pseudo-blank template construction for repeated exam forms."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, Sequence

from .io_utils import atomic_write_json


class CohortConsensusBuilder:
    """Estimate stable printed ink from multiple independently filled papers.

    Student pages are normalized and registered to the teacher coordinate
    system. Ink that survives a robust cross-document vote becomes a
    pseudo-blank template. The teacher is excluded from voting so that a black
    teacher answer cannot become template ink by itself.
    """

    MIN_SAMPLES = 2
    DEFAULT_MAX_SAMPLES = 12
    VOTE_RATIO = 0.70
    BRIGHTNESS_PERCENTILE = 75.0

    @staticmethod
    def _clean_binary(image):
        import cv2
        import numpy as np

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        if image.ndim == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            colored = cv2.inRange(hsv, (0, 38, 25), (179, 255, 255))
            ink = cv2.subtract(
                ink, cv2.dilate(colored, np.ones((3, 3), np.uint8), iterations=1)
            )
        return ink

    @staticmethod
    def _print_grayscale(image):
        """Return grayscale with saturated pen colours removed from voting."""
        import cv2
        import numpy as np

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        if image.ndim == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            colored = cv2.inRange(hsv, (0, 38, 25), (179, 255, 255))
            colored = cv2.dilate(colored, np.ones((3, 3), np.uint8), iterations=1)
            gray[colored > 0] = 255
        return gray

    @classmethod
    def percentile_template(cls, aligned_images: Sequence[Any], stable_mask,
                            percentile: float = BRIGHTNESS_PERCENTILE):
        """Preserve antialiased print tone while excluding unstable ink."""
        import numpy as np

        images = [image for image in aligned_images if image is not None]
        if not images or stable_mask is None:
            return None
        shape = stable_mask.shape[:2]
        grays = [cls._print_grayscale(image) for image in images
                 if image.shape[:2] == shape]
        if len(grays) < cls.MIN_SAMPLES:
            return None
        template = np.percentile(
            np.stack(grays, axis=0), float(percentile), axis=0
        ).astype("uint8")
        template[stable_mask == 0] = 255
        return template

    @classmethod
    def consensus_mask(cls, aligned_images: Sequence[Any], vote_ratio: float = VOTE_RATIO):
        """Build stable ink from already aligned pages (also used by tests)."""
        import cv2
        import numpy as np

        images = [image for image in aligned_images if image is not None]
        if len(images) < cls.MIN_SAMPLES:
            return None, {"status": "INSUFFICIENT_SAMPLES", "sample_count": len(images)}
        shape = images[0].shape[:2]
        images = [image for image in images if image.shape[:2] == shape]
        if len(images) < cls.MIN_SAMPLES:
            return None, {"status": "SHAPE_MISMATCH", "sample_count": len(images)}
        binaries = [cls._clean_binary(image) for image in images]
        votes = np.sum([(binary > 0).astype(np.uint16) for binary in binaries], axis=0)
        threshold = max(2, int(math.ceil(len(binaries) * float(vote_ratio))))
        stable = ((votes >= threshold) * 255).astype("uint8")
        count, labels, stats, _ = cv2.connectedComponentsWithStats(stable, connectivity=8)
        cleaned = np.zeros_like(stable)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= 3:
                cleaned[labels == label] = 255
        ink_pixels = int(cv2.countNonZero(cleaned))
        return cleaned, {
            "status": "READY" if ink_pixels else "EMPTY",
            "sample_count": len(binaries),
            "vote_ratio": float(vote_ratio),
            "vote_threshold": threshold,
            "printed_ink_pixels": ink_pixels,
            "template_method": "binary_vote_plus_brightness_percentile",
            "brightness_percentile": cls.BRIGHTNESS_PERCENTILE,
        }

    @classmethod
    def build(cls, teacher_paths: Sequence[Path], student_documents: Sequence[Sequence[Path]],
              output_dir: Path, max_samples: int = DEFAULT_MAX_SAMPLES) -> Dict[str, Any]:
        import cv2
        import numpy as np

        from .ingestion import preprocess_page
        from .registration import register_page

        output_dir = Path(output_dir)
        aligned_root = output_dir / "aligned_inputs"
        templates_root = output_dir / "templates"
        page_images = {index: [] for index in range(1, len(teacher_paths) + 1)}
        samples = []
        selected = list(student_documents)[:max(0, int(max_samples))]
        for sample_index, document in enumerate(selected, 1):
            sample_record = {"sample_index": sample_index, "pages": []}
            for page_index, source in enumerate(document, 1):
                if page_index > len(teacher_paths):
                    break
                normalized = aligned_root / f"sample_{sample_index:03d}" / f"normalized_{page_index:02d}.jpg"
                aligned = aligned_root / f"sample_{sample_index:03d}" / f"page_{page_index:02d}.jpg"
                try:
                    preprocess_page(Path(source), normalized)
                    registered_path, meta = register_page(
                        Path(teacher_paths[page_index - 1]), normalized, aligned
                    )
                    accepted = meta.get("status") == "REGISTERED"
                    image = cv2.imread(str(registered_path)) if accepted else None
                    if image is not None:
                        page_images[page_index].append(image)
                    sample_record["pages"].append({
                        "page_index": page_index, "source": str(source),
                        "accepted": bool(image is not None), "registration": meta,
                    })
                except Exception as exc:
                    sample_record["pages"].append({
                        "page_index": page_index, "source": str(source),
                        "accepted": False, "error": str(exc),
                    })
            samples.append(sample_record)

        pages = []
        templates_root.mkdir(parents=True, exist_ok=True)
        for page_index in range(1, len(teacher_paths) + 1):
            mask, meta = cls.consensus_mask(page_images.get(page_index, []))
            record = {"page_index": page_index, **meta}
            if mask is not None and meta.get("status") == "READY":
                gray_template = cls.percentile_template(
                    page_images.get(page_index, []), mask
                )
                if gray_template is None:
                    gray_template = np.full(mask.shape, 255, dtype=np.uint8)
                    gray_template[mask > 0] = 0
                template = cv2.cvtColor(gray_template, cv2.COLOR_GRAY2BGR)
                template_path = templates_root / f"page_{page_index:02d}.png"
                mask_path = templates_root / f"page_{page_index:02d}_print_mask.png"
                cv2.imwrite(str(template_path), template)
                cv2.imwrite(str(mask_path), mask)
                record.update({"template_path": str(template_path), "mask_path": str(mask_path)})
            pages.append(record)

        ready = sum(page.get("status") == "READY" for page in pages)
        minimum_samples = min(
            (page.get("sample_count", 0) for page in pages if page.get("status") == "READY"),
            default=0,
        )
        confidence = min(0.95, 0.58 + 0.06 * minimum_samples) if ready else 0.0
        audit = {
            "schema_version": "cohort_print_consensus.v1",
            "status": "READY" if ready else "UNAVAILABLE",
            "teacher_page_count": len(teacher_paths),
            "candidate_student_count": len(selected),
            "ready_page_count": ready,
            "max_samples": int(max_samples),
            "confidence": round(confidence, 3),
            "pages": pages,
            "samples": samples,
        }
        manifest_path = output_dir / "consensus_manifest.json"
        audit["manifest_path"] = str(manifest_path)
        atomic_write_json(manifest_path, audit)
        return audit


__all__ = ["CohortConsensusBuilder"]
