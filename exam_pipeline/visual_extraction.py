"""VLM-only question localization, slot geometry and answer transcription."""
from __future__ import annotations

import copy
import json
import math
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .contracts import DiagramRef, PageRegion, Slot
from .io_utils import atomic_write_json
from .prompt_loader import load_prompt
from .roi import xyxy_to_yxyx


NORMALIZED_BOX = {
    'type': 'array', 'minItems': 4, 'maxItems': 4,
    'items': {'type': 'number', 'minimum': 0, 'maximum': 1000},
}
VISUAL_REGION_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['answer_bbox', 'transcription', 'legible', 'content_kind', 'confidence'],
    'properties': {
        'answer_bbox': NORMALIZED_BOX,
        'transcription': {'type': 'string'},
        'legible': {'type': 'boolean'},
        'content_kind': {'type': 'string', 'enum': [
            'handwriting', 'printed_answer', 'mixed', 'blank', 'uncertain',
        ]},
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
    },
}
VISUAL_SLOT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['index', 'label', 'anchor_before', 'anchor_after', 'regions', 'confidence'],
    'properties': {
        'index': {'type': 'integer', 'minimum': 1},
        'label': {'type': 'string', 'enum': [
            'choice_response', 'blank_response', 'formula_response',
            'short_response', 'working_response', 'proof_response',
            'drawing_response', 'answer_part',
        ]},
        'anchor_before': {'type': 'string'},
        'anchor_after': {'type': 'string'},
        'regions': {'type': 'array', 'minItems': 0, 'maxItems': 10,
                    'items': VISUAL_REGION_SCHEMA},
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
    },
}
VISUAL_ITEM_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['item_id', 'physical_page_id', 'page_index', 'question_region',
                 'answer_layout', 'slots', 'diagram_regions', 'confidence'],
    'properties': {
        'item_id': {'type': 'string'},
        'physical_page_id': {'type': 'string'},
        'page_index': {'type': 'integer', 'minimum': 1},
        'question_region': NORMALIZED_BOX,
        'answer_layout': {'type': 'string',
                          'enum': ['horizontal', 'vertical', 'single', 'mixed']},
        'slots': {'type': 'array', 'minItems': 1, 'maxItems': 100,
                  'items': VISUAL_SLOT_SCHEMA},
        'diagram_regions': {'type': 'array', 'minItems': 0, 'maxItems': 20,
                            'items': NORMALIZED_BOX},
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
    },
}
VISUAL_PAGE_EXTRACTION_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['items'],
    'properties': {'items': {'type': 'array', 'minItems': 1, 'maxItems': 1000,
                             'items': VISUAL_ITEM_SCHEMA}},
}

GEOMETRY_REGION_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['answer_bbox', 'confidence'],
    'properties': {
        'answer_bbox': NORMALIZED_BOX,
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
    },
}
GEOMETRY_SLOT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['index', 'label', 'anchor_before', 'anchor_after', 'regions', 'confidence'],
    'properties': {
        'index': {'type': 'integer', 'minimum': 1},
        'label': VISUAL_SLOT_SCHEMA['properties']['label'],
        'anchor_before': {'type': 'string'},
        'anchor_after': {'type': 'string'},
        'regions': {'type': 'array', 'minItems': 0, 'maxItems': 10,
                    'items': GEOMETRY_REGION_SCHEMA},
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
    },
}
GEOMETRY_ITEM_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['item_id', 'physical_page_id', 'page_index', 'question_region',
                 'answer_layout', 'slots', 'diagram_regions', 'confidence'],
    'properties': {
        'item_id': {'type': 'string'},
        'physical_page_id': {'type': 'string'},
        'page_index': {'type': 'integer', 'minimum': 1},
        'question_region': NORMALIZED_BOX,
        'answer_layout': VISUAL_ITEM_SCHEMA['properties']['answer_layout'],
        'slots': {'type': 'array', 'minItems': 1, 'maxItems': 100,
                  'items': GEOMETRY_SLOT_SCHEMA},
        'diagram_regions': {'type': 'array', 'minItems': 0, 'maxItems': 20,
                            'items': NORMALIZED_BOX},
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
    },
}
VISUAL_GEOMETRY_PAGE_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['items'],
    'properties': {'items': {'type': 'array', 'minItems': 1, 'maxItems': 1000,
                             'items': GEOMETRY_ITEM_SCHEMA}},
}

TRANSCRIPTION_REGION_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['region_index', 'transcription', 'legible', 'content_kind', 'confidence'],
    'properties': {
        'region_index': {'type': 'integer', 'minimum': 1},
        'transcription': {'type': 'string'},
        'legible': {'type': 'boolean'},
        'content_kind': VISUAL_REGION_SCHEMA['properties']['content_kind'],
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
    },
}
TRANSCRIPTION_SLOT_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['index', 'regions'],
    'properties': {
        'index': {'type': 'integer', 'minimum': 1},
        'regions': {'type': 'array', 'minItems': 0, 'maxItems': 10,
                    'items': TRANSCRIPTION_REGION_SCHEMA},
    },
}
TRANSCRIPTION_ITEM_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['item_id', 'slots'],
    'properties': {
        'item_id': {'type': 'string'},
        'slots': {'type': 'array', 'minItems': 1, 'maxItems': 100,
                  'items': TRANSCRIPTION_SLOT_SCHEMA},
    },
}
VISUAL_TRANSCRIPTION_PAGE_SCHEMA = {
    'type': 'object', 'additionalProperties': False, 'required': ['items'],
    'properties': {'items': {'type': 'array', 'minItems': 1, 'maxItems': 1000,
                             'items': TRANSCRIPTION_ITEM_SCHEMA}},
}


def _box(values):
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError('INVALID_VISUAL_BOX')
    if not all(type(value) in (int, float) and math.isfinite(value) for value in values):
        raise ValueError('INVALID_VISUAL_BOX')
    if not 0 <= values[0] < values[2] <= 1000 or not 0 <= values[1] < values[3] <= 1000:
        raise ValueError('INVALID_VISUAL_BOX')
    return [float(value) for value in values]


def _physical(values, page):
    box = _box(values)
    width, height = page.width or 1654, page.height or 2338
    return [int(round(box[0] * width / 1000)),
            int(round(box[1] * height / 1000)),
            int(round(box[2] * width / 1000)),
            int(round(box[3] * height / 1000))]


def _area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(left, right):
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1]))


def _overlap_over_smaller(left, right):
    return _intersection(left, right) / max(1, min(_area(left), _area(right)))


def _union(boxes):
    return [min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes)]


def _items(package):
    return [item for section in package.sections for question in section.questions
            for item in question.items]


def _question_owner(package):
    return {item.item_id: question for section in package.sections
            for question in section.questions for item in question.items}


def _item_pages(item, page_ids):
    result = [reference.get('page_index')
              for reference in item.quality.get('structure_references', [])
              if reference.get('page_index') in page_ids]
    if not result and item.stem_region and item.stem_region.page_index in page_ids:
        result.append(item.stem_region.page_index)
    if not result:
        result.extend(slot.page_index for slot in item.slots if slot.page_index in page_ids)
    return list(dict.fromkeys(result))


def _focus_crop(page, question_box, output_dir, item_id):
    """Create a deterministic zoom view for a VLM retry, preserving original coordinates."""
    try:
        from PIL import Image
        image = Image.open(page.path)
        width, height = image.size
        left, top, right, bottom = [int(value) for value in question_box]
        pad_x = max(32, int(width * .06))
        pad_y = max(24, int(height * .02))
        crop = (
            max(0, left - pad_x),
            max(0, top - pad_y),
            min(width, right + pad_x),
            height,
        )
        if crop[2] <= crop[0] or crop[3] <= crop[1]:
            return None
        target = Path(output_dir) / 'focus_retries'
        target.mkdir(parents=True, exist_ok=True)
        path = target / '{}_page_{:02d}_focus.jpg'.format(item_id, page.index)
        image.crop(crop).save(path, quality=95)
        return path
    except Exception:
        return None


class VisualExamExtractionService:
    """Read final item/slot geometry and answers directly from original pages."""

    def __init__(self, request, provider='none', max_attempts=2,
                 page_workers=1, batch_items=4):
        self.request = request
        self.provider = provider
        self.max_attempts = max(1, min(3, int(max_attempts)))
        self.page_workers = max(1, int(page_workers or 1))
        self.batch_items = max(1, min(20, int(batch_items or 1)))
        self.metrics = {'page_calls': 0, 'page_retries': 0, 'failed_pages': 0,
                        'partial_pages': 0, 'page_batches': 0,
                        'registration_attempts': 0,
                        'registration_successes': 0,
                        'registration_corrections': 0,
                        'items': 0, 'slots': 0, 'regions': 0}

    @staticmethod
    def _logical_plan(item, student):
        if not student:
            return []
        if item.semantic_slot_plan:
            return copy.deepcopy(item.semantic_slot_plan)
        count = int(item.expected_slot_count or len(item.slots) or 1)
        return [{'slot_id': '{}:slot:{}'.format(item.item_id, index),
                 'index': index, 'label': 'answer_part',
                 'anchor_before': '', 'anchor_after': ''}
                for index in range(1, count + 1)]

    def _page_context(self, package, page, page_items, failures):
        student = package.document_type == 'student'
        owner = _question_owner(package)
        return {
            'document_id': package.exam_id,
            'document_type': package.document_type,
            'physical_page_id': page.physical_page_id,
            'page_index': page.index,
            'width': page.width,
            'height': page.height,
            'items': [{
                'item_id': item.item_id,
                'item_name': item.item_name,
                'item_type': item.item_type,
                'question_id': owner[item.item_id].question_id,
                'question_number': owner[item.item_id].question_num,
                'question_text': item.question_text,
                'logical_slots': self._logical_plan(item, student),
                'expected_slot_count': (item.expected_slot_count if student else None),
                'same_question_item_ids': [sibling.item_id
                                           for sibling in owner[item.item_id].items],
            } for item in page_items],
            'validation_failures': failures,
            'schema': VISUAL_PAGE_EXTRACTION_SCHEMA,
        }

    @staticmethod
    def _template_box(item, page_index, slot_index=None):
        geometry = item.quality.get('template_geometry') or {}
        key = 'question_regions' if slot_index is None else 'slots'
        candidates = [
            record for record in geometry.get(key, [])
            if record.get('page_index') == page_index
            and (slot_index is None or record.get('slot_idx') == slot_index)
            and isinstance(record.get('bbox'), list)
            and len(record['bbox']) == 4
        ]
        return copy.deepcopy(candidates[0]['bbox']) if candidates else None

    @staticmethod
    def _registration_source(page_items, page_index):
        for item in page_items:
            geometry = item.quality.get('template_geometry') or {}
            for source in geometry.get('registration_sources') or []:
                if (source.get('page_index') == page_index
                        and source.get('page_path')):
                    return source
        return None

    @staticmethod
    def _estimate_homography(source, page):
        """Estimate a teacher-to-student projective transform with RANSAC."""
        audit = {
            'status': 'UNAVAILABLE',
            'source_page': str((source or {}).get('page_path') or ''),
            'student_page': str(page.path),
        }
        if not source or not source.get('page_path'):
            audit['reason'] = 'TEACHER_REGISTRATION_SOURCE_MISSING'
            return None, audit
        try:
            import cv2
            import numpy as np

            teacher = cv2.imread(str(source['page_path']), cv2.IMREAD_GRAYSCALE)
            student = cv2.imread(str(page.path), cv2.IMREAD_GRAYSCALE)
            if teacher is None or student is None:
                raise ValueError('REGISTRATION_IMAGE_UNREADABLE')

            def scaled(image):
                height, width = image.shape[:2]
                factor = min(1.0, 1400.0 / max(height, width))
                if factor < 1.0:
                    image = cv2.resize(
                        image, (round(width * factor), round(height * factor)),
                        interpolation=cv2.INTER_AREA)
                return image, factor

            teacher_small, teacher_scale = scaled(teacher)
            student_small, student_scale = scaled(student)
            if hasattr(cv2, 'SIFT_create'):
                detector = cv2.SIFT_create(nfeatures=5000)
                norm = cv2.NORM_L2
                method = 'SIFT_RANSAC'
            else:
                detector = cv2.ORB_create(nfeatures=7000)
                norm = cv2.NORM_HAMMING
                method = 'ORB_RANSAC'
            teacher_points, teacher_desc = detector.detectAndCompute(teacher_small, None)
            student_points, student_desc = detector.detectAndCompute(student_small, None)
            if teacher_desc is None or student_desc is None:
                raise ValueError('REGISTRATION_FEATURES_MISSING')
            matches = cv2.BFMatcher(norm).knnMatch(teacher_desc, student_desc, k=2)
            good = [left for left, right in matches if left.distance < .72 * right.distance]
            if len(good) < 16:
                raise ValueError('REGISTRATION_MATCHES_INSUFFICIENT')
            teacher_xy = np.float32([
                teacher_points[match.queryIdx].pt for match in good
            ]).reshape(-1, 1, 2)
            student_xy = np.float32([
                student_points[match.trainIdx].pt for match in good
            ]).reshape(-1, 1, 2)
            small_h, mask = cv2.findHomography(
                teacher_xy, student_xy, cv2.RANSAC, 4.0)
            if small_h is None or mask is None:
                raise ValueError('REGISTRATION_HOMOGRAPHY_UNRESOLVED')
            inliers = int(mask.ravel().sum())
            inlier_ratio = inliers / max(1, len(good))
            if inliers < 12 or inlier_ratio < .35:
                raise ValueError('REGISTRATION_INLIERS_INSUFFICIENT')

            teacher_to_small = np.array([
                [teacher_scale, 0, 0], [0, teacher_scale, 0], [0, 0, 1],
            ], dtype='float64')
            student_to_small = np.array([
                [student_scale, 0, 0], [0, student_scale, 0], [0, 0, 1],
            ], dtype='float64')
            homography = np.linalg.inv(student_to_small) @ small_h @ teacher_to_small
            height, width = teacher.shape[:2]
            corners = np.float32([
                [0, 0], [width - 1, 0],
                [width - 1, height - 1], [0, height - 1],
            ]).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
            student_height, student_width = student.shape[:2]
            warped_area = abs(float(cv2.contourArea(warped.astype('float32'))))
            area_ratio = warped_area / max(1.0, student_width * student_height)
            margin_x, margin_y = student_width * .35, student_height * .35
            if (not .35 <= area_ratio <= 2.2
                    or warped[:, 0].min() < -margin_x
                    or warped[:, 0].max() > student_width + margin_x
                    or warped[:, 1].min() < -margin_y
                    or warped[:, 1].max() > student_height + margin_y):
                raise ValueError('REGISTRATION_TRANSFORM_IMPLAUSIBLE')
            audit.update({
                'status': 'REGISTERED', 'method': method,
                'matches': len(good), 'inliers': inliers,
                'inlier_ratio': round(inlier_ratio, 4),
                'warped_page_area_ratio': round(area_ratio, 4),
                'homography_teacher_to_student': homography.tolist(),
            })
            return homography, audit
        except Exception as exc:
            audit.update(status='UNAVAILABLE', reason=str(exc))
            return None, audit

    @staticmethod
    def _transform_template_box(box, source, page, homography):
        if not box or homography is None:
            return None
        try:
            import cv2
            import numpy as np
            teacher_width = int(source.get('width') or 0)
            teacher_height = int(source.get('height') or 0)
            if teacher_width <= 0 or teacher_height <= 0:
                return None
            left, top, right, bottom = _box(box)
            points = np.float32([
                [left * teacher_width / 1000, top * teacher_height / 1000],
                [right * teacher_width / 1000, top * teacher_height / 1000],
                [right * teacher_width / 1000, bottom * teacher_height / 1000],
                [left * teacher_width / 1000, bottom * teacher_height / 1000],
            ]).reshape(-1, 1, 2)
            warped = cv2.perspectiveTransform(points, homography).reshape(-1, 2)
            width, height = page.width or 1654, page.height or 2338
            result = [
                max(0.0, min(1000.0, float(warped[:, 0].min()) * 1000 / width)),
                max(0.0, min(1000.0, float(warped[:, 1].min()) * 1000 / height)),
                max(0.0, min(1000.0, float(warped[:, 0].max()) * 1000 / width)),
                max(0.0, min(1000.0, float(warped[:, 1].max()) * 1000 / height)),
            ]
            return _box(result)
        except Exception:
            return None

    def _registered_template_box(self, item, page, source, homography,
                                 slot_index=None):
        return self._transform_template_box(
            self._template_box(item, page.index, slot_index),
            source, page, homography)

    def _validate_or_correct_student_frame(self, page, page_items, accepted, student):
        """Use a registered teacher template only for proven page-frame failure."""
        if not student or not accepted:
            return accepted
        self.metrics['registration_attempts'] = int(
            self.metrics.get('registration_attempts', 0)) + 1
        source = self._registration_source(page_items, page.index)
        homography, registration = self._estimate_homography(source, page)
        if homography is not None:
            self.metrics['registration_successes'] = int(
                self.metrics.get('registration_successes', 0)) + 1
        compared = aligned = 0
        by_id = {item.item_id: item for item in page_items}
        for item_id, entry in accepted.items():
            item = by_id.get(item_id)
            if item is None:
                continue
            for slot_data in entry.get('slots', []):
                reference = (
                    self._registered_template_box(
                        item, page, source, homography,
                        int(slot_data.get('index') or 0))
                    if homography is not None else self._template_box(
                        item, page.index, int(slot_data.get('index') or 0))
                )
                if not reference:
                    continue
                regions = slot_data.get('regions') or []
                if not regions:
                    continue
                compared += 1
                if any(_overlap_over_smaller(region['answer_bbox'], reference) >= .15
                       for region in regions):
                    aligned += 1
        result = copy.deepcopy(accepted)
        mismatch = compared >= 4 and aligned / compared < .40
        if compared < 4:
            for entry in result.values():
                entry['_coordinate_validation'] = {
                    'status': 'UNVERIFIED',
                    'reason': 'INSUFFICIENT_REGISTERED_TEMPLATE_COMPARISONS',
                    'compared_slots': compared,
                    'aligned_slots': aligned,
                    'registration': copy.deepcopy(registration),
                }
            return result
        if mismatch and homography is not None:
            self.metrics['registration_corrections'] = int(
                self.metrics.get('registration_corrections', 0)) + 1
            for item_id, entry in result.items():
                item = by_id.get(item_id)
                if item is None:
                    continue
                question = self._registered_template_box(
                    item, page, source, homography)
                if question:
                    entry['question_region'] = question
                replacements = []
                for slot_data in entry.get('slots', []):
                    reference = self._registered_template_box(
                        item, page, source, homography,
                        int(slot_data.get('index') or 0))
                    if not reference or not slot_data.get('regions'):
                        continue
                    original = copy.deepcopy(slot_data['regions'])
                    for region in slot_data['regions']:
                        region['answer_bbox'] = copy.deepcopy(reference)
                    replacements.append({
                        'slot_index': slot_data.get('index'),
                        'original_regions': original,
                        'registered_bbox': copy.deepcopy(reference),
                    })
                entry['_coordinate_recovery'] = {
                    'reason': 'HOMOGRAPHY_REGISTERED_TEMPLATE_CORRECTION',
                    'compared_slots': compared, 'aligned_slots': aligned,
                    'registration': copy.deepcopy(registration),
                    'replacements': replacements,
                }
            return result
        if not mismatch:
            for entry in result.values():
                entry['_coordinate_validation'] = {
                    'status': 'REGISTRATION_ALIGNED' if homography is not None else 'UNVERIFIED',
                    'reason': ('REGISTERED_TEMPLATE_AGREEMENT' if homography is not None
                               else registration.get('reason', 'REGISTRATION_UNAVAILABLE')),
                    'compared_slots': compared, 'aligned_slots': aligned,
                    'registration': copy.deepcopy(registration),
                }
            return result
        for entry in result.values():
            entry['_coordinate_validation'] = {
                'status': 'STUDENT_LOCAL_RETAINED',
                'reason': 'TEACHER_TEMPLATE_MISMATCH_REGISTRATION_UNAVAILABLE',
                'compared_slots': compared,
                'aligned_slots': aligned,
                'coordinate_authority': 'vlm_original_page_pixels',
                'template_used_for_coordinates': False,
                'registration': copy.deepcopy(registration),
            }
        return result

    @staticmethod
    def _validate(raw, page, page_items, student, sibling_groups=()):
        if not isinstance(raw, dict) or set(raw) != {'items'} or not isinstance(raw['items'], list):
            raise ValueError('INVALID_VISUAL_EXTRACTION_RESPONSE')
        expected = {item.item_id: item for item in page_items}
        found = {}
        for entry in raw['items']:
            fields = {'item_id', 'physical_page_id', 'page_index', 'question_region',
                      'answer_layout', 'slots', 'diagram_regions', 'confidence'}
            if not isinstance(entry, dict) or set(entry) != fields:
                raise ValueError('INVALID_VISUAL_ITEM_FIELDS')
            item_id = entry['item_id']
            if item_id not in expected or item_id in found:
                raise ValueError('UNKNOWN_OR_DUPLICATE_VISUAL_ITEM')
            if type(entry['page_index']) is not int or entry['page_index'] != page.index:
                raise ValueError('VISUAL_PAGE_ID_MISMATCH')
            # Page identity is owned by the request context.  A model may echo
            # a stale/short identifier, but it cannot move an item to another
            # page because page_index is still checked above.
            if (not isinstance(entry['physical_page_id'], str)
                    or entry['physical_page_id'] != page.physical_page_id):
                entry = copy.deepcopy(entry)
                entry['physical_page_id'] = page.physical_page_id
            _box(entry['question_region'])
            if entry['answer_layout'] not in {'horizontal', 'vertical', 'single', 'mixed'}:
                raise ValueError('INVALID_VISUAL_ANSWER_LAYOUT')
            confidence = entry['confidence']
            if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError('INVALID_VISUAL_ITEM_CONFIDENCE')
            slots = entry['slots']
            if not isinstance(slots, list) or not slots:
                raise ValueError('INVALID_VISUAL_SLOT_COUNT')
            item = expected[item_id]
            required_count = int(item.expected_slot_count or len(item.semantic_slot_plan) or 0)
            if required_count and len(slots) != required_count:
                raise ValueError('TEACHER_TOPOLOGY_CONFLICT')
            kind = str(item.item_type or '').casefold()
            if (any(token in kind for token in ('choice', 'large_writing', 'solve',
                                                'proof', 'essay', 'calculation'))
                    and len(slots) != 1):
                raise ValueError('SINGLE_RESPONSE_REQUIRED')
            for position, slot in enumerate(slots, 1):
                required = {'index', 'label', 'anchor_before', 'anchor_after',
                            'regions', 'confidence'}
                if not isinstance(slot, dict) or set(slot) != required:
                    raise ValueError('INVALID_VISUAL_SLOT_FIELDS')
                if type(slot['index']) is not int or slot['index'] != position:
                    raise ValueError('INVALID_VISUAL_SLOT_ORDER')
                if slot['label'] not in {
                        'choice_response', 'blank_response', 'formula_response',
                        'short_response', 'working_response', 'proof_response',
                        'drawing_response', 'answer_part'}:
                    raise ValueError('INVALID_VISUAL_SLOT_LABEL')
                if not all(isinstance(slot[key], str)
                           for key in ('anchor_before', 'anchor_after')):
                    raise ValueError('INVALID_VISUAL_SLOT_ANCHORS')
                if (type(slot['confidence']) not in (int, float)
                        or not math.isfinite(slot['confidence'])
                        or not 0 <= slot['confidence'] <= 1):
                    raise ValueError('INVALID_VISUAL_SLOT_CONFIDENCE')
                if (not isinstance(slot['regions'], list)
                        or len(slot['regions']) > 10):
                    raise ValueError('INVALID_VISUAL_SLOT_REGIONS')
                for region in slot['regions']:
                    required_region = {'answer_bbox', 'transcription', 'legible',
                                       'content_kind', 'confidence'}
                    if not isinstance(region, dict) or set(region) != required_region:
                        raise ValueError('INVALID_VISUAL_REGION_FIELDS')
                    _box(region['answer_bbox'])
                    if (not isinstance(region['transcription'], str)
                            or type(region['legible']) is not bool
                            or region['content_kind'] not in {
                                'handwriting', 'printed_answer', 'mixed', 'blank', 'uncertain'}
                            or type(region['confidence']) not in (int, float)
                            or not math.isfinite(region['confidence'])
                            or not 0 <= region['confidence'] <= 1):
                        raise ValueError('INVALID_VISUAL_REGION_VALUE')
            if not isinstance(entry['diagram_regions'], list) or len(entry['diagram_regions']) > 20:
                raise ValueError('INVALID_VISUAL_DIAGRAM_REGIONS')
            for diagram in entry['diagram_regions']:
                _box(diagram)
            found[item_id] = copy.deepcopy(entry)
        if set(found) != set(expected):
            raise ValueError('MISSING_VISUAL_ITEMS')

        # Keep teacher content when a later printed item appears above an
        # earlier one, but make the coordinate uncertainty visible downstream.
        if not student:
            centers = [
                (found[item.item_id]['question_region'][1]
                 + found[item.item_id]['question_region'][3]) / 2
                for item in page_items
            ]
            if any(current + 15 < previous
                   for previous, current in zip(centers, centers[1:])):
                for entry in found.values():
                    entry['_coordinate_validation'] = {
                        'status': 'UNVERIFIED',
                        'reason': 'TEACHER_QUESTION_ORDER_CONFLICT',
                        'coordinate_authority': 'vlm_original_page_pixels',
                    }

        # Keep sibling overlap as diagnostic evidence.  A full-page VLM can
        # place adjacent long-answer strokes on a shared baseline; rejecting
        # the entire page would discard otherwise valid independent answers.
        siblings_by_id = {}
        for group in sibling_groups:
            for item_id in group:
                siblings_by_id[item_id] = set(group)
        for left_index, left_item in enumerate(page_items):
            left_boxes = [_physical(region['answer_bbox'], page)
                          for slot in found[left_item.item_id]['slots']
                          for region in slot['regions']]
            for right_item in page_items[left_index + 1:]:
                if right_item.item_id not in siblings_by_id.get(left_item.item_id, set()):
                    continue
                right_boxes = [_physical(region['answer_bbox'], page)
                               for slot in found[right_item.item_id]['slots']
                               for region in slot['regions']]
                for left in left_boxes:
                    for right in right_boxes:
                        ratio = _intersection(left, right) / max(1, min(_area(left), _area(right)))
                        if ratio > .60:
                            # The caller still receives the original regions;
                            # downstream quality reports can inspect geometry.
                            continue
        return found

    @staticmethod
    def _batch_record(batch_index, batch_count, page_items):
        return {
            'batch_index': batch_index,
            'batch_count': batch_count,
            'batch_item_ids': [item.item_id for item in page_items],
        }

    @staticmethod
    def _token_limit_failure(exc):
        return (getattr(exc, 'code', '') == 'OUTPUT_INCOMPLETE'
                or '模型输出未完成' in str(exc))

    def _validate_geometry(self, raw, page, page_items, student, sibling_groups):
        if not isinstance(raw, dict) or not isinstance(raw.get('items'), list):
            raise ValueError('INVALID_GEOMETRY_RESPONSE')
        expanded = copy.deepcopy(raw)
        for entry in expanded['items']:
            for slot in entry.get('slots') or []:
                for region in slot.get('regions') or []:
                    region.update(transcription='', legible=False,
                                  content_kind='uncertain')
        return self._validate(expanded, page, page_items, student, sibling_groups)

    @staticmethod
    def _transcription_context(package, page, page_items, geometry, failures):
        return {
            'document_id': package.exam_id,
            'document_type': package.document_type,
            'physical_page_id': page.physical_page_id,
            'page_index': page.index,
            'items': [
                {
                    'item_id': item.item_id,
                    'question_text': item.question_text,
                    'geometry': {
                        'question_region': geometry[item.item_id]['question_region'],
                        'slots': [
                            {
                                'index': slot['index'],
                                'label': slot['label'],
                                'regions': [region['answer_bbox']
                                            for region in slot.get('regions', [])],
                            }
                            for slot in geometry[item.item_id]['slots']
                        ],
                    },
                }
                for item in page_items
            ],
            'validation_failures': failures,
        }

    @staticmethod
    def _merge_transcription(raw, geometry, page_items):
        if not isinstance(raw, dict) or set(raw) != {'items'}:
            raise ValueError('INVALID_TRANSCRIPTION_RESPONSE')
        expected = {item.item_id for item in page_items}
        entries = raw.get('items')
        if not isinstance(entries, list):
            raise ValueError('INVALID_TRANSCRIPTION_ITEMS')
        by_id = {}
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {'item_id', 'slots'}:
                raise ValueError('INVALID_TRANSCRIPTION_ITEM')
            item_id = entry['item_id']
            if item_id not in expected or item_id in by_id:
                raise ValueError('UNKNOWN_OR_DUPLICATE_TRANSCRIPTION_ITEM')
            target = geometry[item_id]
            slots = entry['slots']
            if not isinstance(slots, list) or len(slots) != len(target['slots']):
                raise ValueError('TRANSCRIPTION_SLOT_COUNT_CONFLICT')
            for position, (reported, target_slot) in enumerate(
                    zip(slots, target['slots']), 1):
                if (not isinstance(reported, dict)
                        or set(reported) != {'index', 'regions'}
                        or reported['index'] != position):
                    raise ValueError('INVALID_TRANSCRIPTION_SLOT')
                regions = reported['regions']
                target_regions = target_slot.get('regions') or []
                if not isinstance(regions, list) or len(regions) != len(target_regions):
                    raise ValueError('TRANSCRIPTION_REGION_COUNT_CONFLICT')
                for region_index, (result, target_region) in enumerate(
                        zip(regions, target_regions), 1):
                    fields = {'region_index', 'transcription', 'legible',
                              'content_kind', 'confidence'}
                    if (not isinstance(result, dict) or set(result) != fields
                            or result['region_index'] != region_index):
                        raise ValueError('INVALID_TRANSCRIPTION_REGION')
                    target_region.update({
                        'transcription': str(result['transcription']),
                        'legible': bool(result['legible']),
                        'content_kind': result['content_kind'],
                        'confidence': float(result['confidence']),
                    })
            by_id[item_id] = True
        if set(by_id) != expected:
            raise ValueError('MISSING_TRANSCRIPTION_ITEMS')

    def _extract_transcription_batches(self, package, page, page_items,
                                       geometry, attempts):
        batches = [page_items[index:index + self.batch_items]
                   for index in range(0, len(page_items), self.batch_items)]
        self.metrics['page_batches'] += len(batches)
        for batch_index, batch_items in enumerate(batches, 1):
            batch_record = {
                'mode': 'transcription_only_after_token_limit',
                **self._batch_record(batch_index, len(batches), batch_items),
            }
            failures = []
            accepted = False
            for attempt in range(1, self.max_attempts + 1):
                transcription_context = self._transcription_context(
                    package, page, batch_items, geometry, failures)
                prompt = (load_prompt('batch_transcription')
                          + json.dumps(transcription_context, ensure_ascii=False))
                self.metrics['page_calls'] += 1
                self.metrics['page_retries'] += 1
                record = {**batch_record, 'attempt': attempt,
                          'failures_in': copy.deepcopy(failures)}
                try:
                    raw_text = self.request(
                        prompt, [Path(page.path)], VISUAL_TRANSCRIPTION_PAGE_SCHEMA)
                    self._merge_transcription(raw_text, geometry, batch_items)
                    record.update(status='ACCEPTED', response=copy.deepcopy(raw_text))
                    accepted = True
                    attempts.append(record)
                    break
                except Exception as exc:
                    failures = [{'reason': str(exc), 'page_index': page.index}]
                    record.update(status='FAILED', reason=str(exc),
                                  error_type=type(exc).__name__)
                    attempts.append(record)
            if not accepted:
                # Keep compact full-page geometry, but leave transcription
                # uncertain so final quality gates require review.
                continue
        return geometry

    def _extract_page_whole_first(self, package, page, page_items,
                                  sibling_groups):
        attempts = []
        geometry = None
        failures = []
        for attempt in range(1, self.max_attempts + 1):
            context = self._page_context(package, page, page_items, failures)
            geometry_prompt = (load_prompt('whole_page_geometry')
                               + json.dumps(context, ensure_ascii=False))
            record = {'attempt': attempt, 'mode': 'whole_page_geometry'}
            self.metrics['page_calls'] += 1
            if attempt > 1:
                self.metrics['page_retries'] += 1
            try:
                raw_geometry = self.request(
                    geometry_prompt, [Path(page.path)],
                    VISUAL_GEOMETRY_PAGE_SCHEMA)
                geometry = self._validate_geometry(
                    raw_geometry, page, page_items,
                    package.document_type == 'student', sibling_groups)
                record.update(status='ACCEPTED', response=copy.deepcopy(raw_geometry))
                attempts.append(record)
                break
            except Exception as exc:
                reason = str(exc) if isinstance(exc, ValueError) else 'VISUAL_EXTRACTION_BACKEND_ERROR'
                record.update(status='FAILED', reason=reason,
                              error_type=type(exc).__name__)
                attempts.append(record)
                failures = [{'reason': reason, 'page_index': page.index}]
        if geometry is None:
            return None, attempts

        failures = []
        for attempt in range(1, self.max_attempts + 1):
            transcription_context = self._transcription_context(
                package, page, page_items, geometry, failures)
            prompt = (load_prompt('whole_page_transcription')
                      + json.dumps(transcription_context, ensure_ascii=False))
            record = {'attempt': attempt, 'mode': 'whole_page_transcription'}
            self.metrics['page_calls'] += 1
            if attempt > 1:
                self.metrics['page_retries'] += 1
            try:
                raw_text = self.request(
                    prompt, [Path(page.path)], VISUAL_TRANSCRIPTION_PAGE_SCHEMA)
                self._merge_transcription(raw_text, geometry, page_items)
                record.update(status='ACCEPTED', response=copy.deepcopy(raw_text))
                attempts.append(record)
                return geometry, attempts
            except Exception as exc:
                reason = str(exc) if isinstance(exc, ValueError) else 'VISUAL_EXTRACTION_BACKEND_ERROR'
                record.update(status='FAILED', reason=reason,
                              error_type=type(exc).__name__)
                attempts.append(record)
                if self._token_limit_failure(exc):
                    return (self._extract_transcription_batches(
                        package, page, page_items, geometry, attempts), attempts)
                failures = [{'reason': reason, 'page_index': page.index}]
        return geometry, attempts

    def _extract_parallel(self, package, pages, output_dir):
        """Run page requests concurrently and apply accepted pages in order.

        Each worker receives an isolated package copy.  Model calls and
        validation therefore cannot race on shared item state; the original
        package is updated only by the ordered reduction below.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        page_map = {page.index: page for page in pages}
        page_ids = set(page_map)
        all_items = _items(package)
        sibling_groups = [[item.item_id for item in question.items]
                          for section in package.sections
                          for question in section.questions]
        by_page = {page.index: [] for page in pages}
        teacher_expected = {}
        for item in all_items:
            for page_index in _item_pages(item, page_ids):
                by_page.setdefault(page_index, []).append(item)
            expected_by_slot = {}
            for semantic in item.semantic_slot_plan:
                value = semantic.get('expected_text')
                key = str(semantic.get('slot_id') or '')
                if key and value is not None:
                    expected_by_slot.setdefault(key, []).append(str(value))
            for old_slot in item.slots:
                value = old_slot.expected_text
                if value is None:
                    continue
                key = old_slot.semantic_id or '{}:slot:{}'.format(
                    item.item_id, old_slot.slot_idx)
                expected_by_slot.setdefault(key, [])
                if str(value) not in expected_by_slot[key]:
                    expected_by_slot[key].append(str(value))
            teacher_expected[item.item_id] = {
                key: '\n'.join(values) for key, values in expected_by_slot.items()
            }
            item.slots = []
            item.stem_region = None
            item.answer_regions = []
            item.student_regions = []
            item.option_regions = []
            item.blank_regions = []
            item.writing_regions = []
            item.diagrams = []
            item.quality['localization'] = {
                'status': 'PENDING_VLM', 'contexts': [], 'evidence': [],
                'coordinate_source': 'vlm_original_page',
            }
            if package.document_type == 'teacher':
                item.semantic_slot_plan = []
                item.expected_slot_count = None
                item.standard_answer = None
            item.slot_semantics_audit = {
                'provider': self.provider,
                'mode': 'vlm_direct_geometry_and_transcription',
                'status': 'FALLBACK',
                'coordinate_authority': 'vlm_original_page_pixels',
                'attempts': [],
            }

        def run_page(page_index, page_items):
            # The temporary directory prevents page JSON files from colliding
            # while preserving the same persisted page-result contract.
            worker_dir = Path(tempfile.mkdtemp(prefix='visual-page-'))
            try:
                isolated = copy.deepcopy(package)
                wanted = {item.item_id for item in page_items}
                for section in isolated.sections:
                    for question in section.questions:
                        question.items = [item for item in question.items
                                          if item.item_id in wanted]
                        for item in question.items:
                            # The parent package has already cleared runtime
                            # geometry before workers start. Restore a routing
                            # anchor so the isolated service assigns the item
                            # to this physical page.
                            if item.stem_region is None:
                                item.stem_region = PageRegion(
                                    page_index, page_map[page_index].path,
                                    [0, 0, 1, 1], None, item.question_text)
                    section.questions = [question for question in section.questions
                                         if question.items]
                service = VisualExamExtractionService(
                    self.request, self.provider, self.max_attempts,
                    page_workers=1, batch_items=self.batch_items)
                service.extract(isolated, [page_map[page_index]], worker_dir)
                result_path = worker_dir / 'page_{:02d}.json'.format(page_index)
                if not result_path.is_file():
                    candidates = sorted(worker_dir.glob('page_*.json'))
                    if candidates:
                        result_path = candidates[0]
                if not result_path.is_file():
                    raise RuntimeError(
                        'VISUAL_PAGE_RESULT_MISSING:{}'.format(page_index))
                page_result = json.loads(result_path.read_text(encoding='utf-8'))
                return page_index, page_result, service.metrics
            finally:
                shutil.rmtree(worker_dir, ignore_errors=True)

        page_results = {}
        page_metrics = {}
        with ThreadPoolExecutor(max_workers=self.page_workers,
                                thread_name_prefix='exam-page') as pool:
            futures = {
                pool.submit(run_page, page_index, page_items): page_index
                for page_index, page_items in sorted(by_page.items())
                if page_items
            }
            for future in as_completed(futures):
                page_index, page_result, metrics = future.result()
                page_results[page_index] = page_result
                page_metrics[page_index] = metrics

        totals = {'page_calls': 0, 'page_retries': 0, 'failed_pages': 0,
                  'partial_pages': 0, 'page_batches': 0,
                  'registration_attempts': 0,
                  'registration_successes': 0,
                  'registration_corrections': 0,
                  'items': len(all_items), 'slots': 0, 'regions': 0}
        for page_index in sorted(page_results):
            page = page_map[page_index]
            page_items = by_page[page_index]
            page_result = page_results[page_index]
            accepted = page_result.get('items') or None
            attempts = page_result.get('attempts') or []
            if accepted is None:
                for item in page_items:
                    item.slot_semantics_audit['attempts'].extend(copy.deepcopy(attempts))
            else:
                self._apply_page(package, page, page_items, accepted, attempts,
                                 teacher_expected)
            atomic_write_json(output_dir / 'page_{:02d}.json'.format(page_index),
                              page_result)
            metrics = page_metrics[page_index]
            for key in ('page_calls', 'page_retries', 'failed_pages',
                        'partial_pages', 'page_batches',
                        'registration_attempts', 'registration_successes',
                        'registration_corrections'):
                totals[key] += int(metrics.get(key, 0) or 0)
        self._finalize_items(package, pages)
        totals['slots'] = sum(len(item.slots) for item in all_items)
        totals['regions'] = sum(bool(slot.expected_bbox)
                                for item in all_items for slot in item.slots)
        totals['coordinate_authority'] = 'vlm_original_page_pixels'
        totals['ocr_used'] = False
        totals['accepted_items'] = sum(
            item.slot_semantics_audit.get('status') == 'ACCEPTED'
            for item in all_items
        )
        totals['partial_items'] = sum(
            item.slot_semantics_audit.get('status') == 'PARTIAL'
            for item in all_items
        )
        self.metrics = dict(totals)
        atomic_write_json(output_dir / 'summary.json', totals)
        return totals

    def extract(self, package, pages, output_dir):
        if self.request is None:
            raise ValueError('VISUAL_EXTRACTION_BACKEND_UNAVAILABLE')
        if self.page_workers > 1:
            return self._extract_parallel(package, pages, output_dir)
        self.metrics = {'page_calls': 0, 'page_retries': 0, 'failed_pages': 0,
                        'partial_pages': 0, 'page_batches': 0,
                        'registration_attempts': 0,
                        'registration_successes': 0,
                        'registration_corrections': 0,
                        'items': 0, 'slots': 0, 'regions': 0}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        page_map = {page.index: page for page in pages}
        page_ids = set(page_map)
        all_items = _items(package)
        sibling_groups = [[item.item_id for item in question.items]
                          for section in package.sections for question in section.questions]
        by_page = {page.index: [] for page in pages}
        teacher_expected = {}
        for item in all_items:
            for page_index in _item_pages(item, page_ids):
                by_page.setdefault(page_index, []).append(item)
            expected_by_slot = {}
            for semantic in item.semantic_slot_plan:
                value = semantic.get('expected_text')
                key = str(semantic.get('slot_id') or '')
                if key and value is not None:
                    expected_by_slot.setdefault(key, []).append(str(value))
            for old_slot in item.slots:
                value = old_slot.expected_text
                if value is None:
                    continue
                key = old_slot.semantic_id or '{}:slot:{}'.format(
                    item.item_id, old_slot.slot_idx)
                expected_by_slot.setdefault(key, [])
                if str(value) not in expected_by_slot[key]:
                    expected_by_slot[key].append(str(value))
            teacher_expected[item.item_id] = {
                key: '\n'.join(values) for key, values in expected_by_slot.items()
            }
            item.slots = []
            item.stem_region = None
            item.answer_regions = []
            item.student_regions = []
            item.option_regions = []
            item.blank_regions = []
            item.writing_regions = []
            item.diagrams = []
            item.quality['localization'] = {
                'status': 'PENDING_VLM', 'contexts': [], 'evidence': [],
                'coordinate_source': 'vlm_original_page',
            }
            if package.document_type == 'teacher':
                item.semantic_slot_plan = []
                item.expected_slot_count = None
                item.standard_answer = None
            item.slot_semantics_audit = {
                'provider': self.provider,
                'mode': 'vlm_direct_geometry_and_transcription',
                'status': 'FALLBACK',
                'coordinate_authority': 'vlm_original_page_pixels',
                'attempts': [],
            }

        page_results = {}
        for page_index, page_items in sorted(by_page.items()):
            if not page_items:
                continue
            page = page_map[page_index]
            accepted, attempts = self._extract_page_whole_first(
                package, page, page_items, sibling_groups)
            accepted = self._validate_or_correct_student_frame(
                page, page_items, accepted,
                package.document_type == 'student',
            )
            page_results[page.index] = {'status': 'ACCEPTED' if accepted else 'UNRESOLVED',
                                        'attempts': attempts, 'items': accepted or {}}
            if accepted is None:
                self.metrics['failed_pages'] += 1
                for item in page_items:
                    item.slot_semantics_audit['attempts'].extend(copy.deepcopy(attempts))
                atomic_write_json(output_dir / 'page_{:02d}.json'.format(page.index),
                                  page_results[page.index])
                continue
            partial = len(accepted) != len(page_items)
            page_results[page.index]['status'] = 'PARTIAL' if partial else 'ACCEPTED'
            if partial:
                self.metrics.setdefault('partial_pages', 0)
                self.metrics['partial_pages'] += 1
            self._apply_page(package, page, page_items, accepted, attempts,
                             teacher_expected)
            page_results[page.index]['items'] = accepted
            atomic_write_json(output_dir / 'page_{:02d}.json'.format(page.index),
                              page_results[page.index])

        self._finalize_items(package, pages)
        self.metrics['items'] = len(all_items)
        self.metrics['slots'] = sum(len(item.slots) for item in all_items)
        self.metrics['regions'] = sum(bool(slot.expected_bbox)
                                      for item in all_items for slot in item.slots)
        summary = {**self.metrics,
                   'accepted_items': sum(item.slot_semantics_audit.get('status') == 'ACCEPTED'
                                         for item in all_items),
                   'partial_items': sum(item.slot_semantics_audit.get('status') == 'PARTIAL'
                                        for item in all_items),
                   'coordinate_authority': 'vlm_original_page_pixels',
                   'ocr_used': False}
        atomic_write_json(output_dir / 'summary.json', summary)
        return summary

    def _apply_page(self, package, page, page_items, accepted, attempts,
                    teacher_expected):
        student = package.document_type == 'student'
        for item in page_items:
            if item.item_id not in accepted:
                item.slot_semantics_audit['attempts'].extend(copy.deepcopy(attempts))
                item.slot_semantics_audit['status'] = 'PARTIAL'
                continue
            entry = accepted[item.item_id]
            coordinate_recovery = entry.get('_coordinate_recovery') or {}
            recovered_coordinates = bool(coordinate_recovery)
            coordinate_validation = entry.get('_coordinate_validation') or {}
            unverified_coordinates = coordinate_validation.get('status') in {
                'UNVERIFIED', 'STUDENT_LOCAL_RETAINED',
            }
            item.confidence = float(entry['confidence'])
            question_box = _physical(entry['question_region'], page)
            item.stem_region = PageRegion(
                page.index, page.path, question_box, entry['confidence'],
                item.question_text,
                coordinate_role=('homography_registered_teacher_question_region'
                                 if recovered_coordinates else 'vlm_question_region'))
            localization = item.quality.setdefault('localization', {
                'status': 'VLM_LOCALIZED', 'contexts': [], 'evidence': [],
                'coordinate_source': 'vlm_original_page',
            })
            localization['status'] = 'VLM_LOCALIZED'
            localization['coordinate_source'] = (
                'homography_registered_teacher_template'
                if recovered_coordinates else 'vlm_original_page')
            localization['contexts'] = [context for context in localization.get('contexts', [])
                                        if context.get('page_index') != page.index]
            localization['contexts'].append({
                'page_index': page.index, 'bbox': question_box,
                'question_context': question_box,
                'answer_search_domain': question_box,
                'status': ('TEMPLATE_RECOVERED_LOCALIZATION'
                           if recovered_coordinates else 'VLM_FINAL_LOCALIZATION'),
                'coordinate_role': ('homography_registered_teacher_question_region'
                                    if recovered_coordinates
                                    else 'vlm_final_question_region'),
                'layout_axis': entry['answer_layout'],
            })
            localization['evidence'] = [evidence for evidence in localization.get('evidence', [])
                                        if evidence.get('page_index') != page.index]
            localization['evidence'].append({
                'page_index': page.index, 'status': 'VLM_FINAL_LOCALIZATION',
                'confidence': entry['confidence']})
            item.slot_semantics_audit['attempts'].extend(copy.deepcopy(attempts))
            item.slot_semantics_audit['answer_layout'] = entry['answer_layout']
            item.slot_semantics_audit.setdefault('proposals', []).append(copy.deepcopy(entry))
            if recovered_coordinates:
                item.slot_semantics_audit['coordinate_recovery'] = copy.deepcopy(
                    coordinate_recovery)

            if not student and not item.semantic_slot_plan:
                item.semantic_slot_plan = [{
                    'slot_id': '{}:slot:{}'.format(item.item_id, slot['index']),
                    'index': slot['index'], 'label': slot['label'],
                    'anchor_before': slot['anchor_before'],
                    'anchor_after': slot['anchor_after'],
                } for slot in entry['slots']]
                item.expected_slot_count = len(entry['slots'])
                item.slot_count_source = 'teacher_vlm_direct'
            plan = self._logical_plan(item, student) or item.semantic_slot_plan

            page_answer_boxes = []
            for slot_data in entry['slots']:
                slot_index = slot_data['index']
                semantic = (plan[slot_index - 1] if slot_index <= len(plan) else {
                    'slot_id': '{}:slot:{}'.format(item.item_id, slot_index),
                    'anchor_before': slot_data['anchor_before'],
                    'anchor_after': slot_data['anchor_after'],
                })
                if not slot_data['regions']:
                    semantic_id = str(semantic.get('slot_id') or
                                      '{}:slot:{}'.format(item.item_id, slot_index))
                    expected_text = teacher_expected.get(item.item_id, {}).get(semantic_id)
                    blank_slot = Slot(
                        slot_index, 'semantic_region', item.item_id, [],
                        page.index, semantic_id=semantic_id,
                        anchor_before=str(semantic.get('anchor_before') or
                                          slot_data['anchor_before']),
                        anchor_after=str(semantic.get('anchor_after') or
                                         slot_data['anchor_after']),
                        geometry_status='MISSING', content_status='BLANK',
                    )
                    blank_slot.expected_text = expected_text
                    blank_slot.student_answer = None if student else None
                    blank_slot.recognized_text = ''
                    blank_slot.status = 'VLM_BLANK'
                    blank_slot.review_status = 'AUTO_PASS'
                    blank_slot.audit = {
                        'semantic_source': 'vlm_direct',
                        'semantic_label': slot_data['label'],
                        'coordinate_authority': 'vlm_original_page_pixels',
                        'topology_source': 'student_self' if student else 'teacher_vlm',
                        'geometry_mode': 'vlm_original_page',
                        'blank_confirmed': True,
                        'recognition': {
                            'text': '', 'status': 'BLANK',
                            'source': 'vlm_original_page',
                            'content_kind': 'blank', 'legible': True,
                            'confidence': slot_data['confidence'],
                            'issues': [], 'warnings': ['NO_VISIBLE_ANSWER_REGION'],
                            'agreement_type': 'VLM_DIRECT',
                            'attempts': [{'source': 'vlm_original_page',
                                          'page_index': page.index}],
                        },
                    }
                    item.slots.append(blank_slot)
                    continue
                for region_index, region in enumerate(slot_data['regions'], 1):
                    box = _physical(region['answer_bbox'], page)
                    page_answer_boxes.append(box)
                    confidence = min(float(slot_data['confidence']), float(region['confidence']))
                    warnings = []
                    if recovered_coordinates:
                        warnings.append('HOMOGRAPHY_REGISTERED_TEMPLATE_CORRECTION')
                    if unverified_coordinates:
                        warnings.append(str(coordinate_validation.get('reason')
                                            or 'UNVERIFIED_COORDINATE_GEOMETRY'))
                    if confidence < .60:
                        warnings.append('LOW_VLM_GEOMETRY_CONFIDENCE')
                    if region['content_kind'] in {'mixed', 'uncertain'}:
                        warnings.append('VLM_CONTENT_KIND_' + region['content_kind'].upper())
                    legible = bool(region['legible'])
                    text = str(region['transcription'] or '').strip()
                    blank = region['content_kind'] == 'blank' and not text
                    content_status = ('BLANK' if blank else 'RECOGNIZED'
                                      if legible and text else 'VLM_UNCERTAIN')
                    geometry_status = ('UNCERTAIN' if unverified_coordinates
                                       else 'ALIGNED_WITH_WARNING' if warnings else 'ALIGNED')
                    slot = Slot(
                        slot_index, 'semantic_region', item.item_id,
                        xyxy_to_yxyx(box), page.index,
                        semantic_id=str(semantic.get('slot_id') or
                                        '{}:slot:{}'.format(item.item_id, slot_index)),
                        anchor_before=str(semantic.get('anchor_before') or
                                          slot_data['anchor_before']),
                        anchor_after=str(semantic.get('anchor_after') or
                                         slot_data['anchor_after']),
                        geometry_status=geometry_status,
                        content_status=content_status,
                    )
                    yxyx = xyxy_to_yxyx(box)
                    slot.handwriting_bbox = list(yxyx)
                    slot.evidence_bbox = list(yxyx)
                    slot.recognition_bbox = list(yxyx)
                    slot.recognized_text = text
                    slot.has_ink = bool(text)
                    slot.review_status = ('AUTO_PASS' if content_status in {'RECOGNIZED', 'BLANK'}
                                          else 'NEED_REVIEW')
                    slot.status = ('TEACHER_ANSWER_EXTRACTED' if not student and content_status == 'RECOGNIZED'
                                   else 'VLM_ANSWER_EXTRACTED' if student and content_status == 'RECOGNIZED'
                                   else 'VLM_BLANK' if blank else 'VLM_ANSWER_NEEDS_REVIEW')
                    if student:
                        slot.student_answer = text if content_status == 'RECOGNIZED' else None
                        slot.expected_text = teacher_expected.get(item.item_id, {}).get(
                            slot.semantic_id)
                    else:
                        slot.expected_text = text if content_status == 'RECOGNIZED' else None
                    slot.answer_fragments = [{
                        'page_index': page.index, 'page_file': page.path,
                        'bbox': box, 'text': text,
                    }]
                    slot.geometry_evidence = {
                        'status': geometry_status,
                        'reason': ('HOMOGRAPHY_REGISTERED_TEMPLATE_CORRECTION'
                                   if recovered_coordinates
                                   else 'VLM_ORIGINAL_PAGE_REGION'),
                        'coordinate_authority': (
                            'homography_registered_teacher_template'
                            if recovered_coordinates else 'vlm_original_page_pixels'),
                        'confidence': confidence,
                        'warnings': warnings,
                    }
                    recognition = {
                        'text': text,
                        'status': content_status,
                        'source': 'vlm_original_page',
                        'content_kind': region['content_kind'],
                        'legible': legible,
                        'confidence': region['confidence'],
                        'issues': ([] if content_status in {'RECOGNIZED', 'BLANK'}
                                   else ['VLM_ILLEGIBLE_OR_EMPTY']),
                        'warnings': warnings,
                        'agreement_type': 'VLM_DIRECT',
                        'attempts': [{'source': 'vlm_original_page',
                                      'page_index': page.index}],
                    }
                    slot.audit = {
                        'semantic_source': 'vlm_direct',
                        'semantic_label': slot_data['label'],
                        'coordinate_authority': (
                            'homography_registered_teacher_template'
                            if recovered_coordinates else 'vlm_original_page_pixels'),
                        'topology_source': 'student_self' if student else 'teacher_vlm',
                        'geometry_mode': 'vlm_original_page',
                        'geometry_evidence': copy.deepcopy(slot.geometry_evidence),
                        'recognition': recognition,
                        'visual_region_index': region_index,
                    }
                    if recovered_coordinates:
                        slot.audit['coordinate_recovery'] = copy.deepcopy(
                            coordinate_recovery)
                    if coordinate_validation:
                        slot.audit['coordinate_validation'] = copy.deepcopy(
                            coordinate_validation)
                    item.slots.append(slot)

            if page_answer_boxes:
                answer_box = _union(page_answer_boxes)
                region = PageRegion(page.index, page.path, answer_box, entry['confidence'],
                                    '', coordinate_role='vlm_final_answer_region')
                item.answer_regions.append(copy.deepcopy(region))
                if student:
                    item.student_regions.append(copy.deepcopy(region))
                kind = str(item.item_type or '').casefold()
                if 'choice' in kind:
                    item.option_regions.append(copy.deepcopy(region))
                elif any(token in kind for token in ('fill', 'blank')):
                    item.blank_regions.append(copy.deepcopy(region))
                else:
                    item.writing_regions.append(copy.deepcopy(region))
                localization['contexts'][-1]['answer_search_domain'] = answer_box

            for diagram_index, normalized in enumerate(entry['diagram_regions'], 1):
                diagram_box = _physical(normalized, page)
                item.diagrams.append(DiagramRef(
                    title='{} figure {}'.format(item.item_name or item.item_id, diagram_index),
                    bbox=diagram_box, page_index=page.index,
                    audit={'source': 'vlm_original_page',
                           'coordinate_authority': 'vlm_original_page_pixels'}))

    def _finalize_items(self, package, pages):
        first_page = pages[0].index if pages else 1
        for item in _items(package):
            logical = item.semantic_slot_plan or self._logical_plan(
                item, package.document_type == 'student')
            expected = int(item.expected_slot_count or len(logical) or 1)
            existing = {slot.slot_idx for slot in item.slots}
            for index in range(1, expected + 1):
                if index in existing:
                    continue
                semantic = logical[index - 1] if index <= len(logical) else {}
                page_ids = _item_pages(item, {page.index for page in pages})
                slot = Slot(
                    index, 'semantic_region', item.item_id, [],
                    page_ids[0] if page_ids else first_page,
                    semantic_id=str(semantic.get('slot_id') or
                                    '{}:slot:{}'.format(item.item_id, index)),
                    anchor_before=str(semantic.get('anchor_before') or ''),
                    anchor_after=str(semantic.get('anchor_after') or ''),
                    geometry_status='MISSING', content_status='NOT_EVALUATED')
                slot.status = 'VLM_ANSWER_NEEDS_REVIEW'
                slot.review_status = 'NEED_REVIEW'
                slot.audit = {
                    'topology_source': 'missing_placeholder',
                    'coordinate_authority': 'vlm_original_page_pixels',
                    'reason': 'VLM_SLOT_REGION_MISSING',
                }
                item.slots.append(slot)
            item.slots.sort(key=lambda slot: (slot.slot_idx, slot.page_index,
                                               slot.expected_bbox or []))
            item.expected_slot_count = expected
            item.slot_count_source = item.slot_count_source or 'vlm_direct'
            item.cardinality_evidence = {
                'decision': 'VLM_DIRECT', 'resolved_count': expected,
                'source': 'original_page_visual_model',
            }
            resolved = {
                slot.slot_idx for slot in item.slots
                if slot.expected_bbox or (slot.audit or {}).get('blank_confirmed')
            }
            item.slot_semantics_audit['status'] = ('ACCEPTED' if len(resolved) == expected
                                                   else 'PARTIAL')
            if len(resolved) != expected:
                item.slot_semantics_audit['reason'] = 'VLM_SLOT_REGION_MISSING'
            if package.document_type == 'teacher':
                for semantic in item.semantic_slot_plan:
                    texts = []
                    for slot in item.slots:
                        if (slot.semantic_id == semantic.get('slot_id')
                                and slot.expected_text is not None
                                and str(slot.expected_text) not in texts):
                            texts.append(str(slot.expected_text))
                    semantic['expected_text'] = '\n'.join(texts) if texts else None


def visual_answer_summary(package):
    """Return document-local VLM answer metrics without counting fragments twice."""
    logical = {}
    geometry = {}
    content = {}
    for item in _items(package):
        for slot in item.slots:
            key = slot.semantic_id or '{}:slot:{}'.format(item.item_id, slot.slot_idx)
            logical.setdefault((item.item_id, key), []).append(slot)
            geometry[slot.geometry_status] = geometry.get(slot.geometry_status, 0) + 1
            content[slot.content_status] = content.get(slot.content_status, 0) + 1
    complete = 0
    blank = 0
    unresolved = 0
    for slots in logical.values():
        statuses = {slot.content_status for slot in slots}
        if statuses and statuses <= {'RECOGNIZED'}:
            complete += 1
        elif statuses and statuses <= {'BLANK'}:
            blank += 1
        else:
            unresolved += 1
    return {
        'total_logical_slots': len(logical),
        'answers_extracted': complete,
        'blank_slots': blank,
        'unresolved_slots': unresolved,
        'physical_regions': sum(
            1 for item in _items(package) for slot in item.slots
            for fragment in slot.answer_fragments
        ),
        'geometry': geometry,
        'content': content,
        'coordinate_authority': 'vlm_original_page_pixels',
        'answer_authority': 'vlm_original_page',
        'ocr_used': False,
    }
__all__ = ['VisualExamExtractionService', 'VISUAL_PAGE_EXTRACTION_SCHEMA',
           'visual_answer_summary']
