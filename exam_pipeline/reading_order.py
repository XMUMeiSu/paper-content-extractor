"""Geometric reading order shared by OCR, slots and answer fragments."""
from statistics import median


def row_order(values, box=lambda value: value.bbox):
    """Cluster overlapping baselines before ordering within a row (xyxy)."""
    rows = []
    valid, absent = [], []
    for value in values:
        b = box(value)
        (valid if b and len(b) == 4 else absent).append(value)
    for value in sorted(valid, key=lambda v: (box(v)[1], box(v)[0])):
        b = box(value)
        center = (b[1] + b[3]) / 2
        height = max(1, b[3] - b[1])
        candidates = []
        for index, row in enumerate(rows):
            centers = [(box(v)[1]+box(v)[3])/2 for v in row]
            heights = [max(1, box(v)[3]-box(v)[1]) for v in row]
            delta = abs(center-median(centers))
            if delta <= .45 * min(height, median(heights)):
                candidates.append((delta, index))
        if candidates:
            rows[min(candidates)[1]].append(value)
        else:
            rows.append([value])
    rows.sort(key=lambda row: median((box(v)[1]+box(v)[3])/2 for v in row))
    return [v for row in rows for v in sorted(row, key=lambda v: box(v)[0])] + absent


def column_groups(blocks, width):
    """Require a persistent empty gutter, never split by text-block center alone.

    Full-width headings may precede columns. Body text crossing the proposed
    gutter invalidates it. Ambiguous layouts retain row order.
    """
    valid = [b for b in blocks if len(b.bbox) == 4]
    if len(valid) < 6:
        return [row_order(blocks)]
    for fraction in (.5, .45, .55, .4, .6):
        split = width*fraction
        left = [b for b in valid if b.bbox[2] < split-width*.015]
        right = [b for b in valid if b.bbox[0] > split+width*.015]
        crossing = [b for b in valid if b not in left and b not in right]
        if len(left) < 3 or len(right) < 3:
            continue
        body_top = max(min(b.bbox[1] for b in left), min(b.bbox[1] for b in right))
        body_bottom = min(max(b.bbox[3] for b in left), max(b.bbox[3] for b in right))
        typical = median(max(1, b.bbox[3]-b.bbox[1]) for b in valid)
        if body_bottom-body_top < typical*4:
            continue
        if any(b.bbox[3] > body_top+typical for b in crossing):
            continue
        # A row of choices/figure labels is not a text column.
        if min(sum(len(b.text.strip()) >= 8 for b in side) for side in (left, right)) < 3:
            continue
        return [row_order(crossing), row_order(left), row_order(right)]
    return [row_order(blocks)]


def ordered_slots(slots):
    result = []
    for page in sorted({s.page_index for s in slots}):
        result.extend(row_order([s for s in slots if s.page_index == page],
            lambda s: ([s.expected_bbox[1], s.expected_bbox[0], s.expected_bbox[3], s.expected_bbox[2]]
                       if len(s.expected_bbox or []) == 4 else [])))
    return result
