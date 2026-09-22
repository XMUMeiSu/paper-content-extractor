"""Stable logical slot identities from row order and nearby printed anchors."""
import re
from .reading_order import ordered_slots


def bind_semantic_slots(item, pages):
    slots = ordered_slots(item.slots)
    for index, slot in enumerate(slots, 1):
        slot.slot_idx = index
        logical_index = 1 if slot.slot_type == "free_response" else index
        slot.semantic_id = '{}:slot:{}'.format(item.item_id, logical_index)
        page = pages.get(slot.page_index)
        if not page or len(slot.expected_bbox or []) != 4:
            continue
        y1,x1,y2,x2 = slot.expected_bbox
        center = (y1+y2)/2
        # Neighboring print is a reusable cue; actual recognized answers never
        # participate in identity construction.
        same_row = [b for b in page.ocr if len(b.bbox) == 4
                    and abs((b.bbox[1]+b.bbox[3])/2-center) <= max(18,(y2-y1)*.6)]
        before = [b for b in same_row if b.bbox[2] <= x1+8]
        after = [b for b in same_row if b.bbox[0] >= x2-8]
        if before:
            slot.anchor_before = max(before,key=lambda b:b.bbox[2]).text[-48:]
        if after:
            slot.anchor_after = min(after,key=lambda b:b.bbox[0]).text[:48]
        slot.audit['semantic_anchor'] = {'before':slot.anchor_before,'after':slot.anchor_after,
                                         'source':'printed_neighbors', 'order':'row_then_column'}
    item.slots = slots
