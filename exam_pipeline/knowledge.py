"""Versioned, bounded runtime view of the grounding Markdown knowledge base."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class KnowledgeRule:
    rule_id: str
    title: str
    scope: str
    summary: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    source_headings: Tuple[str, ...] = ()


@dataclass
class GroundingPolicy:
    version: str = "grounding-kb.v1"
    source_path: str = ""
    source_sha256: str = ""
    rules: List[KnowledgeRule] = field(default_factory=list)

    def parameter(self, name: str, default: Any = None) -> Any:
        for rule in self.rules:
            if name in rule.parameters:
                return rule.parameters[name]
        return default

    def rule_ids(self, scope: Optional[str] = None) -> List[str]:
        return [rule.rule_id for rule in self.rules
                if scope is None or rule.scope in {scope, "shared"}]

    def prompt_context(self, scopes: Sequence[str] = ("prompt", "grounding", "slot"),
                       limit: int = 10) -> str:
        selected = [rule for rule in self.rules if rule.scope in set(scopes) | {"shared"}][:limit]
        if not selected:
            return ""
        lines = ["视觉定位知识库约束（只约束物理证据，不得据此脑补答案）："]
        lines.extend(f"- [{rule.rule_id}] {rule.summary}" for rule in selected)
        return "\n".join(lines)

    def audit(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "active_rules": self.rule_ids(),
        }


class GroundingKnowledgeBase:
    """Compile human-readable badcases into a safe deterministic policy.

    The compiler intentionally uses a curated allowlist. Coordinates, scoring
    decisions, UI styling, timing claims and unbounded prose never become
    executable runtime instructions.
    """

    MAX_BYTES = 2 * 1024 * 1024
    HEADING = re.compile(r"^#{2,4}\s+(.+?)\s*$", re.MULTILINE)

    # canonical id, title keywords, scope, safe summary, deterministic parameters
    BINDINGS = (
        ("center_on_ink", ("真实墨迹优先", "质心紧致居中"), "grounding",
         "答案框以合法槽位内的真实墨迹连通域为中心，不以空白或印刷文字为中心。",
         {"center_on_ink": True}),
        ("stroke_safe_padding", ("零笔画截断", "安全 Padding", "防截断"), "verification",
         "墨迹外包络保留安全呼吸边距，不得切断边缘连续笔画。",
         {"breathing_padding_px": 6, "padding_min_px": 4, "padding_max_px": 8}),
        ("multiline_union", ("多行主观题", "多区域并集", "1对多作答框"), "slot",
         "同一采分点的换行或离散区域保持一个业务 Item，并按自然阅读顺序联合。",
         {"merge_subjective_lines": True}),
        ("printed_stem_isolation", ("题干打印文字剔除", "印刷题干与答题槽位"), "slot",
         "公式括号、题号、选项和印刷题干不是答案槽位；只接受合法空白结构及其内部墨迹。",
         {"exclude_formula_parentheses": True, "max_anonymous_anchor_count": 8}),
        ("column_isolation", ("双栏试卷防穿栏", "防穿栏"), "grounding",
         "先确定所在栏边界，候选框和联合框不得跨越栏间空白。",
         {"column_isolation": True}),
        ("reading_order", ("阅读顺序拓扑", "物理空间绝对定序"), "grounding",
         "区域和文字严格按页码、栏、纵坐标、横坐标的物理顺序排列。",
         {"physical_reading_order": True}),
        ("ink_presence_gate", ("真实存在性", "零假阳性", "真实击中"), "verification",
         "槽位内无足够墨迹时必须输出未作答，不得生成假阳性答案框。",
         {"minimum_ink_pixels": 10}),
        ("template_difference", ("模板配准差分", "全图解耦"), "verification",
         "配准可信时用教师模板抵消印刷背景；配准失败时禁止宣称自动收敛。",
         {"use_registered_template_difference": True,
          "print_alignment_tolerance_px": 1}),
        ("trailing_ink_closure", ("末尾字符连通闭包", "右侧墨迹闭包"), "verification",
         "框边缘存在连续墨迹时继续扩展，直到进入稳定空白谷底。",
         {"trailing_scan_px": 25, "trailing_blank_run_px": 15}),
        ("bounded_iteration", ("迭代判断", "闭环"), "verification",
         "几何框采用有界迭代更新，未收敛或缺少物理证据时升级人工复核。",
         {"max_iterations": 3, "current_weight": 0.35, "detected_weight": 0.65}),
    )

    @classmethod
    def load(cls, path: Path) -> GroundingPolicy:
        resolved = Path(path).expanduser().resolve()
        data = resolved.read_bytes()
        if len(data) > cls.MAX_BYTES:
            raise ValueError(f"grounding knowledge base exceeds {cls.MAX_BYTES} bytes")
        text = data.decode("utf-8")
        headings = [re.sub(r"\s+", " ", value).strip()
                    for value in cls.HEADING.findall(text)]
        rules: List[KnowledgeRule] = []
        for rule_id, keywords, scope, summary, parameters in cls.BINDINGS:
            matched = tuple(heading for heading in headings
                            if any(keyword.casefold() in heading.casefold() for keyword in keywords))
            if matched:
                rules.append(KnowledgeRule(
                    rule_id, matched[0], scope, summary, dict(parameters), matched
                ))
        if not rules:
            raise ValueError("grounding knowledge base contains no recognized visual rules")
        return GroundingPolicy(
            source_path=str(resolved),
            source_sha256=hashlib.sha256(data).hexdigest(),
            rules=rules,
        )


__all__ = ["GroundingKnowledgeBase", "GroundingPolicy", "KnowledgeRule"]
