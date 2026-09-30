"""Load version-controlled natural-language prompts from package data."""
from functools import lru_cache
from pathlib import Path
import re


PROMPT_DIRECTORY = Path(__file__).resolve().parent / "prompts"
PROMPT_NAME = re.compile(r"^[a-z0-9_]+$")


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Return one UTF-8 Markdown prompt with a single trailing newline."""
    if not PROMPT_NAME.fullmatch(name):
        raise ValueError(f"无效提示词名称: {name}")
    path = PROMPT_DIRECTORY / f"{name}.md"
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"无法读取提示词文件: {path}") from exc
    if not content:
        raise RuntimeError(f"提示词文件为空: {path}")
    return content + "\n"
